"""Main bot loop. Plays the level, and with --learn practices it until it gets through.

    python -m gdbot.play --level seven_seas --learn             # practice mode, learn the level
    python -m gdbot.play --level seven_seas --policy plan       # normal mode, replay the learned run
    python -m gdbot.play --level seven_seas --policy net        # normal mode, pure neural net

Press Esc any time to stop. Model, plan and data are saved on exit.

How --learn works (run it in PRACTICE MODE, from a fresh practice attempt):
  * The bot keeps a plan: the inputs of its furthest run since the last checkpoint.
  * Each attempt it replays the plan up to just before where it died, then explores
    new inputs past that point. Beat the old distance and that run becomes the plan.
  * Once a plan gets a few seconds past the checkpoint, on the next attempt it drops
    a new checkpoint at a spot it already knows it survives, so it never replays
    old sections again.
  * Stuck for a while? It starts exploring earlier. Still stuck, it removes the last
    checkpoint and goes back a segment.
  * Every frame it survived is training data for the CNN, labeled with what it did.
    The net retrains in a background thread after each death.
  * When the level is done the segment plans get stitched into one full run
    (models/<level>_plan.json) for normal mode.
"""
import argparse
import bisect
import copy
import json
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np
import Quartz
import torch

from . import model as M
from .controls import Clicker
from .data import load_all, save_chunk
from .train import level_folders, train
from .vision import OBS_H, OBS_W, STACK, DeathDetector, FrameStack, preprocess
from .window import OWNER_HINTS, ThreadedGrabber, find_window

REPLAY_CAP = 150_000
SURVIVE_MARGIN = 0.5      # frames this close to a death are not trusted as good data
CP_MIN_GAP = 3.0          # plan must reach this far past the checkpoint before we place a new one
CP_BEHIND = 1.0           # place the checkpoint this far behind the plan's furthest point
STUCK_ATTEMPTS = 25
ESC = 53

# buttons, as a fraction of the game area (window minus title bar). Measured on
# Meltdown's layout. The full game's menus may differ, check before relying on them.
PAUSE_RESTART = (0.294, 0.829)   # green framed attempt menu
PAUSE_RESUME = (0.439, 0.582)    # dark pause menu, big play button
CP_PLACE = (0.442, 0.890)        # practice mode, place checkpoint
CP_REMOVE = (0.586, 0.890)       # practice mode, remove checkpoint


class Log:
    """Prints and appends to a log file. Writes straight to the file so a killed process never loses lines."""
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        stamp = time.strftime("%H:%M:%S")
        try:
            print(line, flush=True)
        except Exception:
            pass
        self.f.write(f"{stamp} {line}\n")


def game_in_front():
    """True if the frontmost normal window belongs to the game."""
    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    for info in Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []:
        if info.get("kCGWindowLayer", 0) != 0:
            continue
        owner = str(info.get("kCGWindowOwnerName", "")).lower()
        return any(h in owner for h in OWNER_HINTS)
    return False


def esc_pressed():
    """Esc while the game is focused. Esc in other apps (like your terminal or chat) is ignored."""
    if not Quartz.CGEventSourceKeyState(Quartz.kCGEventSourceStateCombinedSessionState, ESC):
        return False
    return game_in_front()


def _is_lime(px):
    b, g, r = px
    return g > 180 and b < 140 and g > b + 60


def on_attempt_menu(frame):
    """Green framed menu, it has a thick lime frame down both sides."""
    h, w = frame.shape[:2]
    for x0, x1 in ((0.112, 0.124), (0.874, 0.886)):
        p = frame[int(h * 0.3):int(h * 0.6), int(w * x0):int(w * x1)]
        if not _is_lime(p.reshape(-1, 3).mean(axis=0)):
            return False
    return True


def on_pause_menu(frame):
    """Dark pause menu, big lime play button in the middle."""
    h, w = frame.shape[:2]
    cx, cy = int(w * PAUSE_RESUME[0]), int(h * PAUSE_RESUME[1])
    p = frame[cy - 8:cy + 8, cx - 40:cx - 20]
    return _is_lime(p.reshape(-1, 3).mean(axis=0))


def focus_game():
    """Bring the game to the front. GD pauses itself when it loses focus."""
    win = find_window()
    if win is None:
        return
    name = str(win["owner"]).replace('"', "")
    subprocess.run(["osascript", "-e", f'tell application "{name}" to activate'], capture_output=True)
    time.sleep(0.3)


class Learner:
    """Holds the replay data and retrains a copy of the net in the background."""

    def __init__(self, level, net, device, log):
        self.device = device
        self.train_net = copy.deepcopy(net).to(device)
        self.lock = threading.Lock()
        self.new_state = None
        self.busy = False
        self.obs = np.zeros((0, STACK, OBS_H, OBS_W), dtype=np.uint8)
        self.target = np.zeros(0, dtype=np.float32)
        self.weight = np.zeros(0, dtype=np.float32)
        old = load_all(level_folders(level), max_samples=REPLAY_CAP)
        if old is not None:
            self.obs, self.target, self.weight = old
            log(f"loaded {len(self.obs)} old samples")
        self.pending = []

    def add(self, obs, target, weight):
        """Queue new samples. They get merged in on the next kick()."""
        self.pending.append((obs, target, weight))

    def _merge(self):
        if not self.pending:
            return
        o = np.concatenate([(np.asarray(p[0]) * 255).astype(np.uint8) for p in self.pending])
        t = np.concatenate([np.asarray(p[1], np.float32) for p in self.pending])
        w = np.concatenate([np.asarray(p[2], np.float32) for p in self.pending])
        self.pending = []
        self.obs = np.concatenate([self.obs, o])[-REPLAY_CAP:]
        self.target = np.concatenate([self.target, t])[-REPLAY_CAP:]
        self.weight = np.concatenate([self.weight, w])[-REPLAY_CAP:]

    def kick(self):
        """Start a background training round if one isn't already running."""
        if self.busy:
            return
        self._merge()
        if len(self.obs) < 500:
            return
        self.busy = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            n = min(len(self.obs), 25_000)
            recent = min(len(self.obs), 6_000)
            idx = np.concatenate([
                np.random.randint(0, len(self.obs), n - recent),
                np.arange(len(self.obs) - recent, len(self.obs)),
            ])
            train(self.train_net, self.obs[idx], self.target[idx], self.weight[idx],
                  self.device, epochs=1, log=False)
            state = {k: v.detach().cpu() for k, v in self.train_net.state_dict().items()}
            with self.lock:
                self.new_state = state
        finally:
            self.busy = False

    def take_update(self):
        """New weights from the last finished round, or None."""
        with self.lock:
            s, self.new_state = self.new_state, None
        return s

    def wait(self):
        """Block until the current training round is done."""
        while self.busy:
            time.sleep(0.1)


class Plan:
    """Inputs over time since (re)spawn. Lookup returns the input at time rel."""

    def __init__(self, steps=None):
        self.steps = steps or []   # sorted list of (rel, hold)
        self._keys = [s[0] for s in self.steps]

    def at(self, rel):
        """Input state at time rel (0 or 1)."""
        i = bisect.bisect_right(self._keys, rel) - 1
        return self.steps[i][1] if i >= 0 else 0.0

    def shifted(self, offset):
        """Plan for the part after offset, with times relative to offset. Used when a checkpoint is placed."""
        return Plan([(r - offset, a) for r, a in self.steps if r >= offset])

    def to_json(self):
        """Plain list for saving."""
        return self.steps

    def events(self):
        """Only the moments the input changes, as (rel, hold)."""
        out, last = [], 0.0
        for r, a in self.steps:
            if a != last:
                out.append((r, a))
                last = a
        return out


class Replayer:
    """Plays a plan back by events so short taps never get skipped.

    Sampling the plan per frame can miss a 2 frame tap if this run's frames land
    differently. Here every press in the plan always reaches the game, even if
    its release time already passed by the next frame.
    """

    def __init__(self, plan):
        self.ev = plan.events()
        self.i = 0
        self.hold = False
        self.force_release = False

    def step(self, rel):
        """Input to send at time rel."""
        if self.force_release:
            self.force_release = False
            self.hold = False
        pressed = False
        while self.i < len(self.ev) and self.ev[self.i][0] <= rel:
            v = self.ev[self.i][1] > 0.5
            if v:
                pressed = True
            self.hold = v
            self.i += 1
        if pressed and not self.hold:
            # press and release both fell in this frame, press now, let go next frame
            self.hold = True
            self.force_release = True
        return self.hold


class Segment:
    """One stretch of the level between two checkpoints, with its own plan and search state."""
    def __init__(self, plan=None, best=0.0, cp_at=0.0):
        self.plan = plan or Plan()
        self.best = best        # furthest survival time since this segment's checkpoint
        self.cp_at = cp_at      # where in the previous segment this checkpoint was placed
        self.back = 0.5         # how far before best we start exploring
        self.stuck = 0
        self.fails = []         # where replays of the plan died before reaching best
        self.patch = None       # (start, end) window to re-explore inside the plan
        self.patch_tries = 0

    def note_replay_fail(self, d):
        """Plan died at d before its own best. Two fails in the same spot -> patch it."""
        self.fails = [f for f in self.fails[-10:] if abs(f - d) < 0.35] + [d]
        if self.patch is None and len(self.fails) >= 2:
            self.patch = (max(0.0, d - 0.7), max(0.0, d - 0.05))
            self.patch_tries = 0
            return True
        return False

    def finish_patch(self, run_steps, d):
        """Called on death while a patch is active. Returns True if the patch worked."""
        s, e = self.patch
        self.patch_tries += 1
        if d > e + 0.45:
            # got past the bad spot: new inputs up to the end of the window, old plan after
            head = [(r, a) for r, a in run_steps if r < e]
            tail = [(r, a) for r, a in self.plan.steps if r >= e]
            self.plan = Plan(head + tail)
            self.patch = None
            self.fails = []
            return True
        if self.patch_tries >= 30:
            # widen the window, the real mistake is earlier
            self.patch = (max(0.0, s - 0.5), e)
            self.patch_tries = 0
        return False

    def in_patch(self, rel):
        """True if rel is inside the window being re-explored."""
        return self.patch is not None and self.patch[0] <= rel < self.patch[1]


class Explorer:
    """Random but GD shaped input: short taps most of the time, sometimes long holds."""

    def __init__(self):
        self.left = 0
        self.val = False

    def step(self, net_hold):
        """Next frame's input. net_hold is what the CNN would do."""
        if self.left > 0:
            self.left -= 1
            return self.val
        r = random.random()
        if r < 0.07:
            self.val = True
            self.left = random.randint(3, 8) if random.random() < 0.7 else random.randint(8, 30)
        elif r < 0.45:
            # follow the net for a bit, it is often right already
            self.val = net_hold
            self.left = random.randint(3, 10)
        else:
            self.val = False
            self.left = random.randint(1, 6)
        return self.val


class GridExplorer:
    """Systematic search for the input that gets past an obstacle.

    GD is deterministic and the fix for a death is almost always one input in the
    half second before it. So instead of random noise, try candidates in order:
      cube style: one tap at a time before the death point, most likely lead times first
      ship/wave style: holds over a grid of start times and lengths
    Candidates are tried once each for a given death point. If none work the caller
    falls back to random exploration.
    """

    TAP = 0.05
    TAP_LEADS = [0.30, 0.25, 0.35, 0.20, 0.40, 0.15, 0.45, 0.10, 0.50, 0.55, 0.05, 0.60]
    HOLD_LEADS = [0.30, 0.45, 0.15, 0.60, 0.75]
    HOLD_LENS = [0.15, 0.30, 0.50, 0.80, 1.20]

    def __init__(self):
        self.key = None
        self.cands = []
        self.cur = None

    def _build(self, death, start, ship_like):
        leads = list(self.TAP_LEADS)
        extra = 0.65
        while death - extra >= start - 0.05:
            leads.append(round(extra, 2))
            extra += 0.05
        taps = [[(death - lead, self.TAP)] for lead in leads if death - lead >= start - 0.05]
        holds = [[(death - lead, ln)] for lead in self.HOLD_LEADS for ln in self.HOLD_LENS
                 if death - lead >= start - 0.05]
        # two taps, for double obstacles
        doubles = [[(death - a, self.TAP), (death - b, self.TAP)]
                   for a, b in ((0.75, 0.30), (0.6, 0.25), (0.9, 0.35), (0.5, 0.2))
                   if death - a >= start - 0.3]
        nothing = [[]]
        if ship_like:
            return holds + nothing + taps + doubles
        return taps + nothing + doubles + holds

    def next_attempt(self, death, start, ship_like):
        """Pick the candidate for this attempt. Returns False if the grid is used up."""
        key = (round(death, 2), round(start, 2))
        if key != self.key:
            self.key = key
            self.cands = self._build(death, start, ship_like)
        if not self.cands:
            self.cur = None
            return False
        self.cur = self.cands.pop(0)
        return True

    def hold(self, rel):
        """Input for the current candidate at time rel."""
        if self.cur is None:
            return False
        return any(s <= rel < s + ln for s, ln in self.cur)


def ship_like(plan, death):
    """Long holds right before the death point usually mean ship or wave."""
    held, last_on = 0.0, None
    for r, a in plan.steps:
        if r < death - 2.0 or r > death:
            continue
        if a > 0.5 and last_on is None:
            last_on = r
        elif a < 0.5 and last_on is not None:
            held = max(held, r - last_on)
            last_on = None
    return held >= 0.25


def stitch(segments):
    """Join segment plans into one plan from the level start."""
    steps = []
    offset = 0.0
    for i, seg in enumerate(segments):
        end = segments[i + 1].cp_at if i + 1 < len(segments) else float("inf")
        steps += [(offset + r, a) for r, a in seg.plan.steps if r < end]
        if i + 1 < len(segments):
            offset += end
    return Plan(steps)


def main():
    """Parse args, set everything up and run the play / learn loop until Esc."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="default")
    ap.add_argument("--learn", action="store_true", help="practice mode learning")
    ap.add_argument("--policy", choices=["net", "plan"], default="net", help="what plays when not learning")
    ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-checkpoints", action="store_true", help="learn without placing checkpoints")
    ap.add_argument("--resume", action="store_true",
                    help="practice mode: pick up where the last --learn run stopped (game checkpoints must still be there)")
    ap.add_argument("--rebuild", action="store_true",
                    help="practice mode, fresh run: replay saved segments and re-place their checkpoints, then keep learning")
    ap.add_argument("--polish", action="store_true",
                    help="normal mode: start from the saved full plan and patch the spots where it fails")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    log = Log(f"bot_{args.level}.log")
    torch.set_num_threads(2)
    model_path = os.path.join("models", f"{args.level}.pt")
    plan_path = os.path.join("models", f"{args.level}_plan.json")
    cpu = torch.device("cpu")
    if os.path.exists(model_path):
        net = M.load(model_path, cpu)
        log(f"loaded {model_path}")
    else:
        net = M.DashNet().eval()
        if not args.learn and args.policy == "net":
            sys.exit(f"{model_path} not found, run --learn first")

    full_plan = None
    if not args.learn and args.policy == "plan":
        if not os.path.exists(plan_path):
            sys.exit(f"{plan_path} not found, run --learn first")
        full_plan = Plan([tuple(s) for s in json.load(open(plan_path))["steps"]])

    learner = Learner(args.level, net, M.pick_device(), log) if args.learn else None
    self_dir = os.path.join("data", args.level, "self")

    g = ThreadedGrabber()
    clicker = Clicker(g)
    det = DeathDetector()
    stack = FrameStack()
    explorer = Explorer()
    focus_game()

    segments = [Segment()]
    practice_path = os.path.join("models", f"{args.level}_practice.json")
    future = []   # saved segments still to be re-placed (--rebuild)
    if args.learn and (args.resume or args.rebuild) and os.path.exists(practice_path):
        segments = []
        for sd in json.load(open(practice_path)):
            s = Segment(Plan([tuple(x) for x in sd["steps"]]), sd["best"], sd["cp_at"])
            segments.append(s)
        log(f"loaded {len(segments) - 1} checkpoints, ~{sum(s.cp_at for s in segments) + segments[-1].best:.1f}s in")
        if args.rebuild:
            future = segments[1:]
            segments = segments[:1]
            log("rebuilding checkpoints from the start")
    if args.learn and args.polish:
        if not os.path.exists(plan_path):
            sys.exit(f"{plan_path} not found, run --learn in practice mode first")
        saved = json.load(open(plan_path))
        segments = [Segment(Plan([tuple(s) for s in saved["steps"]]), best=saved["length"])]
        args.no_checkpoints = True
        log(f"polishing saved plan ({saved['length']:.1f}s) in normal mode")
    run = []                 # (rel, obs, hold) this attempt
    attempts = 0
    spawn_t = time.time()
    last_motion = time.time()
    last_fix = 0.0
    cp_pending = None        # rel at which to drop a checkpoint this attempt
    anchored = False         # did this attempt start with a respawn we actually saw
    replay = Replayer(Plan())
    grid = GridExplorer()
    use_grid = False
    finished = False
    dt = 1.0 / args.fps

    def seg():
        """The segment we are currently playing."""
        return segments[-1]

    def total_progress():
        """Rough distance into the level in seconds, summed over all segments."""
        allsegs = segments + future
        return sum(s.cp_at for s in allsegs) + allsegs[-1].best

    log(f"playing ({'learn' if args.learn else args.policy}). press Esc to stop")
    steps = []          # exact input changes this attempt, (rel, hold)
    last_actual = False
    net_hold = False
    expl_hold = False
    obs = None
    try:
        while True:
            now = time.time()
            if esc_pressed():
                log("esc pressed, stopping")
                break

            # ---- vision, only when a new frame came in ----
            got = g.poll()
            if got is not None:
                frame, t = got
                ev = det.update(frame, t)

                if ev == "respawned":
                    spawn_t = det.spawn_time
                    anchored = det.anchored
                    stack.reset()
                    run = []
                    steps = []
                    explorer.left = 0
                    cp_pending = None
                    replay = Replayer(seg().plan if learner else (full_plan or Plan()))
                    use_grid = False
                    if learner and anchored:
                        sg = seg()
                        if sg.best > 0 and sg.patch is None:
                            use_grid = grid.next_attempt(sg.best, sg.best - sg.back, ship_like(sg.plan, sg.best))
                    if learner:
                        s = learner.take_update()
                        if s is not None:
                            net.load_state_dict(s)
                        sg = seg()
                        if anchored and future:
                            cp_pending = future[0].cp_at
                        elif (anchored and not args.no_checkpoints and sg.best - CP_BEHIND >= CP_MIN_GAP):
                            cp_pending = sg.best - CP_BEHIND

                if ev == "died":
                    clicker.release()
                    last_actual = False
                    attempts += 1
                    died_at = det.death_time - spawn_t
                    if learner and not anchored:
                        log(f"attempt {attempts}: started mid run, not using it for the plan")
                    elif learner:
                        sg = seg()
                        if sg.patch is not None and sg.finish_patch(steps, died_at):
                            log(f"  patched the plan, got to {died_at:.2f}s")
                        if died_at > sg.best + 0.05:
                            sg.best = died_at
                            sg.plan = Plan(list(steps))
                            sg.stuck = 0
                            sg.back = 0.5
                        elif died_at < sg.best - sg.back - 0.1 and sg.patch is None:
                            if sg.note_replay_fail(died_at):
                                log(f"  replay keeps dying at {died_at:.2f}s, patching {sg.patch[0]:.2f}-{sg.patch[1]:.2f}s")
                        else:
                            sg.stuck += 1
                            if sg.stuck >= STUCK_ATTEMPTS and not grid.cands:
                                sg.stuck = 0
                                if sg.back < sg.best:
                                    sg.back = min(sg.back + 0.75, sg.best)
                                    log(f"  stuck, exploring from {sg.best - sg.back:.2f}s")
                                elif len(segments) > 1:
                                    clicker.click_at(*CP_REMOVE)
                                    old = segments.pop()
                                    prev = seg()
                                    prev.back = prev.best - old.cp_at + 0.75
                                    prev.stuck = 0
                                    log("  stuck on this checkpoint, removed it")
                        good = [(o, a) for r, o, a in run if r < died_at - SURVIVE_MARGIN]
                        if good:
                            o, a = zip(*good)
                            learner.add(list(o), list(a), [1.0] * len(a))
                            save_chunk(self_dir, list(o), list(a), [1.0] * len(a))
                        learner.kick()
                    log(f"attempt {attempts}: died at {died_at:6.2f}s  "
                        f"cp {len(segments) - 1}  seg best {seg().best:5.2f}s  total ~{total_progress():6.1f}s")
                    run = []
                    steps = []

                if det.alive:
                    last_motion = t
                    obs = stack.push(preprocess(frame))
                    with torch.no_grad():
                        p = torch.sigmoid(net(torch.from_numpy(obs)[None])).item()
                    net_hold = p > args.threshold
                    expl_hold = explorer.step(net_hold)
                    run.append((t - spawn_t, obs, 1.0 if last_actual else 0.0))
                else:
                    clicker.release()
                    last_actual = False
                    if t - last_motion > 4 and t - last_fix > 4:
                        last_fix = t
                        focus_game()
                        f2 = g.grab()
                        if on_pause_menu(f2):
                            log("paused, resuming")
                            clicker.click_at(*PAUSE_RESUME)
                        elif on_attempt_menu(f2):
                            log("attempt menu, restarting")
                            clicker.click_at(*PAUSE_RESTART)
                        elif t - last_motion > 8 and attempts and seg().best > 3:
                            log("no respawn and no menu, looks like the level is complete")
                            finished = True
                            break

            # ---- input, every ~1ms so planned presses land on time ----
            if det.alive and obs is not None:
                rel = now - spawn_t
                if learner and not anchored:
                    hold = net_hold
                elif learner:
                    sg = seg()
                    if sg.in_patch(rel):
                        replay.step(rel)  # keep the replayer's position in sync
                        hold = expl_hold
                    elif rel < sg.best - sg.back or (future and rel < future[0].cp_at + 0.5):
                        hold = replay.step(rel)
                    elif use_grid:
                        hold = grid.hold(rel)
                    else:
                        hold = expl_hold
                    if cp_pending is not None and rel >= cp_pending:
                        clicker.click_at(*CP_PLACE)
                        last_actual = False
                        if future:
                            new = future.pop(0)
                            new.cp_at = rel
                        else:
                            new = Segment(sg.plan.shifted(rel), sg.best - rel, cp_at=rel)
                        segments.append(new)
                        log(f"  checkpoint {len(segments) - 1} at +{rel:.2f}s")
                        cp_pending = None
                        spawn_t = now
                        run = []
                        steps = []
                        rel = 0.0
                        replay = Replayer(new.plan)
                        hold = replay.step(0.0)
                elif full_plan is not None:
                    hold = replay.step(rel)
                else:
                    hold = net_hold

                actual = clicker.set(hold)
                if actual != last_actual:
                    steps.append((rel, 1.0 if actual else 0.0))
                    last_actual = actual

            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        clicker.release()
        if learner:
            learner.wait()
            s = learner.take_update()
            if s is not None:
                net.load_state_dict(s)
            if not args.polish:
                json.dump([{"steps": s.plan.steps, "best": s.best, "cp_at": s.cp_at} for s in segments + future],
                          open(practice_path, "w"))
            full = stitch(segments + future)
            json.dump({"steps": full.to_json(), "finished": finished,
                       "length": total_progress(), "checkpoints": len(segments) - 1},
                      open(plan_path, "w"))
            log(f"saved {plan_path} ({total_progress():.1f}s, finished={finished})")
        M.save(net, model_path, attempts=attempts)
        log(f"saved {model_path}  attempts={attempts}")


if __name__ == "__main__":
    main()
