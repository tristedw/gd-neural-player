"""Record yourself playing. Gives the net real training data and gives the bot your run as a plan.

    python -m gdbot.record --level seven_seas

Play normally in NORMAL mode with mouse, space or up arrow. Press Esc to stop
(Esc also pauses GD, that's fine).

Saves:
  data/<level>/human/*.npz        frames + what you were holding, for training the net
  models/<level>_plan.json        your furthest run as a timed plan (if it beats what's there)

Attempts only count for the plan once the bot has seen a respawn, so if you start
from the menu just die or restart once first.
"""
import argparse
import json
import os
import time

from .controls import human_holding
from .data import save_chunk
from .play import Log, esc_pressed, on_pause_menu
from .vision import DeathDetector, FrameStack, preprocess
from .window import ThreadedGrabber

DEATH_MARGIN = 0.45   # the last bit before a death is probably a mistake, don't learn from it


def main():
    """Record until Esc, then save training data and the best run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="default")
    ap.add_argument("--fps", type=float, default=60)
    args = ap.parse_args()

    folder = os.path.join("data", args.level, "human")
    os.makedirs("models", exist_ok=True)
    plan_path = os.path.join("models", f"{args.level}_plan.json")
    log = Log(f"record_{args.level}.log")

    old_len = 0.0
    if os.path.exists(plan_path):
        old_len = json.load(open(plan_path)).get("length", 0.0)

    g = ThreadedGrabber()
    det = DeathDetector()
    stack = FrameStack()
    run = []          # (rel, obs, hold)
    spawn_t = time.time()
    anchored = False
    kept = 0
    best = (0.0, None, False)   # (length, steps, finished)
    last_motion = time.time()
    pending = None    # (end, run, anchored, when) waiting to see if it was a death or a finish

    def keep_run(steps_obs, events, end, finished):
        """Save one attempt: frames as training data, and its inputs as the best plan if it went furthest."""
        nonlocal kept, best
        good = [(o, a) for r, o, a in steps_obs if finished or r < end - DEATH_MARGIN]
        if good:
            o, a = zip(*good)
            save_chunk(folder, list(o), list(a), [1.0] * len(a))
            kept += len(a)
        if anchored and end > best[0]:
            best = (end, list(events), finished)

    log("recording, go play. esc to stop")
    steps = []        # exact input changes, (rel, hold), sampled every ~1ms
    holding = False
    try:
        while True:
            now = time.time()
            if esc_pressed():
                break
            got = g.poll()
            if got is not None:
                frame, t = got
                ev = det.update(frame, t)
                if ev == "respawned":
                    if pending:
                        end, prun, psteps, panch, _ = pending
                        pending = None
                        keep_run(prun, psteps, end, False)
                        log(f"died at {end:6.2f}s{'' if panch else ' (not anchored)'}  best {best[0]:.2f}s  samples {kept}")
                    spawn_t = det.spawn_time
                    anchored = det.anchored
                    stack.reset()
                    run = []
                    steps = []
                    holding = False
                if ev == "died":
                    # could be a death or the level complete screen, the next respawn tells us which
                    pending = (det.death_time - spawn_t, run, steps, anchored, t)
                    run = []
                    steps = []
                if det.alive:
                    obs = stack.push(preprocess(frame))
                    run.append((t - spawn_t, obs, 1.0 if holding else 0.0))
                elif pending and t - pending[4] > 3 and not on_pause_menu(frame):
                    # camera stopped and no respawn: level complete screen
                    end, prun, psteps, panch, _ = pending
                    pending = None
                    anchored_now, anchored = anchored, panch
                    keep_run(prun, psteps, end, True)
                    anchored = anchored_now
                    log(f"level complete! run of {end:.2f}s saved")
            if det.alive:
                h = human_holding()
                if h != holding:
                    holding = h
                    steps.append((now - spawn_t, 1.0 if h else 0.0))
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    if pending:
        keep_run(pending[1], pending[2], pending[0], False)
    if run:
        keep_run(run, steps, run[-1][0], False)
    length, steps, finished = best
    if steps and (finished or length > old_len):
        json.dump({"steps": steps, "finished": finished, "length": length, "checkpoints": 0,
                   "source": "human"}, open(plan_path, "w"))
        log(f"saved your {'complete ' if finished else ''}run ({length:.1f}s) to {plan_path}")
    log(f"saved {kept} samples to {folder}")


if __name__ == "__main__":
    main()
