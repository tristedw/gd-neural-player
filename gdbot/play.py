"""Let the net play. With --learn it keeps practicing until it beats the level.

    python -m gdbot.play --level seven_seas            # just play
    python -m gdbot.play --level seven_seas --learn    # practice and improve

Open the level in Meltdown first (normal or practice mode), then run this.
It clicks inside the game window, so leave the window alone while it runs.
Ctrl+C to stop, the model and data are saved on exit.

How --learn works:
  * every attempt is recorded (frame stack, what the net did, time since spawn)
  * on death, the ~0.5s before the crash is where the mistake was, those frames
    get relabeled to the opposite action
  * everything the net did before that survived, so it is kept as good data,
    with extra weight when the attempt beat the best distance so far
  * the net retrains in a background thread after every death and the new
    weights are swapped in for the next attempt
  * exploration is focused on the spots where it keeps dying
"""
import argparse
import copy
import os
import random
import subprocess
import threading
import time

import numpy as np
import torch

from . import model as M
from .controls import Clicker
from .data import load_all, save_chunk
from .train import level_folders, train
from .vision import DeathDetector, FrameStack, preprocess
from .window import Grabber

MISTAKE_WINDOW = (0.55, 0.03)  # seconds before death that get blamed
REPLAY_CAP = 120_000


def focus_game():
    subprocess.run(["osascript", "-e", 'tell application id "com.robtop.geometrydashmeltdown" to activate'],
                   capture_output=True)
    time.sleep(0.3)


class Learner:
    """Holds the replay data and retrains a copy of the net in the background."""

    def __init__(self, level, net, device):
        self.level = level
        self.device = device
        self.train_net = copy.deepcopy(net).to(device)
        self.lock = threading.Lock()
        self.new_state = None
        self.busy = False
        self.obs = np.zeros((0, *net_input_shape()), dtype=np.uint8)
        self.target = np.zeros(0, dtype=np.float32)
        self.weight = np.zeros(0, dtype=np.float32)
        old = load_all(level_folders(level), max_samples=REPLAY_CAP)
        if old is not None:
            self.obs, self.target, self.weight = old
            print(f"loaded {len(self.obs)} old samples")
        self.pending = []

    def add(self, obs, target, weight):
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
        if self.busy:
            return
        self._merge()
        if len(self.obs) < 500:
            return
        self.busy = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            # train on a random slice so each round stays a couple seconds
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
        with self.lock:
            s, self.new_state = self.new_state, None
        return s


def net_input_shape():
    from .vision import OBS_H, OBS_W, STACK
    return (STACK, OBS_H, OBS_W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="default")
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--explore", type=float, default=0.02, help="base chance per frame to try something different")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-attempts", type=int, default=100000)
    args = ap.parse_args()

    torch.set_num_threads(2)
    model_path = os.path.join("models", f"{args.level}.pt")
    os.makedirs("models", exist_ok=True)
    cpu = torch.device("cpu")
    if os.path.exists(model_path):
        net = M.load(model_path, cpu)
        print(f"loaded {model_path}")
    elif args.learn:
        net = M.DashNet().eval()
        print("no model yet, starting from scratch")
    else:
        raise SystemExit(f"{model_path} not found, run with --learn or train first")

    learner = Learner(args.level, net, M.pick_device()) if args.learn else None
    self_dir = os.path.join("data", args.level, "self")

    g = Grabber()
    clicker = Clicker(g)
    det = DeathDetector()
    stack = FrameStack()
    focus_game()

    run = []            # (t_since_spawn, obs, action)
    deaths = []         # where (seconds since spawn) we have died
    best = 0.0
    attempts = 0
    flip_left = 0
    spawn_t = time.time()
    last_motion = time.time()
    dt = 1.0 / args.fps

    print("playing. ctrl+c to stop")
    try:
        while attempts < args.max_attempts:
            t = time.time()
            frame = g.grab()
            ev = det.update(frame, t)

            if ev == "respawned":
                spawn_t = det.spawn_time
                stack.reset()
                run = []
                if learner:
                    s = learner.take_update()
                    if s is not None:
                        net.load_state_dict(s)

            if ev == "died":
                clicker.release()
                attempts += 1
                died_at = det.death_time - spawn_t
                deaths.append(died_at)
                new_best = died_at > best
                best = max(best, died_at)
                print(f"attempt {attempts}: died at {died_at:6.2f}s  best {best:6.2f}s")
                if learner and run:
                    lo, hi = died_at - MISTAKE_WINDOW[0], died_at - MISTAKE_WINDOW[1]
                    o, tg, w = [], [], []
                    for rt, ob, a in run:
                        if rt >= hi:
                            continue
                        o.append(ob)
                        if rt >= lo:
                            # blame grows the closer we got to the crash
                            frac = (rt - lo) / (hi - lo)
                            tg.append(1.0 - a)
                            w.append(0.4 + 0.8 * frac)
                        else:
                            tg.append(a)
                            w.append(1.0 if new_best else 0.35)
                    learner.add(o, tg, w)
                    save_chunk(self_dir, o, tg, w)
                    learner.kick()
                run = []

            if det.alive:
                last_motion = t
                rel = t - spawn_t
                obs = stack.push(preprocess(frame))
                with torch.no_grad():
                    p = torch.sigmoid(net(torch.from_numpy(obs)[None])).item()
                hold = p > args.threshold
                if learner:
                    # explore harder near places we keep dying
                    near = sum(1 for d in deaths[-30:] if -0.8 < d - rel < 0.15)
                    eps = args.explore * (1 + 3 * near)
                    if flip_left > 0:
                        flip_left -= 1
                        hold = not hold
                    elif random.random() < eps:
                        flip_left = random.randint(2, 14)
                        hold = not hold
                clicker.set(hold)
                run.append((rel, obs, 1.0 if hold else 0.0))
            else:
                clicker.release()
                if t - last_motion > 6 and run == [] and attempts and best > 1:
                    # camera froze and never came back, most likely the level complete screen
                    print(f"no respawn for 6s, looks like the level is done (best run {best:.1f}s)")
                    break

            spare = dt - (time.time() - t)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        clicker.release()
        if learner:
            while learner.busy:
                time.sleep(0.1)
            s = learner.take_update()
            if s is not None:
                net.load_state_dict(s)
        M.save(net, model_path, best=best, attempts=attempts)
        print(f"saved {model_path}  best={best:.2f}s  attempts={attempts}")


if __name__ == "__main__":
    main()
