"""Record yourself playing so the net can learn from you.

    python -m gdbot.record --level seven_seas

Play normally with mouse, space or up arrow. Ctrl+C to stop.
Only the stretch you survived gets kept as good data; the last bit before
each death is dropped (or flipped, see --flip-deaths).
"""
import argparse
import os
import time

from .controls import human_holding
from .vision import DeathDetector, FrameStack, preprocess
from .window import Grabber

DEATH_WINDOW = 0.45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="default")
    ap.add_argument("--fps", type=float, default=60)
    args = ap.parse_args()

    folder = os.path.join("data", args.level, "human")
    g = Grabber()
    det = DeathDetector()
    stack = FrameStack()
    run = []  # (t, obs, action)
    kept = 0
    dt = 1.0 / args.fps
    print("recording, go play. ctrl+c to stop")
    try:
        while True:
            t = time.time()
            frame = g.grab()
            ev = det.update(frame, t)
            if ev == "died":
                keep = [r for r in run if r[0] < det.death_time - DEATH_WINDOW]
                if keep:
                    from .data import save_chunk
                    save_chunk(folder, [r[1] for r in keep], [r[2] for r in keep], [1.0] * len(keep))
                    kept += len(keep)
                print(f"died after {len(run)} frames, kept {len(keep)} (total {kept})")
                run = []
                stack.reset()
            obs = stack.push(preprocess(frame))
            if det.alive:
                run.append((t, obs, 1.0 if human_holding() else 0.0))
            spare = dt - (time.time() - t)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    # finishing a level (or stopping mid run) keeps everything
    if run:
        from .data import save_chunk
        save_chunk(folder, [r[1] for r in run], [r[2] for r in run], [1.0] * len(run))
        kept += len(run)
    print(f"saved {kept} samples to {folder}")


if __name__ == "__main__":
    main()
