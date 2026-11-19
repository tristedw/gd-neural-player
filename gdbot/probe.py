"""Debug tool: prints FPS, camera motion and death events so you can tune thresholds.

    python -m gdbot.probe --seconds 15
"""
import argparse
import time

import cv2

from .vision import DeathDetector
from .window import Grabber


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--save", default="", help="save first frame here")
    args = ap.parse_args()

    g = Grabber()
    det = DeathDetector()
    t0 = time.time()
    n = 0
    last_print = t0
    while time.time() - t0 < args.seconds:
        f = g.grab()
        if n == 0 and args.save:
            cv2.imwrite(args.save, f)
        t = time.time()
        ev = det.update(f, t)
        n += 1
        if ev:
            print(f"{t - t0:7.2f}s  {ev}")
        if t - last_print > 0.25:
            print(f"{t - t0:7.2f}s  fps={n / (t - t0):5.1f}  dx={det.last_dx:6.2f}  alive={det.alive}")
            last_print = t


if __name__ == "__main__":
    main()
