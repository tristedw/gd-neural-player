"""Checks how repeatable input timing is.

Taps once at a fixed time after each respawn and prints where it died. If the same
tap time gives the same death time every attempt, the game and the timing are
deterministic enough for plan replay.

    python -m gdbot.timing_test 0.25,0.30,0.35,0.35,0.35

Run it with the game in front and the level running (normal mode, no input of your own).
"""
import sys
import time

from .controls import Clicker
from .play import focus_game
from .vision import DeathDetector
from .window import ThreadedGrabber


def main():
    """Run the tap schedule given on the command line."""
    times = [float(x) for x in sys.argv[1].split(",")]
    g = ThreadedGrabber()
    c = Clicker(g)
    d = DeathDetector()
    focus_game()
    i = -1
    tapped = True
    spawn = 0.0
    anchored = False
    while i < len(times):
        frame, t = g.grab(), time.time()
        ev = d.update(frame, t)
        if ev == "respawned":
            i += 1
            spawn = d.spawn_time
            anchored = d.anchored
            tapped = False
        if ev == "died" and 0 <= i < len(times):
            print(f"tap {times[i]:.3f}s  died {d.death_time - spawn:.3f}s  anchored {anchored}", flush=True)
        if d.alive and i < len(times) and not tapped and t - spawn >= times[i]:
            c.tap()
            tapped = True
    c.release()


if __name__ == "__main__":
    main()
