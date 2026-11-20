"""Send clicks to the game and read the player's own input while recording."""
import time

import Quartz

SPACE = 49
UP = 126


class Clicker:
    """Holds / releases the left mouse button over the game window.

    Meltdown is an iOS app running on Apple Silicon so a mouse press is a tap.
    Holding the button keeps the cube jumping, ship flying, etc.
    """

    # GD ignores taps while the cube is in the air unless the button is still down when it
    # lands. Short taps right before landing turn into coin flips with a few ms of jitter,
    # so every press is held at least this long.
    MIN_HOLD = 0.10

    def __init__(self, grabber):
        self.grabber = grabber
        self.down = False
        self.down_at = 0.0

    def _post(self, kind):
        x, y = self.grabber.center()
        ev = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    def set(self, hold):
        """Returns what the button is actually doing after this call."""
        if hold and not self.down:
            self._post(Quartz.kCGEventLeftMouseDown)
            self.down = True
            self.down_at = time.time()
        elif not hold and self.down and time.time() - self.down_at >= self.MIN_HOLD:
            self._post(Quartz.kCGEventLeftMouseUp)
            self.down = False
        return self.down

    def click_at(self, fx, fy):
        """Click at a spot given as a fraction of the game area."""
        self.release()
        r = self.grabber.region
        x, y = r["left"] + fx * r["width"], r["top"] + fy * r["height"]
        for kind in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
            ev = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            time.sleep(0.05)

    def tap(self):
        """One short press and release."""
        self.set(True)
        time.sleep(self.MIN_HOLD)
        self.set(False)

    def release(self):
        """Let go of the button right away, ignoring the minimum hold."""
        if self.down:
            self._post(Quartz.kCGEventLeftMouseUp)
            self.down = False


def human_holding():
    """True if the player is holding mouse, space or up arrow right now."""
    st = Quartz.kCGEventSourceStateCombinedSessionState
    if Quartz.CGEventSourceButtonState(st, Quartz.kCGMouseButtonLeft):
        return True
    return bool(Quartz.CGEventSourceKeyState(st, SPACE) or Quartz.CGEventSourceKeyState(st, UP))
