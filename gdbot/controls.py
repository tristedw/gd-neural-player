"""Send clicks to the game and read the player's own input while recording."""
import Quartz

SPACE = 49
UP = 126


class Clicker:
    """Holds / releases the left mouse button over the game window.

    Meltdown is an iOS app running on Apple Silicon so a mouse press is a tap.
    Holding the button keeps the cube jumping, ship flying, etc.
    """

    def __init__(self, grabber):
        self.grabber = grabber
        self.down = False

    def _post(self, kind):
        x, y = self.grabber.center()
        ev = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    def set(self, hold):
        if hold and not self.down:
            self._post(Quartz.kCGEventLeftMouseDown)
            self.down = True
        elif not hold and self.down:
            self._post(Quartz.kCGEventLeftMouseUp)
            self.down = False

    def tap(self):
        self.set(True)
        self.set(False)

    def release(self):
        self.set(False)


def human_holding():
    """True if the player is holding mouse, space or up arrow right now."""
    st = Quartz.kCGEventSourceStateCombinedSessionState
    if Quartz.CGEventSourceButtonState(st, Quartz.kCGMouseButtonLeft):
        return True
    return bool(Quartz.CGEventSourceKeyState(st, SPACE) or Quartz.CGEventSourceKeyState(st, UP))
