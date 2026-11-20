"""Find the Geometry Dash window (Meltdown or the full game) and grab frames from it."""
import time

import cv2
import numpy as np
import Quartz

OWNER_HINTS = ("geometrydashmeltdown", "meltdown", "geometry dash")

# macOS draws a title bar on top of wrapped iOS apps, skip it (points)
TITLE_BAR = 32


def find_window():
    """Return dict with id, x, y, w, h (logical points) for the biggest GD window."""
    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    infos = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []
    best = None
    for info in infos:
        owner = str(info.get("kCGWindowOwnerName", "")).lower()
        name = str(info.get("kCGWindowName", "") or "").lower()
        if not any(h in owner or h in name for h in OWNER_HINTS):
            continue
        if info.get("kCGWindowLayer", 0) != 0:
            continue
        b = info["kCGWindowBounds"]
        w, h = int(b["Width"]), int(b["Height"])
        if w < 200 or h < 150:
            continue
        cand = {
            "id": int(info["kCGWindowNumber"]),
            "pid": int(info["kCGWindowOwnerPID"]),
            "x": int(b["X"]), "y": int(b["Y"]), "w": w, "h": h,
            "owner": info.get("kCGWindowOwnerName", ""),
        }
        if best is None or w * h > best["w"] * best["h"]:
            best = cand
    return best


class Grabber:
    """Grabs just the game window with Quartz at 1x resolution.

    Grabbing the window directly (instead of a screen region) skips the retina
    2x pixels, which roughly doubles the frame rate, and still works if another
    window overlaps the game.
    """

    def __init__(self, refresh_s=2.0):
        self.refresh_s = refresh_s
        self.win = None
        self._last = 0.0
        self._refresh()

    def _refresh(self):
        win = find_window()
        if win is None:
            raise RuntimeError("Geometry Dash window not found. Is the game open and visible?")
        self.win = win
        self._last = time.time()

    @property
    def region(self):
        """Game area in screen points, title bar excluded."""
        w = self.win
        return {"left": w["x"], "top": w["y"] + TITLE_BAR,
                "width": w["w"], "height": w["h"] - TITLE_BAR}

    def center(self):
        """Middle of the game area in screen points, where clicks go."""
        r = self.region
        return r["left"] + r["width"] / 2, r["top"] + r["height"] / 2

    def grab(self):
        """Return BGR uint8 frame of the game area."""
        if time.time() - self._last > self.refresh_s:
            try:
                self._refresh()
            except RuntimeError:
                pass
        img = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionIncludingWindow,
            self.win["id"],
            Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageNominalResolution,
        )
        if img is None:
            self._refresh()
            raise RuntimeError("window capture failed")
        w = Quartz.CGImageGetWidth(img)
        h = Quartz.CGImageGetHeight(img)
        stride = Quartz.CGImageGetBytesPerRow(img)
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, stride // 4, 4)
        # title bar scales with the image if it came back at 2x anyway
        top = int(round(TITLE_BAR * h / self.win["h"]))
        # one contiguous copy, everything downstream is much faster on it
        return cv2.cvtColor(arr[top:, :w], cv2.COLOR_BGRA2BGR)


class ThreadedGrabber(Grabber):
    """Grabs in a background thread so capture overlaps with the bot's own work.

    grab() returns the newest frame and waits only if no new frame came in yet.
    """

    def __init__(self, refresh_s=2.0):
        import threading
        super().__init__(refresh_s)
        self._cond = threading.Condition()
        self._frame = None
        self._ts = 0.0
        self._seq = 0
        self._taken = 0
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop:
            t0 = time.time()
            try:
                f = Grabber.grab(self)
            except RuntimeError:
                time.sleep(0.05)
                continue
            # stamp with the middle of the capture call, closest to when the pixels were read
            ts = (t0 + time.time()) / 2
            with self._cond:
                self._frame = f
                self._ts = ts
                self._seq += 1
                self._cond.notify_all()

    def grab(self):
        with self._cond:
            while self._seq == self._taken:
                self._cond.wait(0.5)
            self._taken = self._seq
            return self._frame

    def poll(self):
        """Newest (frame, capture_time) if there is one we haven't returned yet, else None."""
        with self._cond:
            if self._seq == self._taken:
                return None
            self._taken = self._seq
            return self._frame, self._ts

    def stop(self):
        """Stop the capture thread."""
        self._stop = True
