"""Find the Geometry Dash Meltdown window and grab frames from it."""
import time

import mss
import numpy as np
import Quartz

OWNER_HINTS = ("geometrydashmeltdown", "meltdown", "geometry dash")


def find_window():
    """Return dict with id, x, y, w, h (logical points) for the biggest Meltdown window."""
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
    """Fast window grabber. Re-finds the window every few seconds in case it moved."""

    # macOS draws a title bar on top of wrapped iOS apps, skip it
    TITLE_BAR = 32

    def __init__(self, refresh_s=2.0):
        self.sct = mss.mss()
        self.refresh_s = refresh_s
        self.win = None
        self._last = 0.0
        self._refresh()

    def _refresh(self):
        win = find_window()
        if win is None:
            raise RuntimeError("Geometry Dash Meltdown window not found. Is the game open and visible?")
        self.win = win
        self._last = time.time()

    @property
    def region(self):
        w = self.win
        return {"left": w["x"], "top": w["y"] + self.TITLE_BAR,
                "width": w["w"], "height": w["h"] - self.TITLE_BAR}

    def center(self):
        r = self.region
        return r["left"] + r["width"] / 2, r["top"] + r["height"] / 2

    def grab(self):
        """Return BGR uint8 frame of the game area."""
        if time.time() - self._last > self.refresh_s:
            try:
                self._refresh()
            except RuntimeError:
                pass
        img = np.asarray(self.sct.grab(self.region))
        return img[:, :, :3]
