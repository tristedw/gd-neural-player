"""Frame preprocessing and death detection."""
from collections import deque

import cv2
import numpy as np

OBS_H, OBS_W = 96, 128
STACK = 4


def preprocess(frame_bgr):
    """BGR frame -> small grayscale float32 in [0, 1]."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)
    # edges matter more than colors, GD recolors the whole level all the time
    edges = cv2.Canny(gray, 40, 120)
    out = 0.5 * gray.astype(np.float32) / 255.0 + 0.5 * edges.astype(np.float32) / 255.0
    return out


class FrameStack:
    """Keeps the last n preprocessed frames so the net can see motion."""
    def __init__(self, n=STACK):
        self.n = n
        self.frames = deque(maxlen=n)

    def reset(self):
        """Forget old frames, call on respawn."""
        self.frames.clear()

    def push(self, obs):
        """Add a frame and return the stack. The first frame fills the whole stack."""
        if not self.frames:
            for _ in range(self.n):
                self.frames.append(obs)
        else:
            self.frames.append(obs)
        return self.get()

    def get(self):
        """Current stack as (n, H, W)."""
        return np.stack(self.frames, axis=0)


SNAP_PX = 12.0
SNAP_DIFF = 8.0


class DeathDetector:
    """Detects death by watching the camera.

    While the player is alive the level scrolls left at a steady rate. When the
    player dies the camera freezes for about a second before the respawn, so a
    run of near-zero horizontal motion means we died (or finished the level).
    A respawn shows up as motion starting again.
    """

    def __init__(self, stall_px=0.35, stall_time=0.22, resume_time=0.12, size=(160, 120)):
        self.stall_px = stall_px
        self.stall_time = stall_time
        self.resume_time = resume_time
        self.size = size
        self.prev = None
        self.win = None
        self.alive = False
        self.still_since = None
        self.moving_since = None
        self.last_dx = 0.0
        self.last_diff = 0.0
        self.gray = None
        self.prev_gray = None
        self.snap_time = None
        self.spawn_time = None
        self.anchored = False

    def _prep(self, frame_bgr):
        g = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), self.size, interpolation=cv2.INTER_AREA)
        self.gray = g.astype(np.float32)
        g = cv2.Canny(g, 40, 120).astype(np.float32)
        if self.win is None:
            self.win = cv2.createHanningWindow(self.size, cv2.CV_32F)
        return g

    def update(self, frame_bgr, t):
        """Feed a frame. Returns 'died', 'respawned' or None.

        On 'died', self.death_time is set to when the stall started.
        """
        cur = self._prep(frame_bgr)
        event = None
        if self.prev is not None:
            (dx, _dy), resp = cv2.phaseCorrelate(self.prev, cur, self.win)
            dx = abs(dx) if resp > 0.05 else 0.0
            self.last_dx = dx
            self.last_diff = float(np.mean(np.abs(self.gray - self.prev_gray)))
            # respawn makes the camera snap back in one frame, that frame is the real spawn time
            if dx > SNAP_PX or (not self.alive and self.last_diff > SNAP_DIFF):
                if self.alive:
                    # respawned before the stall timer fired (very fast reset), count it as a death
                    self.alive = False
                    self.death_time = self.still_since if self.still_since is not None else t
                    self.snap_time = t
                    self.moving_since = t
                    self.prev = cur
                    self.prev_gray = self.gray
                    return "died"
                self.snap_time = t
            # also catch a fully frozen frame, phase corr can be noisy on flat screens
            frozen = float(np.mean(np.abs(cur - self.prev))) < 0.5
            still = dx < self.stall_px or frozen
            if still:
                self.moving_since = None
                if self.still_since is None:
                    self.still_since = t
                if self.alive and t - self.still_since >= self.stall_time:
                    self.alive = False
                    self.death_time = self.still_since
                    event = "died"
            else:
                self.still_since = None
                if self.moving_since is None:
                    self.moving_since = t
                if not self.alive and t - self.moving_since >= self.resume_time:
                    self.alive = True
                    # only trust the time base if we actually saw the respawn snap
                    # faster reset can hold the player still for a moment after the snap
                    if self.snap_time is not None and self.moving_since - self.snap_time < 1.5:
                        self.spawn_time = self.snap_time
                        self.anchored = True
                    else:
                        self.spawn_time = self.moving_since
                        self.anchored = False
                    self.snap_time = None
                    event = "respawned"
        self.prev = cur
        self.prev_gray = self.gray
        return event
