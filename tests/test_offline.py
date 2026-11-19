"""Offline checks that do not need the game: python -m tests.test_offline"""
import numpy as np
import torch

from gdbot import model as M
from gdbot.train import train
from gdbot.vision import DeathDetector, FrameStack, OBS_H, OBS_W, STACK, preprocess


def fake_level(width=4000, h=591, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, width, 3), 40, np.uint8)
    img[460:, :] = 90
    for x in rng.integers(0, width - 40, 120):
        img[420:460, x:x + 30] = 230  # blocks
    for x in rng.integers(0, width - 40, 300):
        img[rng.integers(0, 400), x] = 200  # background specks
    return img


def test_death_detector():
    lvl = fake_level()
    det = DeathDetector()
    events = []
    t = 0.0
    x = 0
    # scroll 3s, freeze 1s (death), back to start and scroll again
    for phase, frames in (("run", 180), ("dead", 60), ("run", 120)):
        if phase == "run" and events:
            x = 0
        for _ in range(frames):
            if phase == "run":
                x += 9
            ev = det.update(lvl[:, x:x + 788], t)
            if ev:
                events.append((ev, round(t, 2)))
            t += 1 / 60
    print("events:", events)
    kinds = [e[0] for e in events]
    assert kinds == ["respawned", "died", "respawned"], kinds
    assert abs(events[1][1] - 3.0) < 0.4


def test_model_and_train():
    net = M.DashNet()
    st = FrameStack()
    obs = st.push(preprocess(fake_level()[:, :788]))
    assert obs.shape == (STACK, OBS_H, OBS_W)
    n = 512
    x = (np.random.rand(n, STACK, OBS_H, OBS_W) * 255).astype(np.uint8)
    y = (x[:, -1, :, 64:].mean(axis=(1, 2)) > 127.5).astype(np.float32)
    w = np.ones(n, np.float32)
    train(net, x, y, w, torch.device("cpu"), epochs=2, batch=64)
    out = net(torch.from_numpy(obs)[None])
    assert out.shape == (1,)


if __name__ == "__main__":
    test_death_detector()
    test_model_and_train()
    print("ok")
