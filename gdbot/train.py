"""Train DashNet on whatever data exists for a level.

    python -m gdbot.train --level seven_seas --epochs 8
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import model as M
from .data import load_all


def level_folders(level):
    return [p for p in glob.glob(os.path.join("data", level, "*")) if os.path.isdir(p)]


def train(net, obs, target, weight, device, epochs=4, batch=256, lr=3e-4, log=True):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(obs)
    # hold frames are rarer than release frames, balance them
    pos = max(float((target * weight).sum()), 1.0)
    neg = max(float(((1 - target) * weight).sum()), 1.0)
    pos_w = torch.tensor(min(neg / pos, 10.0), device=device)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        total, seen, correct = 0.0, 0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            x = torch.from_numpy(obs[idx]).to(device).float() / 255.0
            # tiny brightness jitter so level color changes do not throw it off
            x = x * (0.85 + 0.3 * torch.rand(len(idx), 1, 1, 1, device=device))
            y = torch.from_numpy(target[idx]).to(device)
            w = torch.from_numpy(weight[idx]).to(device)
            logit = net(x)
            loss = F.binary_cross_entropy_with_logits(logit, y, weight=w, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            seen += len(idx)
            correct += ((logit > 0).float() == (y > 0.5).float()).sum().item()
        if log:
            print(f"  epoch {ep + 1}/{epochs}  loss={total / seen:.4f}  acc={correct / seen:.3f}")
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="default")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--fresh", action="store_true", help="start from scratch instead of the saved model")
    args = ap.parse_args()

    device = M.pick_device()
    data = load_all(level_folders(args.level), max_samples=300_000)
    if data is None:
        raise SystemExit(f"no data in data/{args.level}/, record or play --learn first")
    obs, target, weight = data
    print(f"{len(obs)} samples, {target.mean():.2%} hold, device={device}")

    path = os.path.join("models", f"{args.level}.pt")
    os.makedirs("models", exist_ok=True)
    if os.path.exists(path) and not args.fresh:
        net = M.load(path, device)
    else:
        net = M.DashNet().to(device)
    train(net, obs, target, weight, device, epochs=args.epochs)
    M.save(net, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
