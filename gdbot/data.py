"""Training data on disk.

Every sample is one stacked observation plus a target (0 = release, 1 = hold)
and a weight. Samples are stored as compressed npz chunks in data/<name>/.
Frames are stored as uint8 to keep the files small.
"""
import glob
import os
import time

import numpy as np


def save_chunk(folder, obs, target, weight):
    if len(obs) == 0:
        return None
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{int(time.time() * 1000)}.npz")
    np.savez_compressed(
        path,
        obs=(np.asarray(obs) * 255).astype(np.uint8),
        target=np.asarray(target, dtype=np.float32),
        weight=np.asarray(weight, dtype=np.float32),
    )
    return path


def load_all(folders, max_samples=None):
    obs, target, weight = [], [], []
    files = []
    for f in folders:
        files += sorted(glob.glob(os.path.join(f, "*.npz")))
    # newest data last, so trimming keeps the most recent
    for path in files:
        d = np.load(path)
        obs.append(d["obs"])
        target.append(d["target"])
        weight.append(d["weight"])
    if not obs:
        return None
    obs = np.concatenate(obs)
    target = np.concatenate(target)
    weight = np.concatenate(weight)
    if max_samples and len(obs) > max_samples:
        obs, target, weight = obs[-max_samples:], target[-max_samples:], weight[-max_samples:]
    return obs, target, weight
