"""
prepare_data.py

One-time preprocessing for the Food-11 dataset. Every JPEG is decoded once,
converted to RGB, resized to a fixed square size, and stored as a uint8 NumPy
array. Training scripts then load these arrays directly (and keep them on the
GPU) instead of decoding thousands of JPEGs again on every epoch.

The raw images are not a uniform size. Most are 512x512 but many are
non-square (for example 512x384 or 1024x768). They are resized straight to a
square, which keeps the whole food item in frame instead of cropping part of
it away. The small aspect ratio distortion this causes is applied equally to
every split.

Three square sizes are written in a single pass over the data.
  64  and 128 are used by the from-scratch CNNs (phases 1 to 3)
  224 is the standard input size for ShuffleNet (phase 4)

The per-channel mean and standard deviation of the training split are also
computed and saved for each size, and are later used to normalize inputs.
Only the training split is used for these statistics so that no information
from the validation or test images leaks into preprocessing.
"""

import os
import json
import glob
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from common import CLASSES, DATA_DIR, CACHE_DIR, SPLITS

SIZES = [64, 128, 224]


def load_one(path):
    """Decode one image and return it resized to every target size."""
    im = Image.open(path).convert("RGB")
    return [np.asarray(im.resize((s, s), Image.BILINEAR), dtype=np.uint8) for s in SIZES]


def channel_stats(x, chunk=500):
    """
    Per channel mean and std over all pixels of a uint8 image array. This is
    done in chunks because converting the whole training split to float64 at
    once would need more memory than the machine has.
    """
    n = x.shape[0] * x.shape[1] * x.shape[2]
    total, total_sq = np.zeros(3), np.zeros(3)
    for i in range(0, len(x), chunk):
        c = x[i:i + chunk].astype(np.float64) / 255.0
        total += c.sum(axis=(0, 1, 2))
        total_sq += (c ** 2).sum(axis=(0, 1, 2))
    mean = total / n
    std = np.sqrt(total_sq / n - mean ** 2)
    return {"mean": mean.tolist(), "std": std.tolist()}


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    stats = {s: None for s in SIZES}

    for split_key, folder in SPLITS.items():
        paths, labels = [], []
        for label, cls in enumerate(CLASSES):
            files = sorted(glob.glob(os.path.join(DATA_DIR, folder, cls, "*.jpg")))
            paths += files
            labels += [label] * len(files)
        print(f"{split_key}: {len(paths)} images", flush=True)

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(load_one, paths))

        y = np.array(labels, dtype=np.int64)
        np.save(os.path.join(CACHE_DIR, f"{split_key}_y.npy"), y)
        with open(os.path.join(CACHE_DIR, f"{split_key}_files.txt"), "w") as f:
            f.write("\n".join(paths))

        for i, s in enumerate(SIZES):
            x = np.stack([r[i] for r in results])
            np.save(os.path.join(CACHE_DIR, f"{s}_{split_key}_x.npy"), x)
            if split_key == "train":
                stats[s] = channel_stats(x)
            print(f"  saved size {s}, shape {x.shape}", flush=True)
        del results

    for s in SIZES:
        with open(os.path.join(CACHE_DIR, f"{s}_stats.json"), "w") as f:
            json.dump(stats[s], f)
        print(f"size {s} train mean {stats[s]['mean']} std {stats[s]['std']}")


if __name__ == "__main__":
    main()
