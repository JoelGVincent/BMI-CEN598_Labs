"""
prepare_dataset.py

Phase 3 of the Lab 3.

1. Loads every trial CSV in IMU_Data_Collection/, labels it by filename
   prefix (ground-truth labeling / annotation -- the file name IS the label).
2. Splits whole TRIALS (not individual rows) into train/val/test, so that
   near-duplicate rows from the same held-still trial never appear in more
   than one split. Splitting by row instead of by trial would leak
   information between splits and inflate test accuracy artificially.
3. Adds synthetic Gaussian noise to each split independently (this happens
   AFTER splitting, so noise never leaks information across the split
   boundary either) to represent the small-motion sensor noise that a real,
   worn deployment would have, since our hand-held data collection is
   artificially clean.
4. Writes train.csv / val.csv / test.csv (each: ax,ay,az,gx,gy,gz,label)
   to Lab3/dataset/, plus a class_counts.txt summary.

Run:
    python3 prepare_dataset.py
"""

import csv
import random
from pathlib import Path
from collections import defaultdict

RANDOM_SEED = 42

DATA_DIR = Path(__file__).resolve().parent.parent / "IMU_Data_Collection"
OUT_DIR = Path(__file__).resolve().parent.parent / "dataset"

# Class labels per the assignment: 1=supine, 2=prone, 3=side, 4=sitting, 5=unknown
CLASS_NAMES = {1: "supine", 2: "prone", 3: "side", 4: "sitting", 5: "unknown"}


def label_for_filename(name: str) -> int:
    """Ground-truth label, derived from the filename prefix."""
    n = name.lower()
    if n.startswith("supine"):
        return 1
    if n.startswith("prone"):
        return 2
    if n.startswith("side"):
        return 3
    if n.startswith("sitting"):
        return 4
    if n.startswith("standing") or n.startswith("unknown"):
        return 5
    raise ValueError(f"Don't know how to label file: {name}")


def load_trial(path: Path):
    """Returns a list of [ax, ay, az, gx, gy, gz] float rows from one trial CSV."""
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) != 7:
                continue  # skip any malformed/partial row
            # columns: timestamp_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps
            rows.append([float(v) for v in row[1:]])
    return rows


def group_split(trial_files, seed):
    """Splits a class's trial FILES (not rows) into train/val/test.
    Uses 1 trial for val and 1 for test (whatever's left goes to train),
    which generalizes cleanly to both the 6-trial and 5-trial classes here.
    """
    rng = random.Random(seed)
    files = list(trial_files)
    rng.shuffle(files)
    if len(files) < 3:
        raise ValueError("Need at least 3 trials per class to hold out val+test.")
    test = files[:1]
    val = files[1:2]
    train = files[2:]
    return train, val, test


def add_noise(rows, accel_std, gyro_std, n_copies, rng):
    """For each clean row, keep the original AND add n_copies noisy versions
    Returns the expanded list of rows."""
    out = []
    for r in rows:
        out.append(r)  # keep the original clean sample too
        for _ in range(n_copies):
            noisy = [
                r[0] + rng.gauss(0, accel_std),
                r[1] + rng.gauss(0, accel_std),
                r[2] + rng.gauss(0, accel_std),
                r[3] + rng.gauss(0, gyro_std),
                r[4] + rng.gauss(0, gyro_std),
                r[5] + rng.gauss(0, gyro_std),
            ]
            out.append(noisy)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Group all trial files by class.
    files_by_class = defaultdict(list)
    for path in sorted(DATA_DIR.glob("*.csv")):
        label = label_for_filename(path.name)
        files_by_class[label].append(path)

    print("Trials found per class:")
    for label in sorted(files_by_class):
        print(f"  {label} ({CLASS_NAMES[label]}): {len(files_by_class[label])} trials")

    # 2. Split trials (not rows) into train/val/test, per class.
    split_files = {"train": [], "val": [], "test": []}
    for label, files in files_by_class.items():
        train, val, test = group_split(files, seed=RANDOM_SEED + label)
        split_files["train"] += [(p, label) for p in train]
        split_files["val"] += [(p, label) for p in val]
        split_files["test"] += [(p, label) for p in test]

    print("\nTrials assigned per split:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(split_files[split])} trials ->",
              [p.name for p, _ in split_files[split]])

    # 3. Load raw rows per split, add noise, write out final CSVs.
    #    Noise levels: accel std 0.02 g, gyro std 1.5 deg/s -- chosen as a
    #    few times larger than the natural jitter already visible in the
    #    clean hand-held data.
    ACCEL_STD = 0.02
    GYRO_STD = 1.5
    N_COPIES = 4

    rng = random.Random(RANDOM_SEED)
    summary_lines = []
    for split in ("train", "val", "test"):
        out_rows = []  # (ax,ay,az,gx,gy,gz,label)
        per_class_count = defaultdict(int)
        for path, label in split_files[split]:
            raw_rows = load_trial(path)
            expanded = add_noise(raw_rows, ACCEL_STD, GYRO_STD, N_COPIES, rng)
            for r in expanded:
                out_rows.append(r + [label])
            per_class_count[label] += len(expanded)

        rng.shuffle(out_rows)
        out_path = OUT_DIR / f"{split}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps", "label"])
            writer.writerows(out_rows)

        line = f"{split}: {len(out_rows)} rows total | " + ", ".join(
            f"{CLASS_NAMES[l]}={c}" for l, c in sorted(per_class_count.items())
        )
        print(line)
        summary_lines.append(line)

    with open(OUT_DIR / "class_counts.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nWrote train.csv / val.csv / test.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
