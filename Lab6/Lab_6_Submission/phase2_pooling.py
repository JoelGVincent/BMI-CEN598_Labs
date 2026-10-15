"""
phase2_pooling.py

Phase 2 adds max pooling to the phase 1 network. Nothing else changes. The
conv widths, the fully connected layers, the learning rate, and the number of
epochs are identical, so any difference in the results comes from pooling
alone.
"""

import os
import json
from common import (get_device, load_data, run_seeds, summarize, save_runs,
                    save_result, plot_curves, plot_confusion, RESULTS_DIR, CLASSES)
from models import FoodCNN

INPUT_SIZE = 64
CHANNELS = (16, 32, 64)
FC_UNITS = (128, 64)
EPOCHS = 60
LR = 1e-3
BATCH_SIZE = 64

POOL_CONFIGS = {
    "1_pool": (True, False, False),
    "2_pools": (True, True, False),
    "3_pools": (True, True, True),
}
MAIN = "3_pools"


def main():
    device = get_device()
    data, mean, std = load_data(INPUT_SIZE, device)

    all_runs, ablation = {}, {}
    for name, pools in POOL_CONFIGS.items():
        def build(pools=pools):
            return FoodCNN(INPUT_SIZE, CHANNELS, pools, FC_UNITS)

        results = run_seeds(build, data, mean, std, f"phase2_{name}", EPOCHS, LR, BATCH_SIZE)
        all_runs[name] = results
        ablation[name] = summarize(results)
        ablation[name]["pool_after"] = list(pools)

    # The no pooling reference is the phase 1 result, which used identical
    # settings. It is read back from disk instead of retrained.
    with open(os.path.join(RESULTS_DIR, "phase1_basic_cnn.json")) as f:
        ablation["0_pools"] = json.load(f)["summary"]
        ablation["0_pools"]["pool_after"] = [False, False, False]
    save_result(ablation, "phase2_pool_ablation")

    save_runs(all_runs[MAIN], "phase2_pooling")
    main_run = all_runs[MAIN][0]
    plot_curves(main_run, "Phase 2, CNN with max pooling",
                os.path.join(RESULTS_DIR, "phase2_curves.png"))
    plot_confusion(main_run["confusion_val"], "Phase 2, validation confusion matrix",
                   os.path.join(RESULTS_DIR, "phase2_confusion_val.png"))

    print("pooling layers    params    val acc          test acc")
    for name in ("0_pools", "1_pool", "2_pools", "3_pools"):
        a = ablation[name]
        print(f"{name:14s} {a['params_total']:10d}  "
              f"{a['val_acc']['mean']:.4f}+-{a['val_acc']['std']:.4f}  "
              f"{a['test_acc']['mean']:.4f}+-{a['test_acc']['std']:.4f}")
    for cls, acc in zip(CLASSES, main_run["test_per_class_acc"]):
        print(f"  {cls:15s} test acc {acc:.3f}")


if __name__ == "__main__":
    main()
