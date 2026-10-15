"""
phase4_shufflenet.py

Phase 4 uses transfer learning with ShuffleNet V2 (1.0x), a network designed
for mobile and embedded devices.
"""

import os
from common import (get_device, load_data, run_seeds, summarize, save_runs,
                    plot_curves, plot_confusion, RESULTS_DIR, CLASSES)
from models import build_shufflenet, shufflenet_train_mode

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The first attempt used lr 1e-3 for 20 epochs. Validation accuracy was still
# rising at the end (79.2 percent), so that setting was under-trained.
EPOCHS = 40
LR = 1e-2
BATCH_SIZE = 64


def main():
    device = get_device()
    data, mean, std = load_data(INPUT_SIZE, device, mean_std=(IMAGENET_MEAN, IMAGENET_STD))

    print(build_shufflenet().fc)
    results = run_seeds(build_shufflenet, data, mean, std, "phase4_shufflenet",
                        EPOCHS, LR, BATCH_SIZE, train_mode_fn=shufflenet_train_mode)
    save_runs(results, "phase4_shufflenet")

    main_run = results[0]
    plot_curves(main_run, "Phase 4, ShuffleNet V2 with a frozen backbone",
                os.path.join(RESULTS_DIR, "phase4_curves.png"))
    plot_confusion(main_run["confusion_val"], "Phase 4, validation confusion matrix",
                   os.path.join(RESULTS_DIR, "phase4_confusion_val.png"))

    s = summarize(results)
    print(f"params total {s['params_total']} trainable {s['params_trainable']}")
    for key in ("train_acc_at_best", "val_acc", "test_acc"):
        print(f"{key:18s} {s[key]['mean']:.4f} +- {s[key]['std']:.4f}")
    for cls, acc in zip(CLASSES, main_run["test_per_class_acc"]):
        print(f"  {cls:15s} test acc {acc:.3f}")


if __name__ == "__main__":
    main()
