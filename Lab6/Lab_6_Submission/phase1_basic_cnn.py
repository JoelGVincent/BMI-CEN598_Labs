"""
phase1_basic_cnn.py

Phase 1 is the baseline. A few convolutional layers followed by a few fully
connected layers, with no pooling and no dropout.

Preprocessing. Images are resized to 64x64 and normalized with the training
set's per channel mean and std (see prepare_data.py). 64x64 was chosen over
128x128 after comparing both on the validation set. The larger size was not
better for this small network, and it makes the network without pooling far
more expensive to train.
"""

import os
from common import (get_device, load_data, run_seeds, summarize, save_runs,
                    plot_curves, plot_confusion, RESULTS_DIR, CLASSES)
from models import FoodCNN

INPUT_SIZE = 64
CHANNELS = (16, 32, 64)
POOL_AFTER = (False, False, False)
FC_UNITS = (128, 64)

# Phases 1 to 3 all use the same training budget. The best epoch by validation
# accuracy is the one that is kept, so a long budget cannot hurt a model that
# overfits early.
EPOCHS = 60
LR = 1e-3
BATCH_SIZE = 64


def main():
    device = get_device()
    data, mean, std = load_data(INPUT_SIZE, device)

    def build():
        return FoodCNN(INPUT_SIZE, CHANNELS, POOL_AFTER, FC_UNITS)

    print(build())
    results = run_seeds(build, data, mean, std, "phase1_basic_cnn", EPOCHS, LR, BATCH_SIZE)
    save_runs(results, "phase1_basic_cnn")

    main_run = results[0]
    plot_curves(main_run, "Phase 1, basic CNN (no pooling, no dropout)",
                os.path.join(RESULTS_DIR, "phase1_curves.png"))
    plot_confusion(main_run["confusion_val"], "Phase 1, validation confusion matrix",
                   os.path.join(RESULTS_DIR, "phase1_confusion_val.png"))

    s = summarize(results)
    print(f"params total {s['params_total']}")
    for key in ("train_acc_at_best", "val_acc", "test_acc"):
        print(f"{key:18s} {s[key]['mean']:.4f} +- {s[key]['std']:.4f}")
    for cls, acc in zip(CLASSES, main_run["test_per_class_acc"]):
        print(f"  {cls:15s} test acc {acc:.3f}")


if __name__ == "__main__":
    main()
