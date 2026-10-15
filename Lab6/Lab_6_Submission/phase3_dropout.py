"""
phase3_dropout.py

Phase 3 adds dropout to the phase 2 network (the CNN with pooling) and checks
whether it stops overfitting.
"""

import os
import json
from common import (get_device, load_data, run_seeds, summarize, save_runs,
                    save_result, plot_curves, plot_confusion, plot_overlay,
                    RESULTS_DIR, CLASSES)
from models import FoodCNN

INPUT_SIZE = 64
CHANNELS = (16, 32, 64)
POOL_AFTER = (True, True, True)
FC_UNITS = (128, 64)
EPOCHS = 60
LR = 1e-3
BATCH_SIZE = 64

# name -> (fully connected dropout, conv Dropout2d)
SETTINGS = {
    "fc0.3": (0.3, 0.0),
    "fc0.5": (0.5, 0.0),
    "fc0.7": (0.7, 0.0),
    "fc0.5_conv0.1": (0.5, 0.1),
    "fc0.5_conv0.2": (0.5, 0.2),
    "fc0.3_conv0.2": (0.3, 0.2),
}
FINAL = "fc0.5_conv0.2"
RESOLVED_TOLERANCE = 0.10


def main():
    device = get_device()
    data, mean, std = load_data(INPUT_SIZE, device)

    all_runs, sweep = {}, {}
    for name, (fc_p, conv_p) in SETTINGS.items():
        def build(fc_p=fc_p, conv_p=conv_p):
            return FoodCNN(INPUT_SIZE, CHANNELS, POOL_AFTER, FC_UNITS,
                           fc_dropout=fc_p, conv_dropout=conv_p)

        results = run_seeds(build, data, mean, std, f"phase3_{name}", EPOCHS, LR, BATCH_SIZE)
        all_runs[name] = results
        sweep[name] = summarize(results)
        sweep[name]["fc_dropout"] = fc_p
        sweep[name]["conv_dropout"] = conv_p

    # The no dropout reference is the phase 2 result, which used identical
    # settings. It is read back from disk instead of retrained.
    with open(os.path.join(RESULTS_DIR, "phase2_pooling.json")) as f:
        phase2 = json.load(f)
    sweep["no_dropout"] = phase2["summary"]
    sweep["no_dropout"]["fc_dropout"] = 0.0
    sweep["no_dropout"]["conv_dropout"] = 0.0

    for name, s in sweep.items():
        rise = s["final_val_loss"]["mean"] / s["min_val_loss"]["mean"] - 1.0
        s["val_loss_rise_fraction"] = rise
        s["overfitting_resolved"] = rise <= RESOLVED_TOLERANCE
    save_result(sweep, "phase3_dropout_sweep")

    save_runs(all_runs[FINAL], "phase3_dropout")
    main_run = all_runs[FINAL][0]
    plot_curves(main_run, "Phase 3, CNN with pooling and dropout (FC 0.5, conv 0.2)",
                os.path.join(RESULTS_DIR, "phase3_curves.png"))
    plot_confusion(main_run["confusion_val"], "Phase 3, validation confusion matrix",
                   os.path.join(RESULTS_DIR, "phase3_confusion_val.png"))

    # Validation loss of the main (first seed) run for the settings that tell
    # the story, so the loss climbing or not climbing is visible at a glance.
    overlay = {"no dropout": phase2["runs"][0]["history"],
               "FC 0.3": all_runs["fc0.3"][0]["history"],
               "FC 0.5": all_runs["fc0.5"][0]["history"],
               "FC 0.5 + conv 0.2": all_runs["fc0.5_conv0.2"][0]["history"]}
    plot_overlay(overlay, "val_loss", "Validation loss with different dropout settings",
                 "Validation loss", os.path.join(RESULTS_DIR, "phase3_val_loss_overlay.png"))

    print("setting             val acc          test acc         loss rise  resolved")
    for name in ("no_dropout",) + tuple(SETTINGS):
        s = sweep[name]
        print(f"{name:18s} {s['val_acc']['mean']:.4f}+-{s['val_acc']['std']:.4f}  "
              f"{s['test_acc']['mean']:.4f}+-{s['test_acc']['std']:.4f}  "
              f"{s['val_loss_rise_fraction']*100:7.1f}%  {s['overfitting_resolved']}")
    for cls, acc in zip(CLASSES, main_run["test_per_class_acc"]):
        print(f"  {cls:15s} test acc {acc:.3f}")


if __name__ == "__main__":
    main()
