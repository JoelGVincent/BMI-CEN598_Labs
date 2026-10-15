"""
compare_phases.py

Reads the saved results of the four phases and produces the comparison
figures and tables used in the report. Run it after all four phase scripts.

  compare_test_accuracy.png   test accuracy of each phase with its spread
  compare_per_class.png       test accuracy per food class for each phase
  compare_val_curves.png      validation loss and accuracy of every phase
  compare_summary.json        every number the report tables are built from
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS_DIR, CLASSES

PHASES = [
    ("Phase 1\nbasic CNN", "phase1_basic_cnn"),
    ("Phase 2\n+ pooling", "phase2_pooling"),
    ("Phase 3\n+ dropout", "phase3_dropout"),
    ("Phase 4\nShuffleNet", "phase4_shufflenet"),
]


def load(tag):
    with open(os.path.join(RESULTS_DIR, f"{tag}.json")) as f:
        return json.load(f)


def top_confusions(cm, k=5):
    """The k largest off diagonal cells as (true class, predicted class, count, share of the true class)."""
    cm = np.array(cm)
    pairs = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                pairs.append((CLASSES[i], CLASSES[j], int(cm[i, j]), float(cm[i, j] / cm[i].sum())))
    pairs.sort(key=lambda p: -p[2])
    return pairs[:k]


def main():
    data = {tag: load(tag) for _, tag in PHASES}
    summary = {}

    for label, tag in PHASES:
        runs = data[tag]["runs"]
        per_class = np.mean([r["test_per_class_acc"] for r in runs], axis=0)
        summary[tag] = {
            "summary": data[tag]["summary"],
            "per_class_test_acc_mean": per_class.tolist(),
            "top_confusions_test_seed42": top_confusions(runs[0]["confusion_test"]),
            "top_confusions_val_seed42": top_confusions(runs[0]["confusion_val"]),
        }
    with open(os.path.join(RESULTS_DIR, "compare_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    labels = [l for l, _ in PHASES]
    colors = ["#9aa5b1", "#5b8fb9", "#2f6f9f", "#d9822b"]

    # Test accuracy per phase
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means = [summary[t]["summary"]["test_acc"]["mean"] * 100 for _, t in PHASES]
    stds = [summary[t]["summary"]["test_acc"]["std"] * 100 for _, t in PHASES]
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=4)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 1.5, f"{m:.1f}%", ha="center")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Test accuracy of each phase (mean of 3 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "compare_test_accuracy.png"), dpi=150)
    plt.close(fig)

    # Per class test accuracy
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.2
    x = np.arange(len(CLASSES))
    for i, ((label, tag), color) in enumerate(zip(PHASES, colors)):
        vals = np.array(summary[tag]["per_class_test_acc_mean"]) * 100
        ax.bar(x + (i - 1.5) * width, vals, width, label=label.replace("\n", " "), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Test accuracy per food class")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "compare_per_class.png"), dpi=150)
    plt.close(fig)

    # Validation curves of every phase's main run
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (label, tag), color in zip(PHASES, colors):
        h = data[tag]["runs"][0]["history"]
        epochs = range(1, len(h["val_loss"]) + 1)
        axes[0].plot(epochs, h["val_loss"], label=label.replace("\n", " "), color=color)
        axes[1].plot(epochs, np.array(h["val_acc"]) * 100, label=label.replace("\n", " "), color=color)
    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_yscale("log")
    axes[1].set_title("Validation accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "compare_val_curves.png"), dpi=150)
    plt.close(fig)

    print("phase            params     train acc   val acc    test acc   best epoch  seconds")
    for label, tag in PHASES:
        s = summary[tag]["summary"]
        print(f"{label.replace(chr(10), ' '):18s} {s['params_total']:9d}  "
              f"{s['train_acc_at_best']['mean']*100:6.1f}%   {s['val_acc']['mean']*100:6.1f}%   "
              f"{s['test_acc']['mean']*100:6.1f}%   {s['best_epoch']['mean']:6.1f}   {s['train_seconds']['mean']:6.0f}")
    print()
    print("per class test accuracy (mean of 3 seeds, percent)")
    print(f"{'class':16s}" + "".join(f"{'P' + str(i + 1):>8s}" for i in range(4)))
    for c, cls in enumerate(CLASSES):
        print(f"{cls:16s}" + "".join(
            f"{summary[t]['per_class_test_acc_mean'][c] * 100:8.1f}" for _, t in PHASES))
    print()
    for _, tag in PHASES:
        print(tag, "top test confusions (seed 42)")
        for t, p, n, share in summary[tag]["top_confusions_test_seed42"]:
            print(f"   {t} predicted as {p}  {n} images  ({share * 100:.0f}% of {t})")


if __name__ == "__main__":
    main()
