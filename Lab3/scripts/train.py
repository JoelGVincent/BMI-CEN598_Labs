"""
train.py

Phases 4-6 of Lab 3: build a small feedforward neural network,
train three versions of it that differ ONLY in hidden-layer activation
function (sigmoid, tanh, ReLU), and evaluate/compare them on train, val,
and test data.

Architecture: 6 inputs (ax,ay,az,gx,gy,gz) -> hidden(32) -> hidden(16) ->
5 output logits (supine/prone/side/sitting/unknown). Multi-class output
uses softmax.

Run:
    python3 train.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")  # no display available; just save PNGs
import matplotlib.pyplot as plt

SEED = 42
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CLASS_NAMES = ["supine", "prone", "side", "sitting", "unknown"]
FEATURE_COLS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]

EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
HIDDEN1 = 32
HIDDEN2 = 16

ACTIVATIONS = ["sigmoid", "tanh", "relu"]


def load_split(name):
    path = DATASET_DIR / f"{name}.csv"
    X, y = [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[c]) for c in FEATURE_COLS])
            y.append(int(row["label"]) - 1)  # 1..5 -> 0..4 for PyTorch
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class PostureMLP(nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(len(FEATURE_COLS), HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, len(CLASS_NAMES))

        if activation == "sigmoid":
            self.act = nn.Sigmoid()
            nonlinearity_gain = nn.init.calculate_gain("sigmoid")
        elif activation == "tanh":
            self.act = nn.Tanh()
            nonlinearity_gain = nn.init.calculate_gain("tanh")
        elif activation == "relu":
            self.act = nn.ReLU()
            nonlinearity_gain = nn.init.calculate_gain("relu")
        else:
            raise ValueError(activation)

        # Use the appropriate init per activation
        # using a ReLU-style init for sigmoid/tanh would unfairly handicap them in
        # the comparison.
        for layer in (self.fc1, self.fc2, self.fc3):
            if activation == "relu":
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                nn.init.xavier_normal_(layer.weight, gain=nonlinearity_gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)  # raw logits; CrossEntropyLoss applies log-softmax


def make_loader(X, y, mean, std, batch_size, shuffle):
    Xn = (X - mean) / std
    ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_n += xb.size(0)

    return total_loss / total_n, total_correct / total_n


def confusion_matrix(model, loader, n_classes):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    with torch.no_grad():
        for xb, yb in loader:
            preds = torch.argmax(model(xb), dim=1)
            for t, p in zip(yb.tolist(), preds.tolist()):
                cm[t][p] += 1
    return cm


def plot_curves(history, activation, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title(f"{activation}: loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title(f"{activation}: accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"curves_{activation}.png", dpi=130)
    plt.close(fig)


def plot_confusion(cm, activation, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{activation}: test confusion matrix")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{activation}.png", dpi=130)
    plt.close(fig)


def plot_comparison(summary, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(summary.keys())
    accs = [summary[n]["test_acc"] for n in names]
    bars = ax.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("test accuracy")
    ax.set_title("Activation function comparison (test accuracy)")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "activation_comparison.png", dpi=130)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    # Standardize using TRAIN stats only, applied to all splits
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    summary = {}
    for activation in ACTIVATIONS:
        print(f"\n=== Training with {activation} ===")
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        train_loader = make_loader(X_train, y_train, mean, std, BATCH_SIZE, shuffle=True)
        val_loader = make_loader(X_val, y_val, mean, std, BATCH_SIZE, shuffle=False)
        test_loader = make_loader(X_test, y_test, mean, std, BATCH_SIZE, shuffle=False)

        model = PostureMLP(activation)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer=None)
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                      f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None)
        cm = confusion_matrix(model, test_loader, len(CLASS_NAMES))
        print(f"  TEST: loss={test_loss:.4f} acc={test_acc:.4f}")
        print(f"  confusion matrix (rows=true, cols=pred, order={CLASS_NAMES}):")
        print(cm)

        plot_curves(history, activation, RESULTS_DIR)
        plot_confusion(cm, activation, RESULTS_DIR)

        summary[activation] = {
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1],
            "test_acc": test_acc,
            "test_loss": test_loss,
            "confusion_matrix": cm.tolist(),
        }

    plot_comparison(summary, RESULTS_DIR)

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary (test accuracy) ===")
    for activation in ACTIVATIONS:
        s = summary[activation]
        print(f"  {activation:8s}: train={s['final_train_acc']:.4f} "
              f"val={s['final_val_acc']:.4f} test={s['test_acc']:.4f}")
    print(f"\nPlots + summary.json written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
