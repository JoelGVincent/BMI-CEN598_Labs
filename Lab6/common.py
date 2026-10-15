"""
common.py

Shared helpers for all four phases of the Food-11 project. Keeping the data
loading, training loop, evaluation, and plotting in one place means every
phase is trained and measured in exactly the same way, so the only thing that
changes between phases is the model itself. That makes the phase to phase
comparison in the report a fair one.
"""

import os
import json
import time
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths. Each one can be overridden with an environment variable.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("FOOD_DATA_DIR", "/mnt/c/Users/joelj/Downloads/Food_Image_Data")
CACHE_DIR = os.environ.get("FOOD_CACHE_DIR", os.path.expanduser("~/lab6_cache"))
RESULTS_DIR = os.environ.get("FOOD_RESULTS_DIR", os.path.join(HERE, "results"))

CLASSES = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
           "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
NUM_CLASSES = len(CLASSES)

# The dataset's own folder names for the three splits. The test split is the
# folder called "evaluation".
SPLITS = {"train": "training", "val": "validation", "test": "evaluation"}

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data loading. Images are cached as uint8 arrays by prepare_data.py and held
# on the GPU, so a batch is just an index into a tensor already in GPU memory.
# ---------------------------------------------------------------------------
def load_data(size, device, mean_std=None):
    """
    Load the cached splits at the given square image size.

    mean_std can be used to override the normalization constants. By default
    the training split's own per channel mean and std are used. ShuffleNet
    overrides them with the ImageNet constants its pretrained weights expect.
    """
    data = {}
    for split in SPLITS:
        x = np.load(os.path.join(CACHE_DIR, f"{size}_{split}_x.npy"))
        y = np.load(os.path.join(CACHE_DIR, f"{split}_y.npy"))
        x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).contiguous()
        data[split] = (x, torch.from_numpy(y).to(device))

    if mean_std is None:
        with open(os.path.join(CACHE_DIR, f"{size}_stats.json")) as f:
            st = json.load(f)
        mean_std = (st["mean"], st["std"])
    mean = torch.tensor(mean_std[0], device=device).view(1, 3, 1, 1)
    std = torch.tensor(mean_std[1], device=device).view(1, 3, 1, 1)
    return data, mean, std


def normalize(xb, mean, std):
    """uint8 batch in 0..255 to a float batch scaled to 0..1 then standardized."""
    return (xb.float() / 255.0 - mean) / std


def batches(x, y, batch_size, shuffle, gen=None):
    n = x.shape[0]
    if shuffle:
        idx = torch.randperm(n, device=x.device, generator=gen)
    else:
        idx = torch.arange(n, device=x.device)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield x[j], y[j]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, x, y, mean, std, batch_size=256):
    """Return (mean loss, accuracy, predictions) with the model in eval mode."""
    model.eval()
    total_loss, correct, preds = 0.0, 0, []
    for xb, yb in batches(x, y, batch_size, shuffle=False):
        out = model(normalize(xb, mean, std))
        total_loss += F.cross_entropy(out, yb, reduction="sum").item()
        p = out.argmax(dim=1)
        correct += (p == yb).sum().item()
        preds.append(p)
    return total_loss / len(y), correct / len(y), torch.cat(preds)


def confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    """Rows are the true class and columns are the predicted class."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.cpu().numpy(), y_pred.cpu().numpy()):
        cm[t, p] += 1
    return cm


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def fit(model, data, mean, std, epochs, lr, batch_size, name,
        weight_decay=0.0, train_mode_fn=None, seed=SEED):
    """
    Train with Adam and cross entropy, then report on the validation and test
    sets using the epoch that had the best validation accuracy.

    After every epoch the model is scored on the full training set and on the
    validation set, both in eval mode. Scoring the training set in eval mode
    matters when dropout is used. Dropout makes the running training accuracy
    during an epoch look worse than the model really is, which would hide the
    overfitting that these experiments are trying to measure.

    train_mode_fn lets a phase decide how the model is put into training mode.
    Phase 4 uses it to keep the frozen ShuffleNet backbone in eval mode so its
    batch norm statistics are not updated.
    """
    set_seed(seed)
    device = next(model.parameters()).device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    x_tr, y_tr = data["train"]
    x_va, y_va = data["val"]
    x_te, y_te = data["test"]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    start = time.time()

    for epoch in range(1, epochs + 1):
        if train_mode_fn is None:
            model.train()
        else:
            train_mode_fn(model)

        for xb, yb in batches(x_tr, y_tr, batch_size, shuffle=True, gen=gen):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(normalize(xb, mean, std)), yb)
            loss.backward()
            optimizer.step()

        tr_loss, tr_acc, _ = evaluate(model, x_tr, y_tr, mean, std)
        va_loss, va_acc, _ = evaluate(model, x_va, y_va, mean, std)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best["val_acc"]:
            best.update(val_acc=va_acc, epoch=epoch,
                        state=copy.deepcopy(model.state_dict()))

        print(f"[{name}] epoch {epoch:3d}/{epochs}  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}", flush=True)

    train_seconds = time.time() - start

    model.load_state_dict(best["state"])
    va_loss, va_acc, va_pred = evaluate(model, x_va, y_va, mean, std)
    te_loss, te_acc, te_pred = evaluate(model, x_te, y_te, mean, std)
    tr_loss, tr_acc, _ = evaluate(model, x_tr, y_tr, mean, std)

    cm_val = confusion_matrix(y_va, va_pred)
    cm_test = confusion_matrix(y_te, te_pred)
    total, trainable = count_params(model)

    val_losses = history["val_loss"]
    min_val_loss = min(val_losses)

    result = {
        "seed": seed,
        "min_val_loss": min_val_loss,
        "min_val_loss_epoch": val_losses.index(min_val_loss) + 1,
        "final_val_loss": val_losses[-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "name": name,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "params_total": total,
        "params_trainable": trainable,
        "train_seconds": train_seconds,
        "best_epoch": best["epoch"],
        "train_acc_at_best": tr_acc,
        "train_loss_at_best": tr_loss,
        "val_acc": va_acc,
        "val_loss": va_loss,
        "test_acc": te_acc,
        "test_loss": te_loss,
        "val_per_class_acc": (cm_val.diagonal() / cm_val.sum(axis=1)).tolist(),
        "test_per_class_acc": (cm_test.diagonal() / cm_test.sum(axis=1)).tolist(),
        "confusion_val": cm_val.tolist(),
        "confusion_test": cm_test.tolist(),
        "history": history,
    }
    return result


SEEDS = (42, 43, 44)


def run_seeds(build_model, data, mean, std, name, epochs, lr, batch_size,
              seeds=SEEDS, **fit_kwargs):
    """
    Train the same configuration once per seed and return every result.

    A single training run can land a point or two higher or lower than the
    typical outcome just from its random initialization and batch order. When
    two configurations differ by only a few points, one run each cannot say
    which is really better. Repeating over three seeds gives a mean and a
    spread to compare. The first seed is the main run that is plotted.
    """
    device = next(iter(data.values()))[0].device
    results = []
    for s in seeds:
        set_seed(s)
        model = build_model().to(device)
        results.append(fit(model, data, mean, std, epochs, lr, batch_size,
                           name=f"{name}_seed{s}", seed=s, **fit_kwargs))
    return results


def summarize(results):
    """Mean and standard deviation over seeds for the headline numbers."""
    out = {"n_seeds": len(results), "params_total": results[0]["params_total"],
           "params_trainable": results[0]["params_trainable"]}
    for key in ("train_acc_at_best", "val_acc", "test_acc", "best_epoch",
                "min_val_loss", "final_val_loss", "final_train_acc",
                "final_val_acc", "train_seconds"):
        vals = np.array([r[key] for r in results], dtype=float)
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out


def save_result(result, tag=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = tag or result["name"]
    path = os.path.join(RESULTS_DIR, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    return path


def save_runs(results, tag):
    """Save every seed's result plus the summary for one configuration."""
    return save_result({"summary": summarize(results), "runs": results}, tag)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curves(result, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = result["history"]
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, h["train_loss"], label="train")
    axes[0].plot(epochs, h["val_loss"], label="validation")
    axes[0].axvline(result["best_epoch"], color="gray", linestyle=":", label="best epoch")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, h["train_acc"], label="train")
    axes[1].plot(epochs, h["val_acc"], label="validation")
    axes[1].axvline(result["best_epoch"], color="gray", linestyle=":", label="best epoch")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overlay(histories, key, title, ylabel, path):
    """Plot one history metric (for example val_loss) for several models together."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, history in histories.items():
        ax.plot(range(1, len(history[key]) + 1), history[key], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
