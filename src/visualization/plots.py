"""Matplotlib figures for training and evaluation results."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_curves(history, path, title=None):
    """Train vs validation loss and accuracy per epoch."""
    epochs = [row["epoch"] for row in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [r["train_loss"] for r in history], label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], label="validation")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, [r["train_acc"] for r in history], label="train")
    ax_acc.plot(epochs, [r["val_acc"] for r in history], label="validation")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax_acc.legend()

    if title:
        fig.suptitle(title)
    return _save(fig, path)


def plot_confusion_matrix(cm, path, title="Confusion matrix"):
    """Row-normalised 200x200 heatmap; a bright diagonal means correct predictions."""
    row_sums = cm.sum(axis=1, keepdims=True)
    normalised = cm / row_sums.clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalised, cmap="viridis", vmin=0, vmax=1)
    ax.set(xlabel="predicted class index", ylabel="true class index", title=title)
    fig.colorbar(image, ax=ax, fraction=0.046, label="share of true class")
    return _save(fig, path)
