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


def plot_search_comparison(results_by_method, path):
    """Grid vs Random: best-so-far accuracy over trials and over time, and each trial's score."""
    fig, (ax_trials, ax_time, ax_scatter) = plt.subplots(1, 3, figsize=(15, 4))
    for method, results in results_by_method.items():
        ordered = sorted(results, key=lambda r: r["trial"])
        scores = [r["best_val_acc"] for r in ordered]
        best_so_far = [max(scores[: i + 1]) for i in range(len(scores))]
        minutes, total = [], 0.0
        for r in ordered:
            total += r["time_sec"] / 60
            minutes.append(total)

        trials = range(1, len(scores) + 1)
        ax_trials.step(trials, best_so_far, where="post", label=method)
        ax_time.step(minutes, best_so_far, where="post", label=method)
        ax_scatter.scatter(trials, scores, label=method, alpha=0.7)

    ax_trials.set(xlabel="trial", ylabel="best val accuracy so far", title="Best score vs trials")
    ax_time.set(xlabel="minutes", ylabel="best val accuracy so far", title="Best score vs time")
    ax_scatter.set(xlabel="trial", ylabel="val accuracy", title="Every trial")
    for ax in (ax_trials, ax_time, ax_scatter):
        ax.legend()
    return _save(fig, path)
