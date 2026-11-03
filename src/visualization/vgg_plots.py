"""Figures for the VGG comparison. Training curves come from ``plots.py``."""
import math

import matplotlib.pyplot as plt

from src.visualization.plots import _save, _short

BAR_COLOR = "#2a78d6"
CORRECT_COLOR = "#0ca30c"
WRONG_COLOR = "#d03b3b"
GRID_COLOR = "#e1e0d9"
SECONDARY_TEXT = "#52514e"


def plot_accuracy_comparison(models, accuracies, path, split="test"):
    """One bar per model, on a full 0-100% axis so differences are not exaggerated."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, accuracies, width=0.5, color=BAR_COLOR)
    ax.bar_label(bars, labels=[f"{a:.1%}" for a in accuracies], padding=4)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set(ylabel=f"{split} accuracy", title=f"{split.capitalize()} accuracy by model")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelcolor=SECONDARY_TEXT)
    return _save(fig, path)


def plot_example_predictions(rows, path, title=None, n_cols=4):
    """Grid of images. Each ``row``: ``image``, ``true``, ``pred``, ``confidence``.

    Every image says "correct" or "wrong" in words, so the coloured frame is
    never the only signal.
    """
    n_rows = math.ceil(len(rows) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.7 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, rows):
        correct = row["true"] == row["pred"]
        ax.axis("on")
        ax.imshow(row["image"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(CORRECT_COLOR if correct else WRONG_COLOR)
            spine.set_linewidth(3)
        ax.set_title(
            f"{'correct' if correct else 'wrong'} ({row['confidence']:.2f})\n"
            f"true: {_short(row['true'], 30)}\npred: {_short(row['pred'], 30)}",
            fontsize=9, loc="left",
        )
    if title:
        fig.suptitle(title)
    return _save(fig, path)
