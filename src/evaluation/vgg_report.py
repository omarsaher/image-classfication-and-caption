"""Final evaluation of trained VGGs and the table that compares them.

Run this only after training is finished: this is the one place in the VGG
experiment that loads the test set, and nothing here feeds back into training.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.loaders import build_eval_loader
from src.data.transforms import denormalize
from src.evaluation.evaluate import _worst_classes, _write_predictions
from src.evaluation.metrics import (
    classification_metrics,
    compute_confusion_matrix,
    top_confusions,
)
from src.models.vgg_classifier import load_vgg_checkpoint
from src.training.train_vgg import best_checkpoint_path, run_dir
from src.utils.seed import get_device


@torch.no_grad()
def predict_with_top_k(model, loader, device, k=5, max_batches=None):
    """Return true labels, top predictions, their confidence, and whether the truth is in the top ``k``."""
    model.eval()
    y_true, y_pred, confidence, in_top_k = [], [], [], []
    for step, (images, labels) in enumerate(tqdm(loader, desc="predict", leave=False)):
        if max_batches and step >= max_batches:
            break
        probs = torch.softmax(model(images.to(device)), dim=1)
        top_probs, top_classes = probs.topk(k, dim=1)
        labels = labels.to(device)
        y_true.append(labels.cpu().numpy())
        y_pred.append(top_classes[:, 0].cpu().numpy())
        confidence.append(top_probs[:, 0].cpu().numpy())
        in_top_k.append((top_classes == labels.view(-1, 1)).any(dim=1).cpu().numpy())
    return tuple(np.concatenate(part) for part in (y_true, y_pred, confidence, in_top_k))


def _example_rows(dataset, y_true, y_pred, confidence, class_names, mean, std, seed, n=12):
    """A random, repeatable mix of correct and wrong predictions, for the figure."""
    rng = np.random.default_rng(seed)
    wrong = np.flatnonzero(y_true != y_pred)
    correct = np.flatnonzero(y_true == y_pred)
    n_wrong = min(n // 2, len(wrong))
    n_correct = min(n - n_wrong, len(correct))
    chosen = np.concatenate([
        rng.choice(correct, n_correct, replace=False),
        rng.choice(wrong, n_wrong, replace=False),
    ]).astype(int)
    return [
        {
            "image": denormalize(dataset[int(i)][0], mean, std),
            "true": class_names[y_true[i]],
            "pred": class_names[y_pred[i]],
            "confidence": float(confidence[i]),
        }
        for i in chosen
    ]


def evaluate_vgg(cfg, arch, split="test", max_batches=None):
    """Evaluate the best checkpoint of ``arch``; writes to ``<output_dir>/<arch>/eval_<split>/``.

    ``max_batches`` is only for quick checks. A limited run is marked by its
    ``n_images`` in the report.
    """
    from src.visualization.plots import plot_confusion_matrix
    from src.visualization.vgg_plots import plot_example_predictions

    device = get_device()
    model, checkpoint = load_vgg_checkpoint(best_checkpoint_path(cfg, arch), device)
    class_names = checkpoint["class_names"]
    mean, std = checkpoint["norm_stats"]["mean"], checkpoint["norm_stats"]["std"]

    loader, dataset = build_eval_loader(cfg, split, mean, std)
    if split == "test" and list(dataset.class_names) != list(class_names):
        raise ValueError("Test split class order does not match the training classes")

    y_true, y_pred, confidence, in_top_5 = predict_with_top_k(
        model, loader, device, k=5, max_batches=max_batches
    )
    num_classes = len(class_names)
    metrics = classification_metrics(y_true, y_pred, num_classes)
    metrics["top5_accuracy"] = float(in_top_5.mean())
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    output_dir = run_dir(cfg, arch) / f"eval_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": arch,
        "split": split,
        "n_images": int(len(y_true)),
        **metrics,
        "worst_classes": _worst_classes(cm, class_names, k=10),
        "top_confusions": top_confusions(cm, class_names, k=10),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    np.save(output_dir / "confusion_matrix.npy", cm)
    _write_predictions(output_dir / "predictions.csv", y_true, y_pred, confidence, class_names)
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png",
                          title=f"{arch.upper()} confusion matrix ({split})")
    rows = _example_rows(dataset, y_true, y_pred, confidence, class_names, mean, std,
                         cfg["experiment"]["seed"])
    plot_example_predictions(rows, output_dir / "example_predictions.png",
                             title=f"{arch.upper()} example predictions ({split})")
    print(f"{arch} on {split} ({len(y_true)} images): accuracy {metrics['accuracy']:.4f}, "
          f"top-5 {metrics['top5_accuracy']:.4f}, macro F1 {metrics['macro_f1']:.4f}")
    return report


def compare_models(cfg, architectures=None, split="test"):
    """Build the comparison table, save it as CSV, and draw the comparison figures.

    Models without finished training and evaluation are skipped with a message,
    so you can compare partial results. Returns a pandas DataFrame.
    """
    from src.visualization.plots import plot_run_comparison
    from src.visualization.vgg_plots import plot_accuracy_comparison

    rows, histories = [], {}
    for arch in architectures or cfg["model"]["architectures"]:
        directory = run_dir(cfg, arch)
        paths = [directory / "train_summary.json", directory / f"eval_{split}" / "metrics.json",
                 directory / "history.json"]
        if not all(path.is_file() for path in paths):
            print(f"Skipping {arch}: it has not been trained and evaluated on '{split}' yet")
            continue
        summary, metrics, histories[arch] = (
            json.loads(path.read_text(encoding="utf-8")) for path in paths
        )
        rows.append({
            "model": arch,
            f"{split}_accuracy": metrics["accuracy"],
            f"{split}_top5_accuracy": metrics["top5_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "best_val_acc": summary["best_val_acc"],
            "best_phase": summary["best_phase"],
            "epochs_run": summary["epochs_run"],
            "train_time_min": summary["train_time_sec"] / 60,
            "trainable_params_head_phase": summary["trainable_params_head_phase"],
            "trainable_params_finetune_phase": summary["trainable_params_finetune_phase"],
            "total_params": summary["total_params"],
            "max_batches": summary["max_batches"],
            "n_images": metrics["n_images"],
        })
    if not rows:
        raise FileNotFoundError(f"No finished models found in {cfg['experiment']['output_dir']}")

    table = pd.DataFrame(rows)
    output_dir = Path(cfg["experiment"]["output_dir"])
    table.to_csv(output_dir / f"comparison_{split}.csv", index=False)
    plot_accuracy_comparison(list(table["model"]), list(table[f"{split}_accuracy"]),
                             output_dir / f"{split}_accuracy_comparison.png", split)
    plot_run_comparison(histories, output_dir / "training_comparison.png")
    return table
