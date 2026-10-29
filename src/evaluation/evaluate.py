"""Evaluate a saved classifier on the validation or test split."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.loaders import build_eval_loader
from src.evaluation.metrics import (
    classification_metrics,
    compute_confusion_matrix,
    per_class_accuracy,
    top_confusions,
)
from src.models.checkpoint import load_classifier
from src.tracking.logger import build_logger
from src.utils.config import apply_overrides
from src.utils.seed import get_device


@torch.no_grad()
def predict_loader(model, loader, device):
    """Return true labels, predicted labels and softmax confidence of the prediction."""
    y_true, y_pred, confidence = [], [], []
    for images, labels in tqdm(loader, desc="predict", leave=False):
        probs = torch.softmax(model(images.to(device)), dim=1)
        conf, pred = probs.max(dim=1)
        y_true.append(labels.numpy())
        y_pred.append(pred.cpu().numpy())
        confidence.append(conf.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(confidence)


def evaluate_checkpoint(checkpoint_path, split="test", overrides=None, output_dir=None):
    """Compute metrics, confusion matrix and per-image predictions for ``split``.

    ``overrides`` (``key=value``) adjust the config stored in the checkpoint,
    e.g. data paths or batch size on another machine.
    """
    from src.visualization.plots import plot_confusion_matrix

    device = get_device()
    model, checkpoint = load_classifier(checkpoint_path, device)
    cfg = apply_overrides(checkpoint["config"], overrides or [])
    class_names = checkpoint["class_names"]
    mean, std = checkpoint["norm_stats"]["mean"], checkpoint["norm_stats"]["std"]

    loader, dataset = build_eval_loader(cfg, split, mean, std)
    if split == "test" and list(dataset.class_names) != list(class_names):
        raise ValueError("Test split class order does not match the training classes")

    y_true, y_pred, confidence = predict_loader(model, loader, device)
    num_classes = len(class_names)
    metrics = classification_metrics(y_true, y_pred, num_classes)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    output_dir = Path(output_dir or Path(checkpoint_path).parent / f"eval_{split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint_path),
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
    cm_path = plot_confusion_matrix(cm, output_dir / "confusion_matrix.png",
                                    title=f"Confusion matrix ({split})")

    logger = build_logger(cfg, f"{cfg['experiment']['name']}_eval_{split}", job_type="eval")
    try:
        logger.log_summary({f"{split}_{k}": v for k, v in metrics.items()})
        logger.log_image(f"{split}_confusion_matrix", cm_path)
    finally:
        logger.finish()

    print(json.dumps(metrics, indent=1))
    print(f"Saved evaluation to {output_dir}")
    return report


def _worst_classes(cm, class_names, k):
    accuracy = per_class_accuracy(cm)
    order = np.argsort(accuracy)[:k]
    return [{"class": class_names[i], "accuracy": float(accuracy[i])} for i in order]


def _write_predictions(path, y_true, y_pred, confidence, class_names):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true", "predicted", "confidence", "correct"])
        for i, (t, p, c) in enumerate(zip(y_true, y_pred, confidence)):
            writer.writerow([i, class_names[t], class_names[p], f"{c:.4f}", int(t == p)])
