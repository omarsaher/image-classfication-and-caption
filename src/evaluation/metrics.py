"""Classification metrics.

- accuracy: share of images whose top prediction is the true species.
- macro precision: for each species, how many of the images predicted as it
  really are it; averaged over species with equal weight.
- macro recall: for each species, how many of its images were found;
  averaged over species with equal weight.
- macro F1: per-species harmonic mean of precision and recall, averaged.
  Macro averaging treats rare and common species equally (CUB is balanced
  in training, nearly balanced in test).
- confusion matrix: counts of (true species, predicted species) pairs; the
  off-diagonal cells show which species get mixed up.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def compute_confusion_matrix(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def per_class_accuracy(cm):
    totals = cm.sum(axis=1)
    return np.divide(np.diag(cm), totals, out=np.zeros(len(cm)), where=totals > 0)


def top_confusions(cm, class_names, k=10):
    """The ``k`` most frequent (true -> predicted) mistakes."""
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    flat = np.argsort(off_diagonal, axis=None)[::-1][:k]
    pairs = []
    for index in flat:
        true_idx, pred_idx = np.unravel_index(index, cm.shape)
        count = int(off_diagonal[true_idx, pred_idx])
        if count == 0:
            break
        pairs.append({
            "true": class_names[true_idx],
            "predicted": class_names[pred_idx],
            "count": count,
        })
    return pairs
