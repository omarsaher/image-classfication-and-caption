"""Saving and loading classifier checkpoints.

A checkpoint holds everything needed to rebuild and use the model on another
machine: weights, the run config, class names and normalisation stats.
"""
from pathlib import Path

import torch

from src.models.registry import build_classifier


def save_checkpoint(path, model, cfg, class_names, norm_stats, epoch, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "class_names": class_names,
            "norm_stats": {"mean": list(norm_stats[0]), "std": list(norm_stats[1])},
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_classifier(path, device):
    """Return ``(model, checkpoint_dict)`` with the model in eval mode on ``device``."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_classifier(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
