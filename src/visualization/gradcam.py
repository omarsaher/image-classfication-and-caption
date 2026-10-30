"""Grad-CAM (Selvaraju et al., 2017): where the model looks for a given class.

For a target class, the gradient of its score with respect to the feature
maps of the last convolutional stage tells how important each channel is.
Averaging those gradients over space gives one weight per channel; the
weighted sum of the maps, passed through ReLU and upsampled to the image
size, is a heatmap of the regions that raised that class score.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.loaders import build_eval_loader
from src.data.transforms import denormalize
from src.evaluation.evaluate import predict_loader
from src.models.checkpoint import load_classifier
from src.tracking.logger import build_logger
from src.utils.config import apply_overrides
from src.utils.seed import get_device


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients = grad

    def __call__(self, images, class_idx=None):
        """Return heatmaps (N x H x W, values in [0, 1]) and the predicted classes.

        ``class_idx`` picks the class to explain per image; by default the
        predicted one. With BatchNorm in eval mode images do not interact,
        so one backward pass over the summed scores explains the whole batch.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(images)
            predicted = logits.argmax(dim=1)
            target = predicted if class_idx is None else class_idx
            logits.gather(1, target.view(-1, 1)).sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        low = flat.min(dim=1).values.view(-1, 1, 1)
        high = flat.max(dim=1).values.view(-1, 1, 1)
        cam = (cam - low) / (high - low + 1e-8)
        return cam.detach().cpu().numpy(), predicted.detach().cpu()

    def remove(self):
        self._handle.remove()


def run_gradcam(checkpoint_path, split="test", n_correct=8, n_wrong=8,
                overrides=None, output_dir=None):
    """Grad-CAM figures for the most confident correct and wrong predictions.

    Confident mistakes are the most informative: they show whether the model
    looked at the bird (and confused similar species) or at the background.
    For wrong predictions the heatmap is drawn for both the predicted and
    the true class.
    """
    from src.visualization.plots import plot_gradcam_grid

    device = get_device()
    model, checkpoint = load_classifier(checkpoint_path, device)
    cfg = apply_overrides(checkpoint["config"], overrides or [])
    class_names = checkpoint["class_names"]
    mean, std = checkpoint["norm_stats"]["mean"], checkpoint["norm_stats"]["std"]

    loader, dataset = build_eval_loader(cfg, split, mean, std)
    y_true, y_pred, confidence = predict_loader(model, loader, device)

    correct_idx = np.where(y_true == y_pred)[0]
    wrong_idx = np.where(y_true != y_pred)[0]
    correct_idx = correct_idx[np.argsort(-confidence[correct_idx])][:n_correct]
    wrong_idx = wrong_idx[np.argsort(-confidence[wrong_idx])][:n_wrong]

    cam = GradCAM(model, model.gradcam_layer)
    output_dir = Path(output_dir or Path(checkpoint_path).parent / f"gradcam_{split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}
    for kind, indices in (("correct", correct_idx), ("wrong", wrong_idx)):
        if len(indices) == 0:
            continue
        images = torch.stack([dataset[int(i)][0] for i in indices]).to(device)
        pred_cams, _ = cam(images)
        rows = []
        for j, i in enumerate(indices):
            row = {
                "image": denormalize(images[j], mean, std),
                "cam_pred": pred_cams[j],
                "true": class_names[y_true[i]],
                "pred": class_names[y_pred[i]],
                "confidence": float(confidence[i]),
            }
            rows.append(row)
        if kind == "wrong":
            true_cams, _ = cam(images, torch.as_tensor(y_true[indices], device=device))
            for j, row in enumerate(rows):
                row["cam_true"] = true_cams[j]
        figures[kind] = plot_gradcam_grid(rows, output_dir / f"gradcam_{kind}.png",
                                          title=f"Grad-CAM: {kind} predictions ({split})")
    cam.remove()

    with open(output_dir / "selected.json", "w", encoding="utf-8") as f:
        json.dump({"correct": correct_idx.tolist(), "wrong": wrong_idx.tolist()}, f)

    logger = build_logger(cfg, f"{cfg['experiment']['name']}_gradcam_{split}", job_type="gradcam")
    try:
        for kind, path in figures.items():
            logger.log_image(f"gradcam_{kind}", path)
    finally:
        logger.finish()
    print(f"Saved Grad-CAM figures to {output_dir}")
    return figures
