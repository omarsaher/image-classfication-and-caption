"""Image transforms and normalisation statistics.

The model is trained from scratch, so images are normalised with the mean and
std of our own training images rather than ImageNet's.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms


def build_train_transform(data_cfg, mean, std):
    """Light on-the-fly augmentation: random crop/scale, flip, mild colour jitter.

    Colour jitter stays small because plumage colour is a key cue for
    telling species apart.
    """
    jitter = data_cfg.get("color_jitter", 0.1)
    return transforms.Compose([
        transforms.RandomResizedCrop(
            data_cfg["image_size"], scale=tuple(data_cfg.get("crop_scale", (0.35, 1.0)))
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_eval_transform(data_cfg, mean, std):
    return transforms.Compose([
        transforms.Resize(data_cfg["resize_size"]),
        transforms.CenterCrop(data_cfg["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def compute_mean_std(image_paths, size=224):
    """Per-channel mean and std over images resized to ``size`` x ``size``."""
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    n_pixels = 0
    for path in image_paths:
        image = Image.open(path).convert("RGB").resize((size, size))
        arr = np.asarray(image, dtype=np.float64).reshape(-1, 3) / 255.0
        pixel_sum += arr.sum(axis=0)
        pixel_sq_sum += (arr**2).sum(axis=0)
        n_pixels += arr.shape[0]
    mean = pixel_sum / n_pixels
    std = np.sqrt(pixel_sq_sum / n_pixels - mean**2)
    return mean.tolist(), std.tolist()


def load_or_compute_stats(stats_file, image_paths):
    stats_file = Path(stats_file)
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]

    mean, std = compute_mean_std(image_paths)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({"mean": mean, "std": std, "n_images": len(image_paths)}, f, indent=1)
    return mean, std


def denormalize(tensor, mean, std):
    """Undo ``Normalize`` on a CxHxW tensor and return an HxWxC array in [0, 1]."""
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(image * np.array(std) + np.array(mean), 0.0, 1.0)
