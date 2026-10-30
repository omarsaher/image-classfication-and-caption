"""Fast check that each VGG can train, using one real batch and no saved files.

For each model it confirms the output shape, a finite loss, and that one training
step changes exactly the parameters it should: frozen blocks stay the same in
phase 1, and they change in phase 2.
"""
import gc
import math
import time

import torch
from torch import nn

from src.data.loaders import build_train_val_loaders
from src.models.vgg_classifier import (
    build_vgg_classifier,
    freeze_feature_extractor,
    parameter_counts,
    unfreeze_feature_extractor,
)
from src.training.train_vgg import make_run_config, validate_data_paths
from src.training.trainer import build_optimizer
from src.utils.seed import get_device, set_seed


def _training_step(model, images, labels, criterion, lr, weight_decay):
    """One optimizer step; returns the loss, the output shape and the seconds it took."""
    optimizer = build_optimizer(model, lr, weight_decay)
    start = time.time()
    model.train()
    logits = model(images)
    loss = criterion(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item(), tuple(logits.shape), time.time() - start


def smoke_test_model(arch, cfg, images, labels, device):
    """Run every check for one model and return its results; raises if a check fails."""
    run_cfg = make_run_config(cfg, arch)
    model_cfg = run_cfg["model"]
    train_cfg = run_cfg["train"]
    num_classes = model_cfg["num_classes"]
    expected_shape = (images.size(0), num_classes)

    model = build_vgg_classifier(
        arch, num_classes, model_cfg["dropout"], model_cfg["pretrained"]
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.get("label_smoothing", 0.0))
    # One watched layer on each side of the freeze boundary
    first_conv = model.encoder.features[0].weight
    final_layer = model.head[1].weight
    assert final_layer.shape == (num_classes, 4096), f"final layer is {final_layer.shape}"

    # Phase 1: only the classifier layers may change
    freeze_feature_extractor(model)
    head_params = parameter_counts(model)
    conv_before, final_before = first_conv.detach().clone(), final_layer.detach().clone()
    loss, shape, head_seconds = _training_step(
        model, images, labels, criterion, run_cfg["phases"]["head"]["lr"], train_cfg["weight_decay"]
    )
    assert shape == expected_shape, f"output shape {shape}, expected {expected_shape}"
    assert math.isfinite(loss), f"loss is {loss}"
    assert torch.equal(first_conv, conv_before), "a frozen conv layer changed"
    assert not torch.equal(final_layer, final_before), "the final layer did not change"

    # Phase 2: the conv layers must now change as well
    unfreeze_feature_extractor(model)
    all_params = parameter_counts(model)
    conv_before = first_conv.detach().clone()
    _, _, finetune_seconds = _training_step(
        model, images, labels, criterion, run_cfg["phases"]["finetune"]["lr"], train_cfg["weight_decay"]
    )
    assert not torch.equal(first_conv, conv_before), "an unfrozen conv layer did not change"
    assert all_params["trainable"] == all_params["total"], "some parameters stayed frozen"

    result = {
        "model": arch,
        "output_shape": str(list(shape)),
        "loss": round(loss, 3),
        "total_params": all_params["total"],
        "head_phase_trainable": head_params["trainable"],
        "head_step_sec": round(head_seconds, 2),
        "finetune_step_sec": round(finetune_seconds, 2),
    }
    # Free this model before the next one; each needs about 2 GB while training
    del model, first_conv, final_layer, conv_before, final_before
    gc.collect()
    return result


def run_smoke_test(cfg, architectures=None):
    """Check every model on the same batch; returns one result dict per model."""
    validate_data_paths(cfg)
    set_seed(cfg["experiment"]["seed"])
    device = get_device()
    train_loader, _, _, _ = build_train_val_loaders(cfg)
    images, labels = next(iter(train_loader))
    print(f"Device: {device} | batch: images {list(images.shape)}, labels {list(labels.shape)} | "
          f"{len(train_loader)} batches per epoch")
    images, labels = images.to(device), labels.to(device)

    results = []
    for arch in architectures or cfg["model"]["architectures"]:
        results.append(smoke_test_model(arch, cfg, images, labels, device))
        print(f"PASS {results[-1]}")
    print(f"Smoke test passed for {len(results)} model(s).")
    return results
