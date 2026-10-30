"""Two-phase transfer learning for one VGG, built on the team's ``Trainer``.

Phase 1 ("head"): the convolutional blocks are frozen; only the classifier layers learn.
Phase 2 ("finetune"): starts from the best phase 1 weights, unfreezes everything
and continues with a much smaller learning rate.

The validation set picks the best weights across both phases. The test set is
never loaded here; see ``src/evaluation/vgg_report.py``.
"""
import copy
import gc
import json
from pathlib import Path

from src.data.loaders import build_train_val_loaders
from src.models.vgg_classifier import (
    VGG_ARCHITECTURES,
    build_vgg_classifier,
    freeze_feature_extractor,
    load_vgg_checkpoint,
    parameter_counts,
    unfreeze_feature_extractor,
    vgg_model_config,
)
from src.tracking.logger import build_logger
from src.training.trainer import Trainer
from src.utils.config import apply_overrides, load_config, save_config
from src.utils.seed import get_device, set_seed

PHASE_DIRS = {"head": "phase1_head", "finetune": "phase2_finetune"}


def load_vgg_config(path, overrides=None):
    """Read the VGG config, with the team's data settings underneath its ``data`` section."""
    path = Path(path)
    cfg = load_config(path)
    team_data = load_config(path.parent / cfg.pop("data_config"))["data"]
    cfg["data"] = {**team_data, **cfg.get("data", {})}
    return apply_overrides(cfg, overrides or [])


def validate_data_paths(cfg):
    """Stop early if a data file is missing.

    This matters most for the split and statistics files: the data pipeline
    silently creates new ones when they are absent, which would change the
    team's validation set without any error.
    """
    data_cfg = cfg["data"]
    keys = ["original_images", "original_metadata", "split_file", "stats_file"]
    if data_cfg.get("use_offline_augmented", False):
        keys += ["augmented_images", "augmented_metadata"]
    missing = [f"  {key}: {data_cfg[key]}" for key in keys if not Path(data_cfg[key]).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing data paths (looked inside '{Path.cwd()}'):\n" + "\n".join(missing)
            + "\nRun from the repository root."
        )


def run_dir(cfg, arch):
    return Path(cfg["experiment"]["output_dir"]) / arch


def best_checkpoint_path(cfg, arch):
    return Path(cfg["experiment"]["checkpoint_dir"]) / f"{arch}_best.pt"


def make_run_config(cfg, arch):
    """Turn the experiment config into the config of one model."""
    run_cfg = copy.deepcopy(cfg)
    model_cfg = cfg["model"]
    run_cfg["experiment"]["name"] = arch
    run_cfg["model"] = vgg_model_config(
        arch, model_cfg["num_classes"], model_cfg["dropout"], model_cfg["pretrained"]
    )
    return run_cfg


def _phase_config(run_cfg, phase):
    """The config one ``Trainer`` sees: shared ``train`` settings plus this phase's."""
    cfg = copy.deepcopy(run_cfg)
    cfg["train"] = {**run_cfg["train"], **run_cfg["phases"][phase], "save_checkpoint": True}
    # Saved inside the checkpoint: its weights are complete, so loading it later
    # should rebuild an empty VGG and not read the ImageNet file first.
    cfg["model"]["pretrained"] = False
    return cfg


def _run_phase(model, run_cfg, phase, train_loader, val_loader, class_names, norm_stats, device):
    cfg = _phase_config(run_cfg, phase)
    arch = run_cfg["experiment"]["name"]
    # One logger per phase, because the Trainer restarts its epoch count at 1
    logger = build_logger(cfg, f"{arch}_{phase}", job_type="train")
    try:
        phase_dir = run_dir(run_cfg, arch) / PHASE_DIRS[phase]
        trainer = Trainer(model, train_loader, val_loader, cfg, phase_dir, logger,
                          device, class_names, norm_stats)
        return trainer.fit()
    finally:
        logger.finish()


def _combined_history(head_result, finetune_result):
    """One history for both phases, with epochs counted continuously."""
    history = []
    for phase, result in (("head", head_result), ("finetune", finetune_result)):
        for row in result.history:
            history.append({**row, "phase": phase, "epoch": len(history) + 1})
    return history


def train_vgg(cfg, arch, overwrite=False):
    """Run both phases for one VGG and return its summary dict.

    Writes to ``<output_dir>/<arch>/`` (histories, curves, summary) and saves
    the best weights to ``<checkpoint_dir>/<arch>_best.pt``.
    """
    # Imported here so training does not depend on matplotlib until the end
    from src.visualization.plots import plot_training_curves

    if arch not in VGG_ARCHITECTURES:
        raise ValueError(f"Unknown VGG '{arch}'. Available: {list(VGG_ARCHITECTURES)}")
    best_path = best_checkpoint_path(cfg, arch)
    if best_path.exists() and not overwrite:
        raise FileExistsError(
            f"'{best_path}' already exists. Pass overwrite=True to replace this result."
        )
    validate_data_paths(cfg)

    run_cfg = make_run_config(cfg, arch)
    set_seed(run_cfg["experiment"]["seed"])   # same seed for each model: a fair comparison
    device = get_device()
    output_dir = run_dir(run_cfg, arch)
    save_config(run_cfg, output_dir / "config.yaml")

    train_loader, val_loader, class_names, norm_stats = build_train_val_loaders(run_cfg)
    data_args = (train_loader, val_loader, class_names, norm_stats, device)
    print(f"{arch} on {device} | {len(train_loader.dataset)} train / "
          f"{len(val_loader.dataset)} val images | {len(train_loader)} batches per epoch")

    model = build_vgg_classifier(arch, **{k: run_cfg["model"][k]
                                          for k in ("num_classes", "dropout", "pretrained")})
    freeze_feature_extractor(model)
    head_params = parameter_counts(model)
    print(f"Phase 1 (head): {head_params['trainable']:,} of {head_params['total']:,} parameters train")
    head_result = _run_phase(model, run_cfg, "head", *data_args)

    # Continue from the best phase 1 weights, not from wherever phase 1 stopped.
    # The old model is freed first: two VGGs at once can exhaust RAM on a laptop.
    del model
    gc.collect()
    model, _ = load_vgg_checkpoint(head_result.checkpoint_path, device)
    unfreeze_feature_extractor(model)
    finetune_params = parameter_counts(model)
    print(f"Phase 2 (finetune): {finetune_params['trainable']:,} parameters train")
    finetune_result = _run_phase(model, run_cfg, "finetune", *data_args)
    del model
    gc.collect()

    # Model selection uses validation accuracy only. A checkpoint is ~0.5 GB, so
    # the winner is moved into place and the other phase's copy is deleted.
    finetune_won = finetune_result.best_val_acc > head_result.best_val_acc
    best_phase, best_result = (
        ("finetune", finetune_result) if finetune_won else ("head", head_result)
    )
    loser = head_result if finetune_won else finetune_result
    best_path.parent.mkdir(parents=True, exist_ok=True)
    Path(best_result.checkpoint_path).replace(best_path)
    Path(loser.checkpoint_path).unlink()

    history = _combined_history(head_result, finetune_result)
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=1)
    plot_training_curves(history, output_dir / "curves.png", title=arch.upper())

    summary = {
        "model": arch,
        "device": str(device),
        "best_phase": best_phase,
        "best_val_acc": best_result.best_val_acc,
        "epochs_run": len(history),
        "train_time_sec": head_result.elapsed_sec + finetune_result.elapsed_sec,
        "trainable_params_head_phase": head_params["trainable"],
        "trainable_params_finetune_phase": finetune_params["trainable"],
        "total_params": finetune_params["total"],
        "batch_size": run_cfg["data"]["batch_size"],
        "max_batches": run_cfg["train"].get("max_batches"),  # not null = a quick check, not a real result
    }
    with open(output_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"{arch}: best val acc {best_result.best_val_acc:.4f} from the {best_phase} phase; "
          f"saved to {best_path}")
    return summary
