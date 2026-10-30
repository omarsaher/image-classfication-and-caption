"""Wire config -> data -> model -> logger -> Trainer for one classification run."""
from pathlib import Path

from src.data.loaders import build_train_val_loaders
from src.models.registry import build_classifier, count_parameters
from src.tracking.logger import build_logger
from src.training.trainer import Trainer
from src.utils.config import save_config
from src.utils.seed import get_device, set_seed


def train_classifier(cfg, job_type="train"):
    """Train one model as described by ``cfg``; returns a ``TrainResult``."""
    set_seed(cfg["experiment"]["seed"])
    device = get_device()
    run_name = cfg["experiment"]["name"]
    output_dir = Path(cfg["experiment"]["output_dir"]) / run_name
    save_config(cfg, output_dir / "config.yaml")

    train_loader, val_loader, class_names, norm_stats = build_train_val_loaders(cfg)
    model = build_classifier(cfg["model"])
    n_params = count_parameters(model)
    print(
        f"Run '{run_name}' on {device} | {n_params:,} parameters | "
        f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images"
    )

    logger = build_logger(cfg, run_name, job_type=job_type)
    logger.log_summary({
        "n_parameters": n_params,
        "n_train_images": len(train_loader.dataset),
        "n_val_images": len(val_loader.dataset),
    })
    try:
        trainer = Trainer(model, train_loader, val_loader, cfg, output_dir, logger,
                          device, class_names, norm_stats)
        result = trainer.fit()
        _save_curves(result.history, output_dir, logger)
    finally:
        logger.finish()
    print(f"Best val acc {result.best_val_acc:.4f} at epoch {result.best_epoch}")
    return result


def _save_curves(history, output_dir, logger):
    # Imported here so training does not depend on matplotlib until the end
    from src.visualization.plots import plot_training_curves

    path = plot_training_curves(history, output_dir / "curves.png")
    logger.log_image("training_curves", path)
