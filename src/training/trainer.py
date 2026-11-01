"""Generic supervised training loop for image classifiers.

The trainer only needs a model, two DataLoaders and the ``train`` config, so
the from-scratch encoder and a pretrained one go through the same loop.

Per epoch it logs train/val loss and accuracy, keeps the best checkpoint by
validation accuracy, and stops early when validation accuracy has not
improved for ``early_stopping_patience`` epochs.

``last.pt`` holds the full training state after every epoch and is removed when
training finishes. ``train.resume_from`` continues an interrupted run from a
``last.pt``, or from a ``best.pt`` (weights only: the optimizer starts fresh and
the learning rate schedule jumps to that epoch).
"""
import json
import math
import shutil
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.models.checkpoint import save_checkpoint


@dataclass
class TrainResult:
    best_val_acc: float = -1.0  # below any real accuracy, so epoch 1 always checkpoints
    best_epoch: int = -1
    epochs_run: int = 0
    elapsed_sec: float = 0.0
    checkpoint_path: str = None
    history: list = field(default_factory=list)


def build_optimizer(model, lr, weight_decay):
    """AdamW without weight decay on biases and BatchNorm parameters.

    Decaying BN scales and biases only fights normalisation and does not
    regularise the model, so only conv/linear weights are decayed.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 else decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def warmup_cosine(total_steps, warmup_steps):
    """LR factor: linear warmup from ~0 to 1, then cosine decay to 0."""

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, output_dir, logger,
                 device, class_names, norm_stats):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.train_cfg = cfg["train"]
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.device = device
        self.class_names = class_names
        self.norm_stats = norm_stats

        self.max_batches = self.train_cfg.get("max_batches")
        steps_per_epoch = len(train_loader)
        if self.max_batches:
            steps_per_epoch = min(steps_per_epoch, self.max_batches)
        self.steps_per_epoch = steps_per_epoch
        epochs = self.train_cfg["epochs"]

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.train_cfg.get("label_smoothing", 0.0)
        )
        self.optimizer = build_optimizer(
            model, self.train_cfg["lr"], self.train_cfg["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            warmup_cosine(
                epochs * steps_per_epoch,
                self.train_cfg.get("warmup_epochs", 0) * steps_per_epoch,
            ),
        )
        self.use_amp = self.train_cfg.get("amp", True) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)

    def fit(self):
        epochs = self.train_cfg["epochs"]
        patience = self.train_cfg.get("early_stopping_patience", epochs)
        checkpoint_path = self.output_dir / "best.pt"
        last_path = self.output_dir / "last.pt"
        result = TrainResult()
        last_epoch, epochs_without_improvement = self._restore(result, checkpoint_path)
        start = time.time() - sum(row["epoch_time_sec"] for row in result.history)

        for epoch in range(last_epoch + 1, epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.val_loader)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time_sec": time.time() - epoch_start,
            }
            result.history.append(row)
            self.logger.log_metrics(row, step=epoch)
            self._write_history(result.history)
            print(
                f"epoch {epoch:3d} | train loss {train_loss:.3f} acc {train_acc:.3f}"
                f" | val loss {val_loss:.3f} acc {val_acc:.3f}"
            )

            if val_acc > result.best_val_acc:
                result.best_val_acc = val_acc
                result.best_epoch = epoch
                epochs_without_improvement = 0
                if self.train_cfg.get("save_checkpoint", True):
                    save_checkpoint(
                        checkpoint_path, self.model, self.cfg, self.class_names,
                        self.norm_stats, epoch, {"val_acc": val_acc, "val_loss": val_loss},
                    )
                    result.checkpoint_path = str(checkpoint_path)
            else:
                epochs_without_improvement += 1

            if self.train_cfg.get("save_checkpoint", True):
                save_checkpoint(
                    last_path, self.model, self.cfg, self.class_names, self.norm_stats, epoch,
                    {"val_acc": val_acc, "val_loss": val_loss},
                    training_state={
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "scaler": self.scaler.state_dict(),
                        "best_val_acc": result.best_val_acc,
                        "best_epoch": result.best_epoch,
                        "epochs_without_improvement": epochs_without_improvement,
                        "history": result.history,
                    },
                )

            if epochs_without_improvement >= patience:
                print(f"Early stopping: no val improvement for {patience} epochs")
                break

        last_path.unlink(missing_ok=True)
        result.epochs_run = len(result.history)
        result.elapsed_sec = time.time() - start
        self.logger.log_summary({
            "best_val_acc": result.best_val_acc,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "train_time_sec": result.elapsed_sec,
        })
        if result.checkpoint_path:
            self.logger.log_checkpoint(result.checkpoint_path, f"{self.output_dir.name}-best")
        return result

    def _restore(self, result, checkpoint_path):
        """Load ``train.resume_from`` if set; returns (last finished epoch, epochs without improvement)."""
        resume_from = self.train_cfg.get("resume_from")
        if not resume_from:
            return 0, 0

        resume_from = Path(resume_from)
        checkpoint = torch.load(resume_from, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        epoch = checkpoint["epoch"]
        state = checkpoint.get("training_state")
        if state:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            if state["scaler"]:
                self.scaler.load_state_dict(state["scaler"])
            result.best_val_acc = state["best_val_acc"]
            result.best_epoch = state["best_epoch"]
            result.history = state["history"]
            epochs_without_improvement = state["epochs_without_improvement"]
            best_source = resume_from.parent / "best.pt"
        else:
            self._skip_schedule_to(epoch)
            result.best_val_acc = checkpoint["metrics"]["val_acc"]
            result.best_epoch = epoch
            result.history = self._read_history(resume_from.parent, epoch)
            epochs_without_improvement = 0
            best_source = resume_from

        # Keep the best weights in this run's folder in case the new epochs never beat them
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if best_source.exists() and best_source.resolve() != checkpoint_path.resolve():
            shutil.copyfile(best_source, checkpoint_path)
        if checkpoint_path.exists():
            result.checkpoint_path = str(checkpoint_path)
        self._write_history(result.history)

        kind = "full training state" if state else "weights only, new optimizer"
        print(f"Resuming from {resume_from} after epoch {epoch} ({kind}); "
              f"best val acc so far {result.best_val_acc:.4f} at epoch {result.best_epoch}")
        return epoch, epochs_without_improvement

    def _skip_schedule_to(self, epoch):
        self.scheduler.last_epoch = epoch * self.steps_per_epoch - 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # step() normally follows optimizer.step()
            self.scheduler.step()

    @staticmethod
    def _read_history(run_dir, last_epoch):
        path = Path(run_dir) / "history.json"
        if not path.exists():
            print(f"No history.json in {run_dir}: curves will start after epoch {last_epoch}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [row for row in json.load(f) if row["epoch"] <= last_epoch]

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss, correct, seen = 0.0, 0, 0
        progress = tqdm(self.train_loader, desc=f"train {epoch}", leave=False)
        for step, (images, labels) in enumerate(progress):
            if self.max_batches and step >= self.max_batches:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.3f}")
        return total_loss / seen, correct / seen

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, seen = 0.0, 0, 0
        for step, (images, labels) in enumerate(loader):
            if self.max_batches and step >= self.max_batches:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        return total_loss / seen, correct / seen

    def _write_history(self, history):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=1)
