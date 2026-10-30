"""Experiment tracking behind one small interface.

Training code only talks to ``ExperimentLogger``; the backend (Weights &
Biases, or nothing for quick local tests) is chosen in the config.
"""
from abc import ABC, abstractmethod


class ExperimentLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics, step=None):
        ...

    @abstractmethod
    def log_summary(self, values):
        ...

    @abstractmethod
    def log_image(self, key, path):
        ...

    @abstractmethod
    def log_checkpoint(self, path, name):
        ...

    @abstractmethod
    def finish(self):
        ...


class NullLogger(ExperimentLogger):
    """Logs nothing; local files (history.json, plots) are still written."""

    def log_metrics(self, metrics, step=None):
        pass

    def log_summary(self, values):
        pass

    def log_image(self, key, path):
        pass

    def log_checkpoint(self, path, name):
        pass

    def finish(self):
        pass


class WandbLogger(ExperimentLogger):
    def __init__(self, tracking_cfg, run_name, config, job_type):
        import wandb

        self.wandb = wandb
        self.upload_checkpoints = tracking_cfg.get("log_checkpoint", True)
        self.run = wandb.init(
            project=tracking_cfg.get("project", "cub-pixels-to-language"),
            entity=tracking_cfg.get("entity"),
            mode=tracking_cfg.get("mode", "online"),
            group=tracking_cfg.get("group"),
            name=run_name,
            job_type=job_type,
            config=config,
        )

    def log_metrics(self, metrics, step=None):
        self.run.log(metrics, step=step)

    def log_summary(self, values):
        for key, value in values.items():
            self.run.summary[key] = value

    def log_image(self, key, path):
        self.run.log({key: self.wandb.Image(str(path))})

    def log_checkpoint(self, path, name):
        self.run.summary["checkpoint_path"] = str(path)
        if self.upload_checkpoints:
            artifact = self.wandb.Artifact(name, type="model")
            artifact.add_file(str(path))
            self.run.log_artifact(artifact)

    def finish(self):
        self.run.finish()


def build_logger(cfg, run_name, job_type="train"):
    tracking_cfg = cfg.get("tracking", {})
    backend = tracking_cfg.get("backend", "none")
    if backend == "wandb":
        return WandbLogger(tracking_cfg, run_name, cfg, job_type)
    if backend == "none":
        return NullLogger()
    raise ValueError(f"Unknown tracking backend '{backend}', expected 'wandb' or 'none'")
