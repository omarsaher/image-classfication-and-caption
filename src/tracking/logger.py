"""Experiment tracking behind one small interface.

Training code only talks to ``ExperimentLogger``; the backend (MLflow,
Weights & Biases, or nothing for quick local tests) is chosen in the config.
"""
from abc import ABC, abstractmethod
from pathlib import Path


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


def _flatten(cfg, prefix=""):
    """{"train": {"lr": 0.001}} -> {"train.lr": 0.001}"""
    flat = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


class MlflowLogger(ExperimentLogger):
    """Local MLflow store, no account needed; browse it with ``mlflow ui``."""

    def __init__(self, tracking_cfg, run_name, config, job_type):
        import mlflow

        self.mlflow = mlflow
        self.upload_checkpoints = tracking_cfg.get("log_checkpoint", False)
        uri = tracking_cfg.get("uri", "sqlite:///outputs/mlflow.db")
        if uri.startswith("sqlite:///"):
            Path(uri[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)

        experiment = tracking_cfg.get("project", "cub-pixels-to-language")
        if mlflow.get_experiment_by_name(experiment) is None:
            artifact_dir = Path(tracking_cfg.get("artifact_dir", "outputs/mlartifacts")).resolve()
            mlflow.create_experiment(experiment, artifact_location=artifact_dir.as_uri())
        mlflow.set_experiment(experiment)

        tags = {"job_type": job_type}
        if tracking_cfg.get("group"):
            tags["group"] = tracking_cfg["group"]
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        mlflow.log_params(_flatten(config))

    def log_metrics(self, metrics, step=None):
        self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_summary(self, values):
        # MLflow has no run summary: numbers become metrics, anything else a tag
        for key, value in values.items():
            if isinstance(value, (int, float)):
                self.mlflow.log_metric(key, float(value))
            else:
                self.mlflow.set_tag(key, str(value)[:5000])

    def log_image(self, key, path):
        self.mlflow.log_artifact(str(path), artifact_path="plots")

    def log_checkpoint(self, path, name):
        self.mlflow.set_tag("checkpoint_path", str(path))
        if self.upload_checkpoints:
            self.mlflow.log_artifact(str(path), artifact_path="checkpoints")

    def finish(self):
        self.mlflow.end_run()


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
    if backend == "mlflow":
        return MlflowLogger(tracking_cfg, run_name, cfg, job_type)
    if backend == "wandb":
        return WandbLogger(tracking_cfg, run_name, cfg, job_type)
    if backend == "none":
        return NullLogger()
    raise ValueError(f"Unknown tracking backend '{backend}', expected 'mlflow', 'wandb' or 'none'")
