"""Build the classification DataLoaders from a run config."""
import torch
from torch.utils.data import DataLoader

from src.data.cub_dataset import (
    HFCUBTestDataset,
    RecordDataset,
    class_names_from_records,
    read_metadata,
)
from src.data.splits import load_or_create_split, split_records
from src.data.transforms import (
    build_eval_transform,
    build_train_transform,
    load_or_compute_stats,
)
from src.utils.seed import seed_worker


def load_records(data_cfg, seed):
    """Read Task 1.1 metadata and split it into training and validation records."""
    originals = read_metadata(data_cfg["original_images"], data_cfg["original_metadata"])
    augmented = []
    if data_cfg.get("use_offline_augmented", False):
        augmented = read_metadata(data_cfg["augmented_images"], data_cfg["augmented_metadata"])

    val_ids = load_or_create_split(
        data_cfg["split_file"], originals, data_cfg["val_per_class"], seed
    )
    train, val = split_records(originals, augmented, val_ids)
    return train, val, class_names_from_records(originals)


def get_norm_stats(data_cfg, train_records):
    """Mean/std from the training originals only (copies are derived from them)."""
    paths = [r["path"] for r in train_records if r["image_type"] == "original"]
    return load_or_compute_stats(data_cfg["stats_file"], paths)


def _loader(dataset, data_cfg, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = data_cfg.get("num_workers", 4)
    return DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=shuffle,
    )


def build_train_val_loaders(cfg):
    """Return ``(train_loader, val_loader, class_names, (mean, std))``."""
    data_cfg = cfg["data"]
    seed = cfg["experiment"]["seed"]
    train_records, val_records, class_names = load_records(data_cfg, seed)
    mean, std = get_norm_stats(data_cfg, train_records)

    train_ds = RecordDataset(train_records, build_train_transform(data_cfg, mean, std))
    val_ds = RecordDataset(val_records, build_eval_transform(data_cfg, mean, std))
    return (
        _loader(train_ds, data_cfg, shuffle=True, seed=seed),
        _loader(val_ds, data_cfg, shuffle=False, seed=seed),
        class_names,
        (mean, std),
    )


def build_eval_loader(cfg, split, mean, std):
    """Loader over the validation originals or the full Hugging Face test split."""
    data_cfg = cfg["data"]
    seed = cfg["experiment"]["seed"]
    transform = build_eval_transform(data_cfg, mean, std)
    if split == "val":
        _, val_records, _ = load_records(data_cfg, seed)
        dataset = RecordDataset(val_records, transform)
    elif split == "test":
        dataset = HFCUBTestDataset(transform, cache_dir=data_cfg.get("hf_cache_dir"))
    else:
        raise ValueError(f"Unknown split '{split}', expected 'val' or 'test'")
    return _loader(dataset, data_cfg, shuffle=False, seed=seed), dataset
