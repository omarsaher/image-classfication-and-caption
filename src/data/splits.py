"""Train/validation split grouped by ``original_id``.

Each augmented copy shares the ``original_id`` of the photo it was made from.
Splitting by that id keeps an image and all its copies on the same side, so a
validation photo is never seen in training through one of its copies.
"""
import json
import random
from collections import defaultdict
from pathlib import Path


def make_grouped_split(original_records, val_per_class, seed):
    """Pick ``val_per_class`` original images of every class for validation."""
    ids_by_label = defaultdict(list)
    for record in original_records:
        ids_by_label[record["label"]].append(record["original_id"])

    rng = random.Random(seed)
    val_ids = []
    for label in sorted(ids_by_label):
        ids = sorted(ids_by_label[label], key=int)
        val_ids.extend(rng.sample(ids, val_per_class))
    return sorted(val_ids, key=int)


def load_or_create_split(split_file, original_records, val_per_class, seed):
    """Reuse the saved split if present, so every run validates on the same images."""
    split_file = Path(split_file)
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return set(saved["val_original_ids"])

    val_ids = make_grouped_split(original_records, val_per_class, seed)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(
            {"seed": seed, "val_per_class": val_per_class, "val_original_ids": val_ids},
            f,
            indent=1,
        )
    return set(val_ids)


def split_records(original_records, augmented_records, val_ids):
    """Training = originals (and copies) outside ``val_ids``; validation = originals in it.

    Augmented copies of validation images are dropped entirely.
    """
    train = [r for r in original_records if r["original_id"] not in val_ids]
    train += [r for r in augmented_records if r["original_id"] not in val_ids]
    val = [r for r in original_records if r["original_id"] in val_ids]
    return train, val
