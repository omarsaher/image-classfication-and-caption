"""CUB-200-2011 datasets for classification.

Training/validation images come from the team's Task 1.1 output: colour
originals plus offline augmented copies, each described by a metadata.csv
with ``filename, label, class_name, original_id`` columns. The test images
come straight from the Hugging Face test split, which Task 1.1 left as is.
"""
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

HF_TEST_NAME = "Multimodal-Fatima/CUB_test"


def read_metadata(images_dir, metadata_csv):
    """Return one record per image listed in a Task 1.1 metadata file."""
    images_dir = Path(images_dir)
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        return [
            {
                "path": images_dir / row["filename"],
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "original_id": row["original_id"],
                "image_type": row["image_type"],
            }
            for row in csv.DictReader(f)
        ]


def class_names_from_records(records):
    names = {}
    for record in records:
        names[record["label"]] = record["class_name"]
    return [names[i] for i in sorted(names)]


class RecordDataset(Dataset):
    """Images on disk, given as records with a ``path`` and a ``label``."""

    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record["label"]


class HFCUBTestDataset(Dataset):
    """The CUB test split from Hugging Face; every image is kept.

    Unlike the training set, grayscale test images are not removed (only
    converted to 3 channels) so results stay comparable with published ones
    that use the full 5,794-image test set.
    """

    def __init__(self, transform=None, cache_dir=None):
        from datasets import load_dataset

        self.ds = load_dataset(HF_TEST_NAME, split="test", cache_dir=cache_dir)
        self.class_names = self.ds.features["label"].names
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = self.ds[index]
        image = row["image"].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, row["label"]
