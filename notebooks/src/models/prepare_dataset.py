
from pathlib import Path
import csv

import numpy as np
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm



from pathlib import Path

# This script is located inside src/models/
SCRIPT_DIR = Path(__file__).resolve().parent

# Go up one level to src/, then enter data/
DATA_DIR = SCRIPT_DIR.parent / "data"

# The processed dataset will be saved in src/data/processed/
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUGMENTATIONS_PER_IMAGE = 2

# Load the CUB training dataset
print("Loading dataset...")
ds = load_dataset(
    "Multimodal-Fatima/CUB_train",
    split="train"
)

label_names = ds.features["label"].names

# Augmentation methods
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02,
    ),
])

def is_grayscale(image):
    """Detect images without meaningful color."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)

    return (
        np.abs(rgb[:, :, 0] - rgb[:, :, 1]).max() <= 2
        and np.abs(rgb[:, :, 1] - rgb[:, :, 2]).max() <= 2
        and np.abs(rgb[:, :, 0] - rgb[:, :, 2]).max() <= 2
    )

# Store metadata alongside the images
metadata_file = OUTPUT_DIR / "metadata.csv"

removed = 0
originals = 0
augmented = 0

with metadata_file.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename", "label", "class_name",
            "description", "image_type", "original_id"
        ],
    )
    writer.writeheader()

    for i, row in enumerate(tqdm(ds, desc="Processing")):
        image = row["image"].convert("RGB")

        # 1. Remove grayscale images
        if is_grayscale(image):
            removed += 1
            continue

        label = row["label"]
        class_name = label_names[label]
        original_id = str(row.get("id", i))

        # Make filenames safe
        safe_id = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in original_id
        )

        # 2. Save original color image
        filename = f"{safe_id}_original.jpg"
        image.save(OUTPUT_DIR / filename, quality=95)
        originals += 1

        writer.writerow({
            "filename": filename,
            "label": label,
            "class_name": class_name,
            "description": row.get("description", ""),
            "image_type": "original",
            "original_id": original_id,
        })

        # 3. Create and save augmented copies
        for j in range(AUGMENTATIONS_PER_IMAGE):
            new_image = augment(image)
            new_filename = f"{safe_id}_aug_{j + 1}.jpg"

            new_image.save(
                OUTPUT_DIR / new_filename,
                quality=95
            )
            augmented += 1

            writer.writerow({
                "filename": new_filename,
                "label": label,
                "class_name": class_name,
                "description": row.get("description", ""),
                "image_type": "augmented",
                "original_id": original_id,
            })

print("\nFinished!")
print("Grayscale images removed:", removed)
print("Original color images saved:", originals)
print("Augmented images saved:", augmented)
print("Total output images:", originals + augmented)
print("Output folder:", OUTPUT_DIR)
