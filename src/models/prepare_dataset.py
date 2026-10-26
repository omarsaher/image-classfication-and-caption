
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
ORIGINAL_DIR = OUTPUT_DIR / "original" / "images"
AUGMENTED_DIR = OUTPUT_DIR / "augmented" / "images"
ORIGINAL_META_DIR = OUTPUT_DIR / "original" / "metadata"
AUGMENTED_META_DIR = OUTPUT_DIR / "augmented" / "metadata"
for d in (ORIGINAL_DIR, AUGMENTED_DIR, ORIGINAL_META_DIR, AUGMENTED_META_DIR):
    d.mkdir(parents=True, exist_ok=True)

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

MAX_SIMILARITY = 0.90
MAX_ATTEMPTS = 10

def thumbnail(image):
    """Small zero-mean, unit-norm grayscale vector for similarity checks."""
    gray = np.asarray(
        image.convert("L").resize((24, 24)), dtype=np.float32
    ).ravel()
    gray -= gray.mean()
    return gray / (np.linalg.norm(gray) + 1e-6)

def too_similar(candidate, references):
    """True if the candidate is a near-copy of any reference image."""
    return any(
        float(candidate @ ref) > MAX_SIMILARITY for ref in references
    )

def is_grayscale(image):
    """Detect images without meaningful color."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)

    return (
        np.abs(rgb[:, :, 0] - rgb[:, :, 1]).max() <= 2
        and np.abs(rgb[:, :, 1] - rgb[:, :, 2]).max() <= 2
        and np.abs(rgb[:, :, 0] - rgb[:, :, 2]).max() <= 2
    )

# Each image set has its own metadata file, in a metadata/ dir beside images/
FIELDNAMES = [
    "filename", "label", "class_name",
    "description", "image_type", "original_id"
]

removed = 0
originals = 0
augmented = 0

with (
    (ORIGINAL_META_DIR / "metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as original_file,
    (AUGMENTED_META_DIR / "metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as augmented_file,
):

    original_writer = csv.DictWriter(original_file, fieldnames=FIELDNAMES)
    augmented_writer = csv.DictWriter(augmented_file, fieldnames=FIELDNAMES)
    original_writer.writeheader()
    augmented_writer.writeheader()

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
        image.save(ORIGINAL_DIR / filename, quality=95)
        originals += 1

        original_writer.writerow({
            "filename": filename,
            "label": label,
            "class_name": class_name,
            "description": row.get("description", ""),
            "image_type": "original",
            "original_id": original_id,
        })

        # 3. Create and save augmented copies
        seen = [thumbnail(image)]
        for j in range(AUGMENTATIONS_PER_IMAGE):
            # Re-draw until the copy differs from the original and the
            # earlier augmentations (a tiny rotation is a near-duplicate)
            for _ in range(MAX_ATTEMPTS):
                new_image = augment(image)
                new_thumb = thumbnail(new_image)
                if not too_similar(new_thumb, seen):
                    break
            else:
                continue
            seen.append(new_thumb)
            new_filename = f"{safe_id}_aug_{j + 1}.jpg"

            new_image.save(
                AUGMENTED_DIR / new_filename,
                quality=95
            )
            augmented += 1

            augmented_writer.writerow({
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
