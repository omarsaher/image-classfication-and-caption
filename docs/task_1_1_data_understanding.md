# Task 1.1 — Data Understanding & EDA

Dataset: CUB-200-2011 with captions (`Multimodal-Fatima/CUB_train`, 5,994 images, 200 bird species).
This document records what we found, what we did about it, and what is still missing.

**Where the work lives**

| What | File |
|---|---|
| Grayscale detection and class-distribution check (EDA notebook) | `notebooks/visualization.ipynb` |
| Cleaning + offline augmentation script | `src/models/prepare_dataset.py` |
| Cleaned originals + metadata | `src/data/original/images/`, `src/data/original/metadata/metadata.csv` |
| Augmented copies + metadata | `src/data/processed/augmented/images/`, `src/data/processed/augmented/metadata/metadata.csv` |
| Fixed validation split | `splits/val_original_ids.json` |
| Normalisation statistics | `splits/norm_stats.json` |
| Training-time augmentation | `src/data/transforms.py` |

Numbers marked *(recomputed)* were calculated from the committed metadata and images while writing this
document; the rest come from the notebook outputs and the preprocessing script.

---

## 1. Dataset analysis

### Class distribution

| | Raw (5,994) | After cleaning (5,989) |
|---|---:|---:|
| Classes | 200 | 200 |
| Images per class | 30 for most classes, 29 for 6 of them (e.g. Bank Swallow, Artic Tern) | 29 or 30 *(recomputed: min 29, max 30, std 0.23; 11 images short of a perfect 6,000)* |

**Finding:** the dataset is essentially perfectly balanced, so **class imbalance is not a problem here**.
No re-weighting or oversampling is needed. The only issue is the opposite one: with about 30 images per
species (about 25 after validation is held out), the model has very little data per class. That is why
the augmentation strategy (section 4) matters.

### Image sizes and quality *(recomputed)*

| | Width | Height |
|---|---:|---:|
| Min | 121 | 120 |
| Median | 500 | 375 |
| Max | 500 | 500 |

- The median image is 500×375 (aspect ratio about 4:3), so images are not square and need a resize + crop.
- 74 of 5,989 images have a side below 224 px; they get upscaled by the 224 px model input.
- **Grayscale:** the notebook found 4 images stored in mode `L` (`Pelagic_Cormorant_0022`, `Ivory_Gull_0040`,
  `Western_Gull_0002`, `Clark_Nutcracker_0020`). The preprocessing script uses a stricter pixel-level test
  (all three channels equal within ±2), which removed **5** images in total (5,994 → 5,989). Plumage
  colour is a key cue for telling species apart, so grayscale images are removed rather than kept.
- **Corrupted files:** none turned up. The pipeline decodes every image, and no image failed to open.

### Caption statistics *(recomputed from the 5,989 cleaned images)*

Each image has **11 captions**, stored in the single `description` column separated by newlines.

| Statistic | Value |
|---|---:|
| Captions per image | 11 (constant) |
| Total captions | 59,890 |
| Unique captions | 50,438 (so about 16% are exact repeats, mostly short generic ones) |
| Words per caption | mean 15.3, median 14, min 1, max 66 |
| Words per image (all 11 captions) | mean 153, min 110, max 243 |
| Vocabulary (lowercased, letters/apostrophes) | 4,074 words |
| Words that appear only once | 1,873 (46% of the vocabulary) |
| Most frequent words | a, and, bird, with, has, black, this, white, brown, is |

**Finding:** the vocabulary is small and dominated by colour and body-part words (black, white, brown,
bill, wings, ...), which is what a captioning model should learn. Almost half of the vocabulary is
singletons, which matters for captioning (Task 3): a minimum word-frequency threshold will cut the
vocabulary a lot.

### Anomalies

| Anomaly | Found? | Detail |
|---|---|---|
| Class imbalance | No | 29–30 images per class |
| Grayscale images | Yes | 5 removed |
| Corrupted files | None found | |
| Duplicate descriptions between images | No | 0 duplicated `description` values *(recomputed)* |
| Duplicate filenames | No | 0 *(recomputed)* |
| Empty descriptions | No | 0 *(recomputed)* |
| **Noisy captions / typos** | **Yes** | 1,873 singleton words include many misspellings (`agrayish`, `reacing`, `winspan`, `magestic`, `loarge`, `bire`, `ligt`, `hthe`, `premaries`, `eyrings`, `torse`, `ashort`) and a few off-topic captions (`poodle`, `dog`, `parrots`, `cigar`, `zoo`). |
| Near-duplicate augmented copies | Yes, and fixed | see section 4 |

---

## 2. Visualizations & plots

The EDA notebook (`notebooks/visualization.ipynb`) prints the grayscale detection and the per-class count
table (all 200 classes, min 29 / max 30). Training-side plots are produced by `src/visualization/plots.py`
and are described in the Task 1.2 document.

> **Gap:** the committed notebook does not contain plots of image sizes, caption length histograms, or a
> vocabulary/word-frequency chart. The statistics in section 1 were recomputed from the metadata; if the
> report needs these figures, they should be added to `notebooks/visualization.ipynb`
> (a bar chart of images per class, histograms of image width/height and words per caption, and a top-30
> word-frequency bar chart would cover it).

---

## 3. Data cleaning & preprocessing

Performed by `src/models/prepare_dataset.py`:

1. Load `Multimodal-Fatima/CUB_train` (5,994 images).
2. Convert every image to RGB and **drop grayscale images** (5,994 → 5,989).
3. Save each kept image as `<id>_original.jpg` (quality 95) with its metadata row
   `filename, label, class_name, description, image_type, original_id`.
4. Generate augmented copies (section 4) into a separate folder with a separate metadata file, keeping the
   `original_id` so copies can always be traced back to their source image.

**Train / validation split** (`src/data/splits.py`, `splits/val_original_ids.json`): 5 original images per
class (1,000 total) are held out for validation, chosen with a fixed seed and **grouped by `original_id`**,
so no augmented copy of a validation image can leak into training. A check confirmed that none of the 1,677
copies of validation images reach the training set. The official CUB test split (5,794 images) is used only
for the final evaluation.

| Set | Images |
|---|---:|
| Train, originals only | 4,989 |
| Train, originals + augmented copies | 13,396 (4,989 + 8,407) |
| Validation (originals only) | 1,000 |
| Test (Hugging Face `Multimodal-Fatima/CUB_test`) | 5,794 |

**Normalisation:** per-channel mean and std computed over the 4,989 training originals
(`splits/norm_stats.json`): mean = (0.487, 0.501, 0.434), std = (0.229, 0.224, 0.262). The from-scratch
models use these; pretrained models must use ImageNet statistics instead.

### Description column: splitting and cleaning

> **Gap — be careful before claiming this deliverable.** The `description` column is currently saved
> **raw**: 11 captions joined by newlines in one cell. The repository contains **no code that splits it
> into separate captions or cleans it** (lowercasing, spelling correction, removing off-topic captions,
> punctuation handling). The EDA above had to split on newlines by hand. Captions are not needed for
> classification (Task 1.2), so nothing broke, but this must be done before the captioning task.
>
> Suggested step, based on the findings above: split `description` on `\n` into 11 rows/lists per image,
> lowercase and strip, remove very short captions (under 3 words), correct or drop captions containing
> singleton misspellings and the off-topic ones, and rebuild the vocabulary with a minimum frequency.

---

## 4. Data augmentation strategy

Augmentation is applied in two places.

### Offline copies (Task 1.1, `prepare_dataset.py`)

Up to **2 augmented copies per original image**, each produced by:

| Transform | Setting |
|---|---|
| Random horizontal flip | p = 0.5 |
| Random rotation | up to ±15° |
| Colour jitter | brightness 0.15, contrast 0.15, saturation 0.15, hue 0.02 |

Rationale: flips and small rotations are label-preserving for birds, and the colour jitter is deliberately
small because plumage colour identifies species. Each copy keeps the original's label and captions.

**Near-duplicate filter.** A tiny rotation or no-op flip produces an almost identical image, which adds no
information. Each candidate copy is compared (24×24 grayscale, cosine similarity) against the original and
the earlier copy, and is redrawn up to 10 times if the similarity exceeds 0.90; if no sufficiently
different copy is found, it is skipped. Result: **10,084 augmented images** for 5,989 originals
(4,162 originals have 2 copies, 1,760 have 1 copy, 67 have none). This is the "removing duplicate images
from augmented dataset" step in the git history.

### Online augmentation (training time, `src/data/transforms.py`)

| Transform | Setting |
|---|---|
| Random resized crop | 224 px, scale (0.35, 1.0) |
| Random horizontal flip | p = 0.5 |
| Colour jitter | brightness / contrast / saturation 0.1 |

Validation and test images use a deterministic resize to 256 + centre crop to 224.

### Does augmentation help?

Yes, measurably, for the from-scratch ResNet-18: **Run A (originals only) reached 53.2% validation
accuracy; Run B (originals + offline copies) reached 61.9%**, a gain of 8.7 points. Caveat from the
architecture notes: Run B also performs about 2.7× more updates per epoch, so the copies may not explain the
whole gain. See `docs/task_1_2_classification_and_encoder.md` for details.
