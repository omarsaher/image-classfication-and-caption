# From Pixels to Language — CUB-200-2011

CSE 485/488 Deep Learning project: one visual encoder used for
(1) bird species classification, (2) image colorization and (3) image captioning
on CUB-200-2011 with captions.

Current state: **Milestone 1** — data preparation (Task 1.1) and the
from-scratch CNN encoder for classification (Task 1.2).

## Project structure

```
configs/                     run settings (YAML)
  cls_scratch.yaml           from-scratch encoder training
  tune_scratch.yaml          grid / random search
notebooks/                   experiments and the Kaggle runner (not part of the main code)
splits/                      fixed validation split and normalisation stats
src/
  data/                      datasets, transforms, train/val split, loaders
  models/                    encoders (scratch_resnet), classifier, registry, checkpoints
  training/                  training loop
  tuning/                    grid and random search
  evaluation/                metrics and checkpoint evaluation
  visualization/             plots and Grad-CAM
  tracking/                  experiment tracking (MLflow; Weights & Biases optional)
  cli.py                     command line entry point
reports/figures/             figures used in the report
outputs/                     run results (not committed)
```

## Setup

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Experiments are tracked with MLflow in a local store (`outputs/mlflow.db`), so no account
is needed. Use `--set tracking.backend=none` to skip tracking, or `tracking.backend=wandb`
to log to Weights & Biases instead.

Data:
- Training/validation images come from Task 1.1 (`src/data/original`, `src/data/processed/augmented`),
  already in the repository.
- The test split (`Multimodal-Fatima/CUB_test`, 5,794 images) is downloaded from Hugging Face
  automatically the first time it is used.

All commands are run from the repository root.

## Task 1.2 — from-scratch CNN encoder

**Model.** ResNet-18-style encoder written from scratch and trained from random weights:
a stem of three 3×3 convolutions, 4 stages of 2 residual blocks, global average pooling,
then dropout and a linear layer for the 200 classes. Default width is half of
ResNet-18 (channels 32-64-128-256, 2.85M parameters); `model.width_mult=1.0` gives
the standard widths (11.3M). The encoder returns the feature maps of every stage
(`c1`..`c5`) so it can be reused for colorization and captioning.

**Data.** 5 original images per class (1,000) are held out for validation, grouped by
`original_id` so augmented copies of a validation image never appear in training.
Training uses either the originals only (4,989 images) or the originals plus the
Task 1.1 augmented copies (13,396 images), set by `data.use_offline_augmented`.

### Training

```bash
# Run A: originals only (default config)
python -m src.cli train --task classification --config configs/cls_scratch.yaml

# Run B: originals + augmented copies
python -m src.cli train --task classification --config configs/cls_scratch.yaml \
    --set experiment.name=scratch_B_augmented data.use_offline_augmented=true
```

Any config value can be changed with `--set key=value`. Each run writes
`outputs/classification/<name>/` with `best.pt`, `history.json`, `curves.png` and `config.yaml`.

### Hyperparameter tuning

```bash
python -m src.cli tune --task classification --config configs/tune_scratch.yaml --method grid
python -m src.cli tune --task classification --config configs/tune_scratch.yaml --method random
python -m src.cli compare-tuning --task classification --results-dir outputs/tuning

# Train with the best hyperparameters found
python -m src.cli train --task classification --config configs/cls_scratch.yaml \
    --best-from outputs/tuning --set experiment.name=scratch_A_originals
```

### Evaluation, Grad-CAM, comparison and inference

```bash
python -m src.cli evaluate --task classification \
    --checkpoint outputs/classification/scratch_A_originals/best.pt --split test

python -m src.cli gradcam --task classification \
    --checkpoint outputs/classification/scratch_A_originals/best.pt --split test

python -m src.cli compare-runs --task classification \
    --runs outputs/classification/scratch_A_originals outputs/classification/scratch_B_augmented \
    --names "A: originals" "B: + augmented copies"

python -m src.cli predict --task classification \
    --checkpoint outputs/classification/scratch_A_originals/best.pt --image path/to/bird.jpg
```

### Experiment tracking

Every run logs its config, per-epoch metrics and plots to MLflow. To browse and compare runs:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
```

then open http://127.0.0.1:5000. Tuning trials are tagged `group=tune_grid` / `tune_random`.

### Running on Kaggle

Open `notebooks/kaggle_train_scratch.ipynb` on Kaggle, enable a GPU and internet,
add the secret `GITHUB_TOKEN`, then use *Save & Run All*. The notebook clones the
repository and runs the CLI commands above; results, including the MLflow store,
are saved in the notebook output (`/kaggle/working/outputs`).

### Resuming an interrupted run

Each run keeps `last.pt` (full training state, removed when training finishes) next to `best.pt`.
To continue on any machine, copy those files and `history.json` into one folder and set `train.resume_from`:

```bash
python -m src.cli train --task classification --config configs/cls_scratch.yaml \
    --set train.resume_from=path/to/last.pt
```

Use `best.pt` if `last.pt` is not available: training continues from that epoch with a new optimizer.
Keep the same `train.epochs` and `data.batch_size` so the learning rate schedule stays on track.

### Quick local check (CPU)

```bash
python -m src.cli train --task classification --config configs/cls_scratch.yaml \
    --set experiment.name=smoke train.epochs=1 train.max_batches=3 \
          data.num_workers=0 data.batch_size=16 tracking.backend=none
```
