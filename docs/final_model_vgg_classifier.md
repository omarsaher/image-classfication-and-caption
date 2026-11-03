# Final Model — Pretrained VGG Classifier

This document explains the model we chose for bird classification: **an ImageNet-pretrained VGG
(VGG13 / VGG16 / VGG19) used as the CNN encoder, with a new 200-class layer on top**. It follows one image
from pixels to a species prediction, then explains how the model is trained, evaluated and explained.

**Code map**

| Piece | File |
|---|---|
| Encoder (VGG body, pretrained) | `src/models/encoders/vgg.py` |
| Encoder interface | `src/models/encoders/base.py` |
| Classifier = encoder + head | `src/models/classifier.py` |
| Build / freeze / unfreeze helpers | `src/models/vgg_classifier.py` |
| Two-phase training | `src/training/train_vgg.py`, `src/training/trainer.py` |
| Test evaluation and comparison | `src/evaluation/vgg_report.py`, `src/visualization/vgg_plots.py` |
| Grad-CAM | `src/visualization/gradcam.py` |
| Settings | `configs/vgg_classification.yaml` |
| Runner notebook | `notebooks/vggclassifier.ipynb` |

> **Results.** The runner notebook `notebooks/vggclassifier.ipynb` was executed on a Tesla T4 and holds the
> results quoted in section 8. **VGG16 is the best model: 69.3% test top-1, 92.0% top-5.** This is clearly
> above the best from-scratch model (GoogLeNet, 59.5%) but below our 85% working target (see section 8).

---

## 1. Why this model

- **Transfer learning fits the data.** We have about 25 training images per species. Our from-scratch
  models memorise the training set (GoogLeNet: 99% train vs 59% validation accuracy; best test accuracy
  59.5%). A VGG that already learned edges, textures, shapes and object parts from 1.2 M ImageNet images
  only has to learn *which combination of those cues identifies each bird species*.
- **It is simple and transparent.** VGG is a plain stack of 3×3 convolutions with no skip connections or
  branches, so every stage's shape and role is easy to explain and to inspect with Grad-CAM.
- **It plugs into the rest of the project.** The VGG body sits behind the common `Encoder` interface, which
  also exposes the multi-scale feature maps `c1`..`c5`. The same encoder can later feed colorization (which
  needs the intermediate maps) and captioning (which needs the pooled vector).
- **Honest trade-off.** VGG is large (130–140 M parameters, almost all in the fully connected layers),
  has no BatchNorm, and is slower to train than modern backbones. We accept this for its simplicity, and
  because three depths (13, 16, 19) can be compared directly.

---

## 2. The big picture: from image to species

```
 image 3×224×224
       │
       ▼
 ┌──────────────────────────  CNN ENCODER (VGGEncoder)  ───────────────────────────┐
 │  features   5 blocks of 3×3 conv + ReLU, each block ends with a 2×2 max-pool      │
 │  avgpool    adaptive average pool to 7×7                                           │
 │  flatten    512×7×7 = 25,088 numbers                                               │
 │  fc         Linear 25088→4096, ReLU, Dropout, Linear 4096→4096, ReLU               │
 └───────────────────────────────────────┬────────────────────────────────────────────┘
                                         │  "pooled" feature vector, 4096 numbers
                                         ▼
 ┌──────────────────────  CLASSIFIER HEAD (ImageClassifier.head)  ─────────────────┐
 │  Dropout(0.5) → Linear 4096→200                                                    │
 └───────────────────────────────────────┬────────────────────────────────────────────┘
                                         ▼
                              200 logits  →  softmax  →  species probabilities
```

In code, this is two tiny classes:

```python
class ImageClassifier(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0.0):
        self.encoder = encoder                                   # VGGEncoder
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(encoder.feature_dim, num_classes))
    def forward(self, x):
        return self.head(self.encoder(x))
```

The whole model is **exactly the official torchvision VGG, with only its last layer replaced**: the
original `Linear(4096, 1000)` (1000 ImageNet classes) is swapped for `Dropout + Linear(4096, 200)`
(200 bird species). Everything else keeps its pretrained ImageNet weights as the starting point.

---

## 3. The CNN encoder, stage by stage

An encoder turns a picture into numbers that describe it. VGG does this in two parts: a **convolutional
body** that builds spatial feature maps, and **fully connected layers** that compress them into one vector.

### 3.1 Convolutional body (`features`)

Every convolution is 3×3 with padding 1 (the size stays the same) followed by ReLU. After each block a
2×2 max-pool halves the height and width. Shapes below are for VGG16 and a 224×224 input:

| Block | Convs (VGG16) | Output name | Output shape | Stride | What it tends to detect |
|---|---:|---|---|---:|---|
| 1 | 2 × 64 ch | `c1` | 64 × 112 × 112 | 2 | edges, colour blobs |
| 2 | 2 × 128 ch | `c2` | 128 × 56 × 56 | 4 | corners, simple textures |
| 3 | 3 × 256 ch | `c3` | 256 × 28 × 28 | 8 | feather patterns, repeated textures |
| 4 | 3 × 512 ch | `c4` | 512 × 14 × 14 | 16 | object parts: beak, eye, wing |
| 5 | 3 × 512 ch | `c5` | 512 × 7 × 7 | 32 | whole-bird, pose-level structure |

VGG13 and VGG19 use the same five blocks with 2 or 4 convolutions in the deeper blocks instead
(VGG13: 2-2-2-2-2, VGG16: 2-2-3-3-3, VGG19: 2-2-4-4-4).
The early blocks capture generic low-level cues that transfer to any image task; the later blocks are more
specific, which is why fine-tuning matters for fine-grained birds (section 5).

Why small 3×3 kernels: two stacked 3×3 convolutions see the same area as one 5×5 but with fewer parameters
and an extra non-linearity; three see a 7×7 area. Depth, not kernel size, gives VGG its power.

### 3.2 Pooling and the fully connected layers

1. **`avgpool`**: adaptive average pooling to 7×7. For a 224×224 input the map is already 7×7, so it changes
   nothing; it only makes the model tolerate other input sizes.
2. **`flatten`**: 512 × 7 × 7 → a vector of 25,088 numbers.
3. **`fc`** = the first five layers of VGG's original classifier:
   `Linear(25088→4096) → ReLU → Dropout(0.5) → Linear(4096→4096) → ReLU`.

The output of this part is the **pooled feature vector (4096 numbers)**. This is the image's summary, the
"embedding" that the classifier reads. The encoder deliberately stops here: the original final
`Dropout + Linear(4096→1000)` is dropped (`vgg.classifier[:5]`).

### 3.3 The `Encoder` interface

`VGGEncoder.forward_features(x)` returns a dictionary:

| Key | Content | Used by |
|---|---|---|
| `c1` … `c5` | the feature map after each of the five blocks | colorization / captioning (later tasks) |
| `pooled` | the 4096-d vector | classification head, captioning |

`forward(x)` returns only `pooled`. The attributes `out_channels = [64, 128, 256, 512, 512]` and
`feature_dim = 4096` tell downstream code the sizes without hard-coding them.

---

## 4. The classifier head: from features to species

`ImageClassifier.head` has two layers:

1. **Dropout(0.5)** randomly zeroes half of the 4096 features during training, so the final layer cannot
   rely on any single feature. This is the main defence against over-fitting with so few images per class.
2. **Linear(4096 → 200)**: one weight vector per species (4096 × 200 + 200 = **819,400 parameters**).
   Each output is a *logit*: how strongly the image's feature vector matches that species.

At prediction time, softmax turns the 200 logits into probabilities and the largest is the predicted
species. During training the loss is **cross-entropy with label smoothing 0.1**: the correct species gets
target probability 0.9 and the rest share 0.1, so the model is penalised for over-confidence. This is useful
because bird species are visually close and some labels are genuinely ambiguous.

**Parameter counts** (computed from torchvision's definitions, including the new 200-class layer):

| | Conv body (`features`) | FC layers (4096-d) | New head | **Total** |
|---|---:|---:|---:|---:|
| VGG13 | 9.40 M | 119.55 M | 0.82 M | **129.8 M** |
| VGG16 | 14.71 M | 119.55 M | 0.82 M | **135.1 M** |
| VGG19 | 20.02 M | 119.55 M | 0.82 M | **140.4 M** |

About 85–92% of the parameters sit in the two FC layers (the first alone, `25088 × 4096`, is about 102.8 M),
which is why the model is heavy and why dropout is applied in two places.

---

## 5. How it is trained: two-phase transfer learning

The pretrained weights are valuable, so they must not be destroyed by large, noisy updates from a randomly
initialised head. Training therefore runs in two phases (`train_vgg.py`, settings in
`configs/vgg_classification.yaml`).

| | Phase 1 — head | Phase 2 — fine-tune |
|---|---|---|
| Convolutional blocks | **frozen** (`requires_grad = False`) | trainable |
| FC layers + new head | trainable | trainable |
| Starts from | ImageNet weights, random new head | best phase-1 weights |
| Learning rate | 3e-4 | 1e-5 (30× smaller) |
| Epochs (max) | 5 | 10 in the YAML; **40 in the executed notebook** |
| Warm-up epochs | 0 | 1 |
| Early-stopping patience | 3 | 4 |

- **Phase 1** lets the new 200-way layer (and the FC layers) adapt to birds while the convolutional features
  stay exactly as ImageNet learned them. Trainable parameters in this phase: the FC layers plus the head
  (about 120.4 M); the conv body (9.4–20.0 M) is frozen.
- **Phase 2** unfreezes everything with a very small learning rate so that the convolutional filters
  gently specialise toward bird-specific cues (bill shapes, plumage patterns) without forgetting their
  general features.
- **Model selection:** the best checkpoint across both phases is chosen by **validation accuracy only**
  and saved to `<checkpoint_dir>/<arch>_best.pt`. The test set is never touched during training.

Experiment tracking: Weights & Biases in the executed notebook (`tracking: wandb`); the YAML default is `none`.

Shared training settings: AdamW (weight decay 0.01, none on biases), linear warm-up then cosine decay of the
learning rate, label smoothing 0.1, mixed precision (only enabled on GPU), batch size 8 in the YAML (**32 in the executed notebook**, on a T4 GPU), seed 42 for every model so the three depths are compared fairly. VGG has no BatchNorm, so freezing
the body has no side effect on running statistics.

**Data.** The same split as every other model: 1,000 validation images (5 per class, grouped by
`original_id`), the official 5,794-image test set, and on-the-fly augmentation (random resized crop 224,
horizontal flip, light colour jitter). The VGG config inherits the data settings from `cls_scratch.yaml`, so
by default it trains on the **4,989 originals only**, and the executed notebook did exactly that (log line: `4989 train / 1000 val images`). Set `data.use_offline_augmented=true` to also use the
Task 1.1 augmented copies (13,396 images), which helped the from-scratch ResNet by 8.7 validation points.

> **Caveat to check.** The pretrained weights were learned with ImageNet normalisation
> (mean 0.485/0.456/0.406, std 0.229/0.224/0.225), but this pipeline normalises with our own dataset
> statistics from `splits/norm_stats.json` (mean ≈ 0.487/0.501/0.434, std ≈ 0.229/0.224/0.262). The values
> are close, and phase 2 can compensate, but switching to ImageNet statistics is the more correct choice for
> a pretrained model and is worth a one-line check against validation accuracy.

---

## 6. Evaluation

`src/evaluation/vgg_report.py` runs once, after training, on the 5,794-image test set for each trained model:

| Metric | Meaning |
|---|---|
| Top-1 accuracy | the single highest-scoring species is correct |
| Top-5 accuracy | the correct species is among the 5 highest scores (useful for look-alike species) |
| Macro precision / recall / F1 | per-class scores averaged over the 200 species, so every species counts equally |
| Confusion matrix | rows = true species, columns = predictions; off-diagonal bright spots are confused pairs |
| Most-confused pairs | listed by name in `metrics.json` |

Selection rule: choose the model with the best **validation** accuracy, then report **its** test accuracy.
`compare_models` writes `comparison_test.csv`, a test-accuracy bar chart and combined training curves for
VGG13 / VGG16 / VGG19.

---

## 7. Explainability with Grad-CAM

For VGG the explained layer is the **last ReLU of block 5**, just before the final max-pool
(`features[-2]`, a 512 × 14 × 14 map for a 224 input).

How it works:

1. Run the image forward and take the score of the predicted species.
2. Back-propagate that score to the 14×14 feature maps.
3. Average each channel's gradients: this is that channel's *importance* for the prediction.
4. Compute the importance-weighted sum of the channels, keep only positive evidence (ReLU).
5. Upsample to 224×224 and overlay as a heatmap.

Bright regions are the parts of the picture that pushed the model toward its answer. For a good bird
classifier they should sit on the head, bill, breast and wing patterns. Heatmaps on wrong predictions show
*why* it failed (background such as water or branches, or a look-alike species' key marking).

```bash
python -m src.cli gradcam --task classification --checkpoint <path to vgg best.pt> --split test
```

---

## 8. Results

Measured in `notebooks/vggclassifier.ipynb` (Tesla T4, seed 42, 4,989 original training images, batch 32,
phase 1 = 5 epochs at lr 3e-4, phase 2 up to 40 epochs at lr 1e-5 with early stopping, patience 4).
Test set: 5,794 images, evaluated once after training.

| Model | Params | Best phase | Epochs run | Train time | Val acc | **Test top-1** | Test top-5 | Macro-F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| VGG13 | 129.8 M | fine-tune | 42 | 28.4 min | 68.1% | 66.83% | 90.47% | 0.667 |
| **VGG16 (chosen)** | 135.1 M | fine-tune | 36 | 28.5 min | **69.6%** | **69.26%** | **92.01%** | **0.691** |
| VGG19 | 140.4 M | fine-tune | 25 | 21.9 min | 69.3% | 68.78% | 91.53% | 0.687 |

**Model selection.** VGG16 has the best validation accuracy, and its test accuracy is also the best, so the
choice is consistent. The gap between VGG16 and VGG19 is 0.5 points of test accuracy (about 29 images out
of 5,794), so the two are close to a tie; VGG13 is about 2.4 points behind. Depth beyond 16 layers did not
help with about 25 images per class.

**What the two phases did (validation accuracy):**

| Model | End of phase 1 (head only) | End of phase 2 (fine-tuned) | Gain |
|---|---:|---:|---:|
| VGG13 | 52.9% | 68.1% | +15.2 |
| VGG16 | 54.9% | 69.6% | +14.7 |
| VGG19 | 54.7% | 69.3% | +14.6 |

Freezing the body gets the model to about 53–55% in 5 epochs; **fine-tuning the convolutional blocks is
worth about 15 points**, so the ImageNet features are good but still benefit from adapting to birds.

**Over-fitting.** VGG16 ends with 81.7% training vs. 69.3% validation accuracy (a gap of about 12 points),
far smaller than the from-scratch models (GoogLeNet: 99% vs 59%; AlexNet: 98% vs 34%). Pretraining is what
keeps the model from simply memorising 25 images per species.

**Comparison with the from-scratch baselines** (test accuracy):

| Model | Initialisation | Test top-1 |
|---|---|---:|
| AlexNet | random | 33.1% |
| GoogLeNet ×1.0 | random | 59.5% |
| ResNet-18 ×0.5, Run B | random | validation 61.9%, test not recorded |
| VGG13 | ImageNet | 66.8% |
| VGG19 | ImageNet | 68.8% |
| **VGG16 (chosen)** | **ImageNet** | **69.3%** |

Transfer learning improves on the best from-scratch model by **+9.8 points** (59.5% → 69.3%), and the
top-5 accuracy of 92.0% shows that the correct species is almost always among the model's first five
guesses, i.e. the remaining errors are mostly between visually similar species.

**Against the 85%+ target.** VGG16 reaches 69.3%, about 16 points short. Published pretrained ResNet-50 and
transformer models reach 84–93% on CUB, so the shortfall is mainly the older VGG backbone and a
training recipe that has not been tuned, not a limit of the approach. Cheap things to try before changing
the architecture, each of which is a one-line config change: train on originals + augmented copies
(`data.use_offline_augmented=true`), normalise with ImageNet statistics (section 5 caveat), tune the
phase-2 learning rate and weight decay, and relax the phase-2 early-stopping patience. The documented next step for a larger jump is the pretrained Swin design in
`docs/fine_grained_model_architecture.md`.

**Still to add to the report:** the confusion matrix and most-confused pairs from the `metrics.json` of
the VGG16 run, and 2–3 correct and wrong Grad-CAM examples with a sentence on what the model attends to
(the notebook's saved figures are in `vgg_results.zip`, not in the repository).

---

## 9. Reproducing

```bash
pip install -r requirements.txt
# open and run notebooks/vggclassifier.ipynb  (set MODELS_TO_TRAIN = ["vgg13", "vgg16", "vgg19"])
# or from code:
python -c "from src.training.train_vgg import load_vgg_config, train_vgg; \
cfg = load_vgg_config('configs/vgg_classification.yaml'); train_vgg(cfg, 'vgg16')"
```

Quick check (one real batch, no files saved; confirms output shape, finite loss, and that frozen blocks stay
fixed in phase 1 and change in phase 2):

```bash
python -c "from src.training.train_vgg import load_vgg_config; \nfrom src.training.vgg_smoke_test import run_smoke_test; \nrun_smoke_test(load_vgg_config('configs/vgg_classification.yaml'))"
```
