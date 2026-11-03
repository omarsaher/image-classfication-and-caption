# From-scratch CNN encoder: recommended architecture

**Status: proposal, not trained yet.** Shapes, parameter counts and compute come from a
prototype run on CPU (forward and backward pass only). No accuracy in this document comes
from this architecture. Section 8 is how we find out.

## 1. Recommendation

**Use a ResNet-50-D: a ResNet-50 with a deep 3×3 stem, average-pool downsampling in the
shortcuts, and light stochastic depth. Train it with the recipe in section 7.**

| | Current (ResNet-18 ×0.5) | Proposed (ResNet-50-D) |
|---|---:|---:|
| Parameters, including head | 2.85 M | 23.94 M |
| Compute at 224 px | 0.52 GMAC | 4.33 GMAC |
| Pooled feature size | 256 | 2048 |
| Best validation accuracy | 61.9% (Run B)\* | not measured |

\* The default config is ×0.5, but tuning could have picked ×1.0 (11.30 M, 2.05 GMAC).
`outputs/` is empty on this machine, so check `config.yaml` from the Kaggle results.

**Changing the architecture alone probably won't be enough.** The training recipe matters at
least as much. The same ResNet-50 on CUB scored 51.9% in one paper and 71.44% in another,
with different recipes (section 2). Test the two changes together, but in a way that shows
how much each one contributes (section 8).

## 2. The evidence

### Our results so far

| Model | Trained from scratch on | Epochs | Val | Test |
|---|---|---:|---:|---:|
| AlexNet (59.1 M) | originals + copies | 80 | 33.7% | 33.1% |
| GoogLeNet, ×1.0 (about 6 M) | originals + copies | 35 | 59.2% | 59.5% |
| ResNet-18, Run A | originals | 100 | 53.2% | not in repo |
| ResNet-18, Run B | originals + copies | 100 | 61.9% | not in repo |

AlexNet has the most parameters and the worst score, so parameter count alone doesn't
predict accuracy here.

### What the ResNet-18 curves show

- **The model memorises the training set.** In Run B, training accuracy passes 90% around
  epoch 20, while validation accuracy is about 45%. The final gap is about 38 points.
- **Training runs out of signal early.** With label smoothing of 0.1 over 200 classes, the
  lowest possible loss is 0.851. Run B's training loss ends at about 0.91, so little gradient
  is left to learn from.
- **More data variety helped.** Run B beat Run A by 8.7 points. The two runs also differ in
  updates per epoch (2.7× more in Run B), so copies may not explain the whole gain.
- **The split is clean.** I checked that none of the 1,677 copies of validation images reach
  training, and that no labels mismatch, so this gain isn't caused by leakage.

The main problem is generalisation, not lack of capacity. That makes a *larger* model a
counter-intuitive choice, so it has to earn its place against the same recipe on ResNet-18
(run R1 in section 8).

### Published from-scratch results on CUB

| Source | Model | Input | Loss | Test |
|---|---|---:|---|---:|
| Brigato et al. 2021 | ResNet-50 | 224 | cross-entropy, tuned | **71.44%**\* |
| Brigato et al. 2021 | Harmonic Networks | 224 | cross-entropy, tuned | 72.26%\* |
| Barz & Denzler 2020 | ResNet-50 | 448 | cross-entropy | 51.9% |
| Barz & Denzler 2020 | ResNet-50 | 448 | cosine + cross-entropy | 68.0% |

\* Balanced accuracy, averaged over runs.

Brigato et al. is the closest match to our task: same dataset, same 224 px input and a similar
crop range. It used SGD for 200 epochs with a small batch size, and tuned the learning rate and
weight decay on a validation split. It concludes that tuning is "immensely undervalued".

**This is not a promise of 71%.** That model trained its final run on all 5,994 images. We
hold out 1,000 for validation and train on 4,989 originals plus copies. My estimate for this
setup is **65–70% on test**. That is a judgement, not a measurement.

## 3. The architecture

### Full network (224 px input)

```text
  Input image   3 × 224 × 224
        │
        ▼
┌──────────────────────────────────────┐
│ STEM                                 │
│ 3×3 conv   32 ch  stride 2  BN ReLU  │
│ 3×3 conv   32 ch  stride 1  BN ReLU  │
│ 3×3 conv   64 ch  stride 1  BN ReLU  │
└───────┬──────────────────────────────┘
        ├──────────────────────────────►  c1     64 × 112 × 112   stride 2
        ▼
  3×3 max pool, stride 2
        │
        ▼
┌──────────────────────────────────────┐
│ STAGE 1    3 × Bottleneck-D          │
│ 1×1, 64  →  3×3, 64  →  1×1, 256     │
│ no downsampling                      │
└───────┬──────────────────────────────┘
        ├──────────────────────────────►  c2    256 × 56 × 56     stride 4
        ▼
┌──────────────────────────────────────┐
│ STAGE 2    4 × Bottleneck-D          │
│ 1×1, 128  →  3×3, 128  →  1×1, 512   │
│ first block has stride 2             │
└───────┬──────────────────────────────┘
        ├──────────────────────────────►  c3    512 × 28 × 28     stride 8
        ▼
┌──────────────────────────────────────┐
│ STAGE 3    6 × Bottleneck-D          │
│ 1×1, 256  →  3×3, 256  →  1×1, 1024  │
│ first block has stride 2             │
└───────┬──────────────────────────────┘
        ├──────────────────────────────►  c4   1024 × 14 × 14     stride 16
        ▼
┌──────────────────────────────────────┐
│ STAGE 4    3 × Bottleneck-D          │
│ 1×1, 512  →  3×3, 512  →  1×1, 2048  │
│ first block has stride 2             │
└───────┬──────────────────────────────┘
        ├──────────────────────────────►  c5   2048 × 7 × 7       stride 32
        ▼
  Global average pool
        ├──────────────────────────────►  pooled   2048
        ▼                                 ── end of encoder ──
  Dropout 0.3
        ▼
  Linear 2048 → 200
        ▼
  200 species logits
```

### One Bottleneck-D block

`m` is the stage's middle width, and the block outputs `4m` channels. The stride `s` is 2 only
in the first block of stages 2–4.

```text
                 x
                 │
        ┌────────┴───────────┐
        │ main path          │ shortcut
        ▼                    ▼
  1×1 conv → m         if the shape changes:
  BN, ReLU               2×2 avg pool (if s = 2)
        ▼                1×1 conv → 4m, BN
  3×3 conv → m         otherwise:
  stride s               identity
  BN, ReLU                   │
        ▼                    │
  1×1 conv → 4m              │
  BN (scale starts at 0)     │
        ▼                    │
  DropPath(p)                │
        │                    │
        └───────► + ◄────────┘
                  │
                 ReLU
```

An optional SE layer (run R3) goes between the last BN and DropPath.

### Sizes by part

| Part | Blocks | Output (224 px) | Output (320 px) | Parameters |
|---|---:|---:|---:|---:|
| Stem | | 64 × 112 × 112 | 64 × 160 × 160 | 0.03 M |
| Stage 1 (`c2`) | 3 | 256 × 56 × 56 | 256 × 80 × 80 | 0.22 M |
| Stage 2 (`c3`) | 4 | 512 × 28 × 28 | 512 × 40 × 40 | 1.22 M |
| Stage 3 (`c4`) | 6 | 1024 × 14 × 14 | 1024 × 20 × 20 | 7.10 M |
| Stage 4 (`c5`) | 3 | 2048 × 7 × 7 | 2048 × 10 × 10 | 14.97 M |
| Linear head | | 200 | 200 | 0.41 M |
| **Total** | **16** | | | **23.94 M** |

With the 3-layer stem this has 52 weighted layers. "50" is just the conventional name.

## 4. Why each part is there

| Part | Why | Evidence |
|---|---|---|
| **Bottleneck blocks, 3-4-6-3** | The backbone behind the published 71.44% | CUB, from scratch |
| **Deep 3×3 stem** | More non-linearity than one 7×7 conv; already in our encoder | ImageNet (ResNet-C) |
| **Avg-pool in shortcuts** | A stride-2 1×1 conv discards 3 of every 4 positions; pooling uses them all | ImageNet (ResNet-D) |
| **Stride on the 3×3, not the first 1×1** | Same reason: no input positions are skipped | ImageNet (ResNet-B) |
| **Zero-initialised last BN** | Each block starts as an identity; already in our encoder | ImageNet |
| **Stochastic depth, 0 → 0.1** | Randomly skips blocks, which regularises a model that overfits quickly | ImageNet, CIFAR |
| **Global average pooling** | Few head parameters, and Grad-CAM keeps working | Standard |
| **Dropout 0.3** | Unchanged from the current model | Our config |

**How strong this evidence is:** only the ResNet-50 backbone has direct support on CUB from
scratch. The other changes are cheap and low-risk, but they are justified by results on other
datasets, and their effect on CUB is untested.

**Receptive field isn't a reason to switch.** The theoretical receptive field is 431 px for
ResNet-50-D and 439 px for ResNet-18, so both already cover the whole image.

## 5. Alternatives considered

| Option | Verdict | Reason |
|---|---|---|
| **ResNet-18** | Keep as the control | Might match ResNet-50-D once it has the better recipe; R1 tests this |
| **ResNet-34** (21.41 M) | Not first choice | Similar cost to ResNet-50-D, and I found no CUB result to back it |
| **ResNet-50-D ×0.5** (6.10 M) | Fallback | Use it if training is too slow or memory is short |
| **SE attention** (26.47 M) | Test, don't assume | Adds about 10% parameters; attention-based Dual Selective Kernel scored 71.02%, no better than the plain baseline |
| **Harmonic Networks** | Skip | Only 0.82 points above the baseline, for a much less standard block |
| **Cosine loss** | Skip | Cosine + cross-entropy scored 70.80% against 71.44% for tuned cross-entropy |
| **AlexNet, VGG-style** | Skip | No residual connections, many parameters; our AlexNet got 33.1% |
| **GoogLeNet** | Skip | Our run got 59.5%, though after only 35 epochs, so it isn't a final verdict |
| **MobileNet, EfficientNet** | Skip | Designed for compute efficiency, not small datasets (my judgement) |
| **ConvNeXt, ViT** | Skip | Usually rely on large datasets and heavy augmentation; about 25 real images per class is a poor fit (my judgement) |

## 6. Reuse for colorization and captioning

The encoder keeps the existing `Encoder` interface in `src/models/encoders/base.py`, so
downstream code doesn't have to care which backbone it gets. The prototype confirmed the
`c1`–`c5` strides of 2, 4, 8, 16 and 32.

| Task | Uses | Note |
|---|---|---|
| Classification | `pooled` (2048) | Same head as today |
| Colorization | `c1`–`c5` as U-Net skips | `in_channels=1` works in the prototype. Reduce `c4` and `c5` with 1×1 convs first, because 1024 and 2048 channels are costly for a decoder |
| Captioning | `c5` as an attention grid | 7 × 7 gives 49 regions of 2048 dimensions; 320 px gives 10 × 10 = 100 |
| Grad-CAM | `stages[-1]` | Unchanged |

**One thing to decide:** an encoder trained on colour images uses colour heavily, and
`in_channels=1` means its first layer can't load RGB weights directly. We need to settle this
when colorization starts, but it doesn't change this design.

## 7. Training recipe

| Setting | Now | Proposed | Why |
|---|---|---|---|
| Training data | Run A or B | **Originals + copies** | Best result so far |
| Optimizer | AdamW | **SGD, momentum 0.9, Nesterov** | Used for the 71.44% result (Nesterov is my addition) |
| Learning rate | 0.001 | **Tune, 0.0007–0.0075** | Range the paper found across its datasets |
| Weight decay | 0.05 | **Tune, 0.0004–0.018** | Same source |
| Batch size | 64 | **32; also try 16** | The paper found small batches work best |
| Schedule | Warmup + cosine | Unchanged | Paper also used cosine |
| Epochs | 100 | **150** | About 2.0 M images seen, against about 1.2 M in the paper |
| Early stopping patience | 20 | **40 or off** | Validation plateaus for long stretches and improves late |
| Crop scale | 0.35–1.0 | Unchanged | Close to the paper's 0.4 minimum |
| Label smoothing | 0.1 | Unchanged | |
| Colour jitter | 0.1 | Unchanged | Plumage colour identifies species |

**Tuning:** the current 12-epoch trials are probably too short to rank SGD settings well. The
paper tuned with full 200-epoch runs, which we can't afford. About 30 epochs per trial is a
reasonable compromise.

**Why not 448 px?** I sampled 300 images: the median short side is 354 px, 85% are at least
320 px, and only 9% reach 448 px. At 448 most images would be upscaled, so **320 px is the
practical limit**. It costs 8.83 GMAC, about twice the compute of 224 px.

## 8. How to test it

Run these one at a time, changing one thing per run. Use the same split and seed throughout.

| Run | Encoder | Recipe | Input | Question it answers |
|---|---|---|---:|---|
| R0 | ResNet-18 | Old | 224 | Baseline: already done, 61.9% val (Run B) |
| R1 | ResNet-18 | New | 224 | How much does the recipe alone add? |
| **R2** | **ResNet-50-D** | **New** | **224** | **How much does the architecture add? Main candidate** |
| R3 | ResNet-50-D + SE | New | 224 | Does attention help? |
| R4 | Best so far | New | 320 | Does extra resolution help? |
| R5 | Best so far + CutMix/Mixup | New | Best so far | Does label mixing reduce the overfitting? |

**How to read the results:**

- **Noise:** with 1,000 validation images, one accuracy has a standard error of about 1.5
  points. Treat a difference **under about 3 points** between two single runs as a tie, and
  keep the simpler option.
- **R2 vs R1:** if R2 doesn't clearly win, keep ResNet-18. It is about 8× cheaper at ×0.5
  width and easier to reuse downstream, and that would be a useful finding for the report.
- **R1 vs R0:** if R1 gains little, the recipe isn't transferring, so check the tuning before
  blaming the architecture.
- **Test set:** make all choices on validation, and evaluate on test only once at the end.
- **Optional last step:** retrain the winner on train + validation for a fixed number of
  epochs, as the paper did. This recovers 1,000 real images, but you lose validation tracking
  for that run.

**Run time:** I haven't measured this. GoogLeNet took about 47 s per epoch on Kaggle on the
same data, which would make 150 epochs about 2 hours. ResNet-50-D may be slower, so time
epoch 1 before committing to a full run.

## 9. Code changes needed

`notebooks/resnet50d_encoder.ipynb` implements runs R1–R4 without changing `src/`. It defines
the encoder and registers it, and swaps the trainer's optimizer for SGD during the session.
It has only been checked in a quick CPU run, so no accuracy results exist yet.

Its default is a 3-hour plan, smaller than the plan in sections 7 and 8: two 20-epoch tuning
trials (learning rates 0.003 and 0.0075), then R2 alone for up to 150 epochs, depending on the
time left. R1 needs a later session, so the architecture and recipe effects stay mixed until then.

If the encoder earns its place, move it into `src/`:

1. **Add** `src/models/encoders/scratch_resnet50d.py`, with `width_mult`, `drop_path` and
   `use_se` options, and **register it** in `src/models/registry.py`. Until then, the
   `src.cli` commands can't load its checkpoints.
2. **Add SGD** to `build_optimizer` in `src/training/trainer.py`, chosen by `train.optimizer`.
   It only supports AdamW today.
3. **Add a config** `configs/cls_resnet50d.yaml`, and a new search space for tuning.
4. **Add CutMix/Mixup** only if we reach R5. The notebook doesn't include it, because it needs
   changes to the training loop.

`ScratchResNet` stays untouched, so R1 and existing checkpoints keep working.

## 10. Open questions

- **Run A and B settings:** which width and hyperparameters did tuning pick? They should be in
  `config.yaml` in the Kaggle results, and R1 needs the same width.
- **ResNet-18 test scores:** they aren't in the repo. GoogLeNet's validation and test scores
  were close, but we should still add the real numbers to section 2.
- **Kaggle GPU quota:** R1–R5 plus tuning could take well over 12 hours in total, so they will
  need separate sessions. `last.pt` resume exists on the `AlexNet` branch, but not on this one.

## Sources

Published results and settings:

- Brigato et al., [Tune It or Don't Use It](https://arxiv.org/abs/2108.13122), ICCV Workshops 2021.
- Barz & Denzler, [Deep Learning on Small Datasets without Pre-Training using Cosine Loss](https://arxiv.org/abs/1901.09054), WACV 2020.

Architecture parts:

- He et al., [Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187), CVPR 2019 (ResNet-B, C and D).
- Huang et al., [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382), ECCV 2016.
- Hu et al., [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507), CVPR 2018.

Our own numbers:

- ResNet-18 curves: `reports/figures/task1_2/runs_A_B_curves.png`.
- AlexNet: `notebooks/dl-phase1-alexnet.ipynb` on the `AlexNet` branch.
- GoogLeNet: `notebooks/kaggle-train-googlenet.ipynb` on the `GoogleNet` branch.
