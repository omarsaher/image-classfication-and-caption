# Fine-grained model: Swin-Base + PIM

**Status: proposal, not trained on the full task yet.** Shapes, parameter counts and head compute
come from a CPU prototype using the real pretrained weights. No accuracy in this document comes
from this exact model on all 200 classes. Section 9 lists what has and hasn't been checked.

**This is not the from-scratch encoder.** It starts from pretrained weights, so it's a separate
experiment and can't replace the from-scratch result if your task requires one.

## 1. Recommendation

**Fine-tune a pretrained Swin Transformer (Swin-Base, 384 px) with the Plug-in Module (PIM) on
top.** Swin finds the features, and PIM finds the small regions that separate similar species.

| | ResNet-50-D (from scratch) | **Swin-Base + PIM** | Swin-Large + PIM |
|---|---:|---:|---:|
| Starting weights | Random | **ImageNet-22k** | ImageNet-22k |
| Input size | 224 | **384** | 384 |
| Parameters | 23.94 M | **90.88 M** | 199.69 M |
| Compute per image | 4.33 GMAC | **about 56 GMAC** | about 113 GMAC |
| Published CUB accuracy | none for this exact model | **none for this exact model** | **92.8%** |
| Fits about 3 hours on a T4? | Yes | **Probably, with up to about 50 epochs** | Probably only about 25 epochs |

**Why not the Swin-Large from the paper?** It has about twice the compute, so at about 3 hours it
gets about half as many epochs. Swin-Base gives up some accuracy for that, and I found no published
number for how much. If the first run has time to spare, Swin-Large is a one-line change.

**Target:** at least 85% on test. My estimate is **88–91%**. That is a judgement, not a
measurement. The reasons it should land below 92.8% are in section 7.

## 2. Why this model

Published CUB results using pretrained weights:

| Model | Accuracy | Source |
|---|---:|---|
| Plain ResNet-50 | 84.5–88.2% | TransFG table, PIM table |
| ResNet-50 + PIM | 89.5% | PIM |
| EfficientNet-B7 + PIM | 89.9% | PIM |
| ViT + PIM | 91.0% | PIM |
| TransFG (ViT-B) | 91.7% | TransFG |
| **Plain Swin-Large** | **91.9%** | PIM |
| **Swin-Large + PIM** | **92.8%** | PIM |
| HERBS (Swin) | 93.1% | HERBS |

Three things follow from this table:

- **The backbone does most of the work.** Plain Swin-Large already reaches 91.9%, and PIM adds 0.9
  points. So a plain Swin-Base is a useful control (section 8).
- **HERBS isn't worth it here.** It adds 0.3 points over PIM, but needs an extra bottom-up path,
  nine classifiers, a temperature-based loss and 80 epochs. That doesn't fit in 3 hours.
- **The paper's "Swin-T" is really Swin-Large.** The authors' code labels it `swin-t` but builds
  `swin_large_patch4_window12_384_in22k`. A real Swin-Tiny has 27.5 M parameters and would not
  be the model behind that 92.8%.

## 3. The architecture

### Full network (384 px input, measured from the prototype)

```text
  Input image   3 × 384 × 384
        │
        ▼
┌──────────────────────────────────────┐
│ SWIN-BASE (pretrained)               │
│                                      │
│ Patch embedding: 4×4 patches         │
│         ▼                            │
│ Stage 1    2 Swin blocks             │──►  128 × 96 × 96     stride 4
│         ▼  patch merging             │
│ Stage 2    2 Swin blocks             │──►  256 × 48 × 48     stride 8
│         ▼  patch merging             │
│ Stage 3    18 Swin blocks            │──►  512 × 24 × 24     stride 16
│         ▼  patch merging             │
│ Stage 4    2 Swin blocks             │──► 1024 × 12 × 12     stride 32
└──────────────────────────────────────┘
        │  all four stage maps
        ▼
┌──────────────────────────────────────┐
│ FEATURE PYRAMID                      │
│ Each map: 1×1 conv, ReLU, 1×1 conv   │
│ All maps become 512 channels         │
│ Top-down: upsample deeper map, add   │
└──────────────────────────────────────┘
        │  512 × 96², 48², 24², 12²
        ▼
┌──────────────────────────────────────┐
│ SELECTOR (one classifier per stage)  │
│ Scores every location for 200 species│
│ Keeps the most confident locations:  │
│                                      │
│ Stage    Locations    Kept   Dropped │
│   1        9,216      256     8,960  │
│   2        2,304      128     2,176  │
│   3          576       64       512  │
│   4          144       32       112  │
└──────────────────────────────────────┘
        │ 480 kept, 512-d       │ dropped
        ▼                       ▼
┌────────────────────────┐   pushed to "no
│ GRAPH COMBINER         │   species" by the
│ 480 → 15 graph nodes   │   background loss
│ Graph convolution      │
│ 15 → 1, Dropout 0.1    │
│ Linear 512 → 200       │
└───────────┬────────────┘
            ▼
  200 species logits
```

### One pair of Swin blocks

Swin blocks always come in pairs. Attention over the whole image would cost too much at this
resolution, so each block only looks inside a 12 × 12 window.

```text
             tokens
               │
    ┌──────────┴───────────┐
    │  LayerNorm           │
    │  W-MSA: attention    │   Block 1: fixed windows
    │  inside each window  │
    └──────────┬───────────┘
               + ◄── skip
    ┌──────────┴───────────┐
    │  LayerNorm, MLP      │
    └──────────┬───────────┘
               + ◄── skip
    ┌──────────┴───────────┐
    │  LayerNorm           │
    │  SW-MSA: attention   │   Block 2: windows shifted by 6,
    │  in shifted windows  │   so information crosses window edges
    └──────────┬───────────┘
               + ◄── skip
    ┌──────────┴───────────┐
    │  LayerNorm, MLP      │
    └──────────┬───────────┘
               + ◄── skip
               ▼
```

Patch merging between stages halves the height and width and doubles the channels, which gives
Swin the same pyramid shape as a CNN.

### Sizes by part

| Part | Output | Parameters |
|---|---:|---:|
| Swin-Base backbone | 4 maps, from 128 × 96 × 96 to 1024 × 12 × 12 | 86.88 M |
| Pyramid, stage classifiers, combiner | 200 logits | 4.01 M |
| **Total** | | **90.88 M** |

Compute is 47.1 GMAC for the backbone (Swin's published figure, which its table lists as FLOPs) and
8.81 GMAC for the new parts (measured). The saved model is about 364 MB.

## 4. What each part does

| Part | Why it's there |
|---|---|
| **Pretrained Swin** | Brings features learned from millions of images, which about 25 photos per species can't provide. Most of the accuracy comes from here. |
| **384 px input** | Species often differ by a small mark, like an eye ring or wing bar, and those need enough pixels to be visible. |
| **Feature pyramid** | Early stages hold fine detail and late stages hold meaning. Adding deeper maps into shallower ones gives detailed locations some context. |
| **Stage classifiers** | Score every location, which gives the selector something to rank. This is weakly supervised: nobody marks where the beak is, and only the species label is used. |
| **Selector** | Keeps the locations that look most like a specific species, and drops the rest. This is the part aimed specifically at fine-grained differences. |
| **Graph combiner** | Learns how the kept regions relate, like a head marking together with a wing pattern. It produces the final prediction. |

### The losses

A plain classifier needs one loss, and PIM needs three, because each part has its own job:

```text
total = 0.5 × blocks  +  5 × background  +  1 × combiner
```

| Loss | What it does | Calculation |
|---|---|---|
| **Blocks** | Each stage must predict the species | Cross-entropy on that stage's average location logits, for each of the 4 stages |
| **Background** | Dropped locations must not claim any species | Mean squared error between `tanh(logits)` and −1 |
| **Combiner** | The final prediction must be correct | Cross-entropy on the combiner's logits |

Starting values measured in the prototype: blocks 21.3, background 4.3, combiner 4.7. That makes
the total about 36.7. **So a first loss near 37 is expected**, not a sign of a problem.

## 5. Settings: paper vs authors' code

The paper and the authors' CUB config disagree in several places, so I had to choose:

| Setting | Paper | Authors' config | Proposed | Why |
|---|---|---|---|---|
| Backbone | "Swin-T" | Swin-Large | **Swin-Base** | Fits the time limit |
| Pyramid channels | not given | 1536 | **512** | The authors' code default. Cuts head compute from 65 to 8.8 GMAC |
| Locations kept | 256, 128, 64, 32 | 2048, 512, 128, 32 | **256, 128, 64, 32** | Paper's best setting; smaller |
| Blocks loss weight | 1 | 0.5 | **0.5** | The config is what actually ran |
| Batch size | 8 | 16, updating every 2 | **16, updating every 2** | Use 8, updating every 4, if memory runs out |

I also upsample with bilinear interpolation on the 2D maps, a standard feature-pyramid step. The
authors' Swin setup instead uses a 1D convolution across the flattened locations.

## 6. Training recipe

| Setting | Value | Note |
|---|---|---|
| Optimizer | SGD, momentum 0.9, Nesterov | Authors' code |
| Learning rate | 0.0005 on every layer | Cosine decay, with warmup over 800 batches |
| Weight decay | 0.0005 | |
| Epochs | Up to 50 | Fewer if the time left requires it, as in the current notebook |
| Label smoothing | None | The authors' code doesn't use it |
| **Normalisation** | **ImageNet: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)** | **Must change: `splits/norm_stats.json` won't match the pretrained weights** |
| Resize and crop | Resize to 510, then a 384 crop | Random crop in training, center crop in evaluation |
| Interpolation | Bicubic | From the pretrained weights' config |
| Augmentation | Flip and Gaussian blur | From the paper |
| Training data | **Originals only (4,989 images)** | About 2.7× faster per epoch than with the copies |

**The copies are left out to save time.** They helped from-scratch training, but I don't know if
they'd help here. If the first run has time to spare, this is worth testing.

## 7. What to expect

The 92.8% isn't a fair target for this setup, for four reasons:

| Difference from the paper | Likely effect |
|---|---|
| Swin-Base instead of Swin-Large | Lower; size unknown |
| 4,989 training images instead of 5,994 | Lower: we keep 1,000 for validation |
| Possibly fewer than 50 epochs | Lower if it can't finish |
| Testing on a clean test set | Lower: the authors' config uses the test folder as its validation set, so if they kept the best epoch from it, their score would be slightly optimistic |

**A caution for the report:** the [CUB page](https://www.vision.caltech.edu/datasets/cub_200_2011)
warns that its images overlap with ImageNet, so pretrained models may have seen some of its test
images. [One recent check](https://arxiv.org/abs/2602.23358) found no test duplicates against
ImageNet-21k, so the effect is probably small, but I haven't verified that paper's method. Mention
this next to your result.

## 8. How to test it

| Run | Model | Question it answers | When |
|---|---|---|---|
| **S1** | **Swin-Base + PIM** | **Does it reach 85%? Main run** | First session |
| S0 | Swin-Base, global pooling, one linear layer | How much does PIM really add? | Next session |
| S2 | Swin-Large + PIM | Is the bigger backbone worth the time? | Only if needed |

**How to read the results:**

- **Noise:** at about 90%, one accuracy on 1,000 validation images has a standard error of about 1
  point. Treat a gap **under about 2 points** between two runs as a tie, and keep the simpler option.
  PIM's published gain of 0.9 points is smaller than that, so **S0 and S1 may well tie**. If they do,
  that's a useful finding: the plain model is simpler and easier to reuse.
- **Test set:** choose between runs on validation only, then evaluate the final choice once on test.
- **First epochs:** pretrained models climb much faster than the from-scratch ones. If validation
  accuracy isn't well above 30% within about 5 epochs, stop and investigate. That threshold is my
  judgement, not a published number.

**Run time hasn't been measured.** My rough guess is about 30 images per second on a T4, or about 3
minutes per epoch, but it could easily be off by a factor of 2. The notebook should measure this in
the first epoch and set the epoch count from it.

## 9. What has been checked

| Check | Result |
|---|---|
| Pretrained weights load | Yes, `timm/swin_base_patch4_window12_384.ms_in22k` |
| All shapes in section 3 | Measured |
| Every parameter gets a gradient | Yes |
| Pretrained features separate birds | 12 of 12 held-out images, nearest class mean, 4 easy species |
| New parts can learn | 92% train, 75% held-out, 4 species, 3 images each, with a frozen backbone |
| Grad-CAM gives non-empty maps | Yes, on real birds; I didn't judge whether they highlight the right regions |
| **Full 200-class accuracy** | **Not tested** |
| **GPU memory and speed** | **Not tested: this machine has no GPU** |
| **Fine-tuning the backbone at the proposed settings** | **Not shown to work:** 15 steps was too few to learn more than the class prior |

**Two behaviours to watch in the real run**, both seen in these small tests:

- The combiner's loss sat flat near ln(4) for about 45 steps before it started to separate classes.
  A slow start may be normal.
- Evaluation accuracy lagged training loss badly at times: at one point the combiner's training loss
  was 0.09 while its evaluation accuracy was 25%. BatchNorm statistics settling on very few images
  may explain it, but I haven't confirmed that. If validation stays far below training on the real
  data, check this first.

## 10. What the notebook will need

- **Add `timm`:** it isn't in `requirements.txt`.
- **New training step:** the project's `Trainer` expects a single tensor of logits and one loss. PIM
  needs three losses and gradient accumulation.
- **New transforms:** the ImageNet normalisation, resize, crop and blur from section 6.
- **Memory check first:** run one batch at 16, and fall back to 8 if it runs out of memory.

### Reuse for colorization and captioning

This model doesn't fully match the project's `Encoder` contract:

| Topic | Contract or need | This model | What to do |
|---|---|---|---|
| Stride-2 map (`c1`) | Required | **Doesn't exist:** the first map is stride 4 | Add a small conv stem, or start the decoder at stride 4 |
| `c2`–`c5` | NCHW maps | NHWC from the backbone | Permute them, as the prototype does |
| Input size | Any | **Only 384:** the built model rejected 224 and 448 | Resize inputs to 384 |
| Grayscale input | 1 channel | Pretrained for 3 channels | Copy the gray channel 3 times |
| Grad-CAM layer | NCHW module | Backbone stages output NHWC | Use the stage 4 projection, as tested |
| Captioning | Region grid | 12 × 12 = 144 regions, 1024-d | Works as is |

Colorization is the weakest fit, and a CNN backbone would suit its decoder better. That doesn't
affect classification, but it's worth settling before making this the project's shared encoder.

## Sources

Models and results:

- Chou et al., [A Novel Plug-in Module for Fine-Grained Visual Classification](https://arxiv.org/abs/2202.03822), 2022, and the [authors' code](https://github.com/chou141253/FGVC-PIM).
- Chou et al., [Fine-grained Visual Classification with High-temperature Refinement and Background Suppression](https://arxiv.org/abs/2303.06442) (HERBS), 2023.
- He et al., [TransFG](https://arxiv.org/abs/2103.07976), AAAI 2022.
- Liu et al., [Swin Transformer](https://arxiv.org/abs/2103.14030), ICCV 2021, and the [compute figures](https://github.com/microsoft/Swin-Transformer).

Weights and data:

- [Pretrained Swin-Base weights](https://huggingface.co/timm/swin_base_patch4_window12_384.ms_in22k) on Hugging Face.
- [CUB-200-2011 dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011), including the ImageNet overlap warning.
- [A Dataset is Worth 1 MB](https://arxiv.org/abs/2602.23358), 2026, for the duplicate check.
