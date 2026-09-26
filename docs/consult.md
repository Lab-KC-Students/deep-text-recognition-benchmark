# Scene Text Recognition (STR) — Model Decoupling, Ablation, and Transformation Notes

Source paper: *"What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis"* (Baek et al., ICCV 2019, Clova AI / NAVER-LINE) — [arXiv:1904.01906](https://arxiv.org/abs/1904.01906)
Official code: [clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

---

## 1. Where the pretrained model comes from

The `TPS-ResNet-BiLSTM-Attn` (nicknamed **TRBA**) and `TPS-ResNet-BiLSTM-CTC` checkpoints are downloadable directly from the official repo's README. They were trained on the union of **MJSynth (8.9M)** and **SynthText (5.5M)** — 14.4M images total — using AdaDelta, batch size 192, 300K iterations, on a single NVIDIA Tesla P40.

### Where each stage's *method* originates

The model name is a literal concatenation of the paper's four-stage framework. Each stage is itself a citation to prior work:

| Stage | Module | Original paper |
|---|---|---|
| Transformation | **TPS** | Shi et al., *"Robust Scene Text Recognition with Automatic Rectification"* (RARE, CVPR 2016) — built on Jaderberg et al.'s Spatial Transformer Networks (NeurIPS 2015) |
| Feature extraction | **ResNet** | He et al., *"Deep Residual Learning for Image Recognition"* (CVPR 2016) — the paper uses the specific 29-layer variant from Cheng et al.'s FAN (ICCV 2017) |
| Sequence modeling | **BiLSTM** | Standard bidirectional LSTM, used for STR in Shi et al.'s CRNN (TPAMI 2017) |
| Prediction | **Attn** | Bahdanau et al. attention (ICLR 2015), adapted for STR in FAN / EP / AON |

The paper's own contribution isn't inventing these — it's showing all 24 combinations (2 Transformation × 3 Feature Extraction × 2 Sequence Modeling × 2 Prediction) can share one framework, which is exactly why the codebase is convenient to modify.

---

## 2. The four-stage architecture (from `model.py`)

```python
class Model(nn.Module):
    def __init__(self, opt):
        # Transformation: TPS or None
        # FeatureExtraction: VGG | RCNN | ResNet  -> outputs [b, c, h, w]
        # AdaptiveAvgPool over height -> [b, w, c]  (sequence of columns)
        # SequenceModeling: BiLSTM or None
        # Prediction: CTC (nn.Linear) or Attn (custom module)

    def forward(self, input, text, is_train=True):
        if Trans != "None":
            input = self.Transformation(input)
        visual_feature = self.FeatureExtraction(input)                    # [b, 512, h', w']
        visual_feature = self.AdaptiveAvgPool(
            visual_feature.permute(0, 3, 1, 2)
        ).squeeze(3)                                                      # [b, w', 512]
        if Seq == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature
        prediction = self.Prediction(contextual_feature, ...)
        return prediction
```

### Stage contracts (what any replacement module must respect)

- **Feature extractor**: input `[B, C_in, H, W]` (grayscale, `C_in=1` by default) → output `[B, 512, h', w']`. The code avg-pools over `h'` to 1, producing a sequence of length `w'` with 512-dim features per timestep.
- **Sequence modeler**: input `[B, T, C_in]` → output `[B, T, C_out]`. Sequence length must be preserved.
- **Prediction**: input `[B, T, C]` → character logits. Untouched by backbone/seq changes.

---

## 3. Swapping ResNet → ConvNeXt

In `modules/feature_extraction.py`, `ResNet_FeatureExtractor` wraps a **custom** 29-layer ResNet (Table 7 of the paper) — not torchvision's ResNet.

**Steps:**
1. Write `ConvNeXt_FeatureExtractor(nn.Module)` in `modules/feature_extraction.py`.
2. Ensure it ends at `[B, 512, h', w']` — **critically**, keep `w'` (~24–26 columns for 100px-wide input) from over-shrinking. Use asymmetric strides `(2,1)` on later stages (height-only downsampling) like the original ResNet's `Pool3`/`Pool4`, or ConvNeXt's default stem+stages will collapse the width dimension the CTC/Attn decoders rely on.
3. Add `elif opt.FeatureExtraction == 'ConvNeXt':` in `model.py`.
4. Either 1×1-conv-project ConvNeXt's final channels to 512, or change `opt.output_channel` and let it propagate automatically (BiLSTM/TCN input size is computed from `self.FeatureExtraction_output`).

**Important caveat:** because ConvNeXt has a different shape/parameterization than the custom ResNet, you **cannot** load the ResNet weights into the ConvNeXt slot. That stage trains from scratch (or from ImageNet-pretrained ConvNeXt weights) rather than reusing the STR-pretrained checkpoint. Downstream weights (Seq/Pred) are separable by state-dict key prefix, so those can still be reused/frozen if desired.

---

## 4. Swapping BiLSTM → TCN

In `modules/sequence_modeling.py`, `BidirectionalLSTM` is two stacked layers taking `[B, T, C_in]` → `[B, T, C_out]`.

```python
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, kernel_size=3):
        super().__init__()
        layers = []
        in_ch = input_size
        for i in range(num_layers):
            dilation = 2 ** i
            layers += [nn.Conv1d(in_ch, hidden_size, kernel_size,
                                  padding=(kernel_size - 1) * dilation // 2, dilation=dilation),
                       nn.ReLU()]
            in_ch = hidden_size
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):              # x: [B, T, C]
        x = x.transpose(1, 2)          # [B, C, T]
        x = self.net(x).transpose(1, 2)  # [B, T, hidden]
        return self.proj(x)
```

Swap into `model.py`'s `""" Sequence modeling"""` block, matching `BidirectionalLSTM(...)`'s call signature. Since input/output dims stay the same, this is the easiest stage to decouple — you can even keep the pretrained CTC/Attn head and fine-tune only the new TCN.

**Practical note:** TCNs parallelize over time (no recurrence), so they're typically **faster and lighter in VRAM** than BiLSTM at equivalent hidden size — a rare case where a swap is both cheaper *and* potentially competitive in accuracy.

---

## 5. Building combinations in a fork (e.g. a personal fork of the repo)

For a target grid like `{TPS, None} × ConvNeXt × TCN × Prediction`, the fork needs:

1. `modules/feature_extraction.py` → `ConvNeXt_FeatureExtractor` class
2. `modules/sequence_modeling.py` → `TCN` class
3. `model.py` → `elif` branches wiring `opt.FeatureExtraction == 'ConvNeXt'` and `opt.SequenceModeling == 'TCN'`
4. `train.py` argparse → must accept those strings for `--FeatureExtraction` / `--SequenceModeling`

If all four exist, generating checkpoints is pure training-run logistics:

```bash
# TPS-ConvNeXt-TCN-CTC
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation \
  --select_data MJ-ST --batch_ratio 0.5-0.5 \
  --Transformation TPS --FeatureExtraction ConvNeXt --SequenceModeling TCN --Prediction CTC \
  --exp_name TPS-ConvNeXt-TCN-CTC

# None-ConvNeXt-TCN-CTC
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation \
  --select_data MJ-ST --batch_ratio 0.5-0.5 \
  --Transformation None --FeatureExtraction ConvNeXt --SequenceModeling TCN --Prediction CTC \
  --exp_name None-ConvNeXt-TCN-CTC
```

Each run drops `best_accuracy.pth` under `saved_models/<exp_name>/`.

---

## 6. GPU requirements

The paper trained on a single **Tesla P40 (24GB)**, batch size 192, 300K iterations, full MJ+ST (14.4M images).

| Tier | GPU examples | Notes |
|---|---|---|
| Minimum (reduced batch ~64–96) | RTX 3060 12GB, RTX 4060 Ti 16GB, T4 16GB (Colab/Kaggle free) | Works, but slow |
| Comfortable (matches paper's batch 192) | RTX 3090/4090 (24GB), A5000 (24GB), V100 (16/32GB), A100 (Colab Pro) | Recommended |

**Backbone/stage cost adjustments:**
- **ConvNeXt tax**: expect **~20–40% more VRAM** than the ResNet run at equal batch size, and slower throughput per step. If constrained, drop batch size to 96–128 with gradient accumulation.
- **TCN**: negligible extra cost — typically *faster and lighter* than BiLSTM at equal hidden size (no recurrence to serialize).
- **Full 300K-iteration run time**: roughly **1–3 days** on a single 24GB modern GPU, varying by module cost (CTC combos faster, Attn combos slower, per the paper's own ms/image numbers). ConvNeXt pushes toward the slower end.

If GPU budget is tight: cut iterations (e.g. 100–150K) or validate the pipeline on a data subset before committing to a full run.

---

## 7. Suggested ablation grids

| Grid | Combinations | What it isolates |
|---|---|---|
| **Minimal** | `{TPS, None} × ConvNeXt × TCN × CTC` | 2 runs — does TPS still help with the new backbone |
| **Better** | `{TPS, None} × {ConvNeXt, ResNet} × TCN × CTC` | 4 runs — isolates ConvNeXt's actual contribution vs. paper's ResNet baseline |
| **Full swap-in** | `{TPS, None} × {ConvNeXt, ResNet} × {TCN, BiLSTM} × CTC` | 8 runs — independently answers "does TCN replace BiLSTM cleanly" and "does ConvNeXt replace ResNet cleanly," holding Attn out to save cost |

Going beyond this (re-adding Attn, full 24-combo re-derivation) roughly doubles cost again for diminishing insight if the core question is specifically about ConvNeXt/TCN.

### Reference table to fill in (mirrors paper's Table 8 style)

| Combination | Acc (total) | ms/image | params (M) | FLOPs (G) |
|---|---|---|---|---|
| None-ResNet-BiLSTM-CTC (paper baseline) | 80.0 | 4.7 | 44.3 | 10.1 |
| None-ConvNeXt-TCN-CTC | ? | ? | ? | ? |
| TPS-ResNet-BiLSTM-CTC (paper baseline) | 81.9 | 7.8 | 47.0 | 10.3 |
| TPS-ConvNeXt-TCN-CTC | ? | ? | ? | ? |

---

## 8. Latency testing methodology

`test.py --benchmark_all_eval` already reports per-image ms. For a fair, paper-consistent comparison:

- **Same hardware, same batch size** (or batch=1 for true per-image latency), one GPU, nothing else running.
- Report **both GPU and CPU latency** if deployment target matters — ConvNeXt's convolutions may behave very differently on CPU vs. the custom ResNet's dense 3×3 convs, depending on inference framework.
- Track **params + FLOPs** alongside ms/image (as in the paper's Table 8) — ConvNeXt-Tiny (~28M params) vs. the paper's custom ResNet (~44M) may look lighter on paper but not track linearly to FLOPs/latency.
- Run **multiple trials, average** — paper used 5 seeds for accuracy; for latency, ~50–100 warmed-up forward passes smooths out first-call/cuDNN autotune noise.

---

## 9. Replacing TPS with diffusion-based restoration — feasibility

### Why they aren't really the same job

- **TPS**: learned *geometric rectification*. A small localization network predicts ~20 fiducial points; a closed-form spline warps curved/tilted/perspective text into a canonical rectangle. Deterministic, few hundred thousand parameters, sub-few-ms (paper: 3.6ms overhead), trained **end-to-end** with the recognizer — fiducial points are shaped purely by what helps downstream recognition, no independent restoration objective.
- **Diffusion-based restoration**: a *learned image prior* that denoises/deblurs/super-resolves via an iterative reverse process, typically trained on a separate pixel-fidelity/perceptual objective. Fixes blur, noise, low resolution — **not** geometric distortion.

Swapping one for the other conflates two different failure modes: TPS solves *shape*, diffusion restoration solves *signal quality*. A straight swap likely regresses on irregular datasets (IC15/SVTP/CUTE) where TPS's geometric correction is what drives its documented gain (+3.4% accuracy on irregular sets per the paper's Table 2).

### Three integration options

| Option | Structure | Trade-off |
|---|---|---|
| **A — Replace outright** | `input → diffusion_refine(input) → FeatureExtraction` | Literal swap, but loses geometric rectification entirely; likely regresses on curved/rotated text |
| **B — Add as extra stage** | `input → diffusion_refine → TPS → FeatureExtraction` | Preserves TPS's geometric correction, adds restoration on top; cleaner science, but now comparing 5-stage vs. 4-stage models |
| **C — Frozen offline preprocessing** | Run a pretrained restoration model once over the LMDB dataset; train the rest of the pipeline on refined images | Avoids end-to-end backprop cost entirely; most practical starting point |

### The practical cost problem

TPS is cheap because it's a light regression net trained jointly. A diffusion model in the forward pass, if end-to-end and backprop-able, means:
- Backpropagating through however many denoising steps used — even fast distilled 4–8 step models are far more expensive than TPS's single forward pass.
- Per-image latency potentially exploding from ~3.6ms to **tens–hundreds of ms**, dwarfing the ConvNeXt/TCN latency differences.
- Substantially higher training VRAM — likely pushing toward **A100 40GB+** rather than RTX 3090/4090-class, unless frozen (Option C) or gradient-checkpointed.

### Recommendation

- Keep `{TPS, None}` as the Transformation axis for the ConvNeXt/TCN ablation — clean, cheap, comparable to the paper.
- Treat diffusion restoration as a **separate, additive experiment** (Option C) rather than folding it into the same swap-in grid — mixing it in makes it hard to tell whether a gain came from the new backbone/seq stage or just cleaner input pixels.

---

## 10. Visually checking TPS before/after images

### Where to hook in

`TPS_SpatialTransformerNetwork` (in `modules/transformation.py`) takes the raw batch and returns the rectified batch — the line `input = self.Transformation(input)` in `model.py`'s forward pass. Capture both tensors instead of letting one overwrite the other.

### Standalone script (no training loop needed)

```python
import torch
from PIL import Image
import torchvision.transforms as T
from modules.transformation import TPS_SpatialTransformerNetwork

# same opt values you'd train with
F = 20            # num_fiducial
imgH, imgW = 32, 100
input_channel = 1  # grayscale, matches the paper's default

tps = TPS_SpatialTransformerNetwork(
    F=F, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel
)
tps.eval()

# load a saved checkpoint's TPS weights -- NOT a random init
state_dict = torch.load('saved_models/TPS-ResNet-BiLSTM-Attn/best_accuracy.pth', map_location='cpu')
tps_state = {k.replace('Transformation.', ''): v for k, v in state_dict.items() if k.startswith('Transformation.')}
tps.load_state_dict(tps_state)

# preprocess an image the same way dataset.py does
img = Image.open('demo_image/demo_1.png').convert('L')  # grayscale
img = img.resize((imgW, imgH), Image.BICUBIC)
tensor = T.ToTensor()(img)
tensor = tensor.sub_(0.5).div_(0.5)          # normalize to [-1, 1]
batch = tensor.unsqueeze(0)                   # [1, 1, H, W]

with torch.no_grad():
    rectified = tps(batch)                    # [1, 1, H, W]

def to_img(t):
    t = t.squeeze().mul_(0.5).add_(0.5).clamp_(0, 1)
    return t.numpy()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].imshow(to_img(tensor), cmap='gray'); axes[0].set_title('Before TPS'); axes[0].axis('off')
axes[1].imshow(to_img(rectified), cmap='gray'); axes[1].set_title('After TPS'); axes[1].axis('off')
plt.tight_layout()
plt.savefig('tps_before_after.png', dpi=150)
```

**Critical caveat:** the TPS localization network's weights matter enormously. A randomly-initialized TPS produces garbage or near-identity warps — fiducial points are only meaningful once trained jointly with the recognizer's loss. Always load actual checkpoint `Transformation.*` weights.

### Batch grid comparison

```python
def visualize_batch(before, after, n=8):
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(to_img(before[i:i+1]), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(to_img(after[i:i+1]), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Before', fontsize=10)
    axes[1, 0].set_ylabel('After', fontsize=10)
    plt.tight_layout()
    plt.savefig('tps_grid.png', dpi=150)

with torch.no_grad():
    rectified_batch = tps(image_batch)   # image_batch: [N, 1, H, W] from your LMDB loader
visualize_batch(image_batch, rectified_batch, n=8)
```

Pull `image_batch` straight from `dataset.py`'s `AlignCollate`/`hierarchical_dataset` so preprocessing exactly matches training — mismatched preprocessing is the most common reason a manual check looks worse than what the model actually sees.

### What to look for

- **Curved/rotated text (irregular sets — IC15, SVTP, CUTE)**: after TPS should look noticeably straightened, roughly horizontal, evenly spaced. This is where the paper reports the biggest gain (+3.4% on irregular sets, Table 2).
- **Already-regular text (IIIT, SVT, IC03)**: after should look close to identity, maybe minor cropping/centering. Heavy distortion here signals the localization network hasn't converged well.
- **Fiducial point overlay (more diagnostic)**: the localization network's raw `[B, 2F]` coordinate output, before grid generation, can be scatter-plotted on the *original* image to show where it thinks the text boundary is — often more informative than eyeballing the warped result. Grab it via a forward hook on `tps.LocalizationNetwork`, or call it directly before the grid-generation step.

This same script works against any fork's saved checkpoints (e.g. a ConvNeXt+TCN combination's `Transformation.*` weights), so it can double as a check for whether TPS still learns sensible rectification once paired with a different backbone/sequence stage.
