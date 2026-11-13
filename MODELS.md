# MODELS.md — INbreast Lesion Segmentation

## Overview

Three models are compared under identical conditions (same splits, loss, optimizer, epochs, augmentation).
All parameters — encoder, decoder, attention gates, BatchNorm affine weights — have `requires_grad=True`.
An assertion at startup verifies total params == trainable params.

| Model | Backbone | Encoder weights | Segmentation head | Source |
|-------|----------|----------------|-------------------|--------|
| U-Net | ResNet34 | ImageNet (IMAGENET1K_V1) | Conv2d(16,1,k=3,pad=1) | `smp.Unet` |
| Attention U-Net | ResNet34 | ImageNet (IMAGENET1K_V1) | Conv2d(16,1,k=3,pad=1) | smp encoder + custom decoder + Oktay gates |
| DeepLabV3 | ResNet34 | ImageNet (IMAGENET1K_V1) | ASPP head, 1 class | `smp.DeepLabV3` |

---

## Parameter Counts (from instantiated models)

Counts come from `sum(p.numel() for p in model.parameters())` on instantiated models.

| model | total | trainable | encoder | decoder_head |
|-------|------:|----------:|--------:|-------------:|
| unet | 24,436,369 | 24,436,369 | 21,284,672 | 3,151,697 |
| attunet | 24,794,485 | 24,794,485 | 21,284,672 | 3,509,813 |
| deeplabv3 | 26,007,105 | 26,007,105 | 21,284,672 | 4,722,433 |

The 358,116-parameter difference between U-Net and Attention U-Net is the four grid attention gate modules (theta/phi/psi/W convolutions + BatchNorm).

---

## Pretrained Weight Verification

At startup, `assert_imagenet_conv1(model)` loads a fresh `torchvision.models.resnet34(weights='IMAGENET1K_V1')` and compares `model.encoder.conv1.weight` to the reference.
The assertion fails if the max absolute difference exceeds 1e-5.

---

## U-Net (`smp.Unet`)

```python
smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
```

Standard ResNet34 U-Net from [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).  
Segmentation head: `Conv2d(16, 1, kernel_size=3, stride=1, padding=1)` — confirmed by `print(model.segmentation_head)`.

---

## DeepLabV3 (`smp.DeepLabV3`)

```python
smp.DeepLabV3(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
```

Uses atrous convolutions (ASPP) on the ResNet34 backbone.  
**Limitation:** DeepLabV3's atrous strategy is designed for stride-16/8 backbones. ResNet34 (originally stride-32) is adapted by smp via output-stride changes; this affects receptive-field trade-offs but keeps backbone family consistent with U-Net for fair comparison.

---

## Attention U-Net

**Description: "Oktay attention gates in a ResNet34 U-Net"**

### Architecture

The encoder is the smp ResNet34 (identical to U-Net above).  
The decoder adds grid attention gates (Oktay et al., MIDL 2018) to each skip connection before concatenation.

```
Encoder outputs (smp):   _x0, x1, x2, x3, x4, x5
  _x0 : stem (not used as skip)
  x1  : after layer1  — 64 ch
  x2  : after layer2  — 128 ch
  x3  : after layer3  — 256 ch
  x4  : after layer4  — 512 ch
  x5  : bottleneck    — 512 ch  (gating signal for deepest gate)

Gate wiring:
  gate4(skip=x4, g=x5)  → gated_x4;  dec4(x5, gated_x4)  → d4  [256 ch]
  gate3(skip=x3, g=d4)  → gated_x3;  dec3(d4, gated_x3)  → d3  [128 ch]
  gate2(skip=x2, g=d3)  → gated_x2;  dec2(d3, gated_x2)  → d2  [64 ch]
  gate1(skip=x1, g=d2)  → gated_x1;  dec1(d2, gated_x1)  → d1  [32 ch]
  dec0(d1) → 16-ch feature map
  segmentation_head(16) → logits (1 ch)
```

Each gate: `inter_channels = in_channels // 2` (matches upstream default).  
Each `DecoderBlock`: bilinear upsample → concatenate → Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU.  
Segmentation head: `Conv2d(16, 1, kernel_size=3, padding=1)` — identical to `smp.Unet`.

---

## Attention Gate Implementation — Deviation Log

**Upstream reference:** [`ozan-oktay/Attention-Gated-Networks`](https://github.com/ozan-oktay/Attention-Gated-Networks)  
**Pinned commit:** `eee4881fdc31920efd873773e0b744df8dacbfb6` (2020-11-13)  
**Our vendored file:** `third_party/Attention-Gated-Networks/models/layers/grid_attention_layer.py`  

The table below documents every difference vs the upstream file at commit eee4881.

| # | Location | Upstream | Ours | Reason |
|---|----------|----------|------|--------|
| 1 | Class structure | `_GridAttentionBlockND` base class supporting 2D and 3D via `dimension` and `mode` parameters (`concatenation`, `concatenation_debug`, `concatenation_residual`) | Only `GridAttentionBlock2D` (2D only, additive attention) | We only need 2D; N-D generality adds complexity with no benefit |
| 2 | TORR variants | `_GridAttentionBlockND_TORR` with softmax/mean/range normalization; `GridAttentionBlock2D_TORR`, `GridAttentionBlock3D_TORR` | Dropped | Only the additive attention formulation is used |
| 3 | Weight init | `from models.networks_other import init_weights; init_weights(m, init_type='kaiming')` → `kaiming_normal_` with `mode='fan_in'`; BatchNorm: `normal_(weight, 1.0, 0.02)` and `constant_(bias, 0.0)` | **Fixed to match upstream:** `kaiming_normal_(weight, a=0, mode='fan_in')`; BatchNorm: `normal_(weight, 1.0, 0.02)`, `constant_(bias, 0.0)` | Restores upstream init exactly |
| 4 | `F.upsample()` | `F.upsample(input, size=..., mode='bilinear')` (deprecated) | `F.interpolate(input, size=..., mode='bilinear', align_corners=True)` | Removes deprecation warning |
| 5 | `F.sigmoid()` | `F.sigmoid(psi_f)` (deprecated) | `torch.sigmoid(psi_f)` | Removes deprecation warning |
| 6 | `return_attention` | Not present — always returns `(W_y, sigm_psi_f)` | Added `return_attention: bool = False`; when `True` returns `(output, alpha)` | Required for attention map visualization |
| 7 | `inter_channels` default | `in_channels // 2` | `in_channels // 2` ✓ | **Matches upstream exactly** |
| 8 | Attention coefficient multiply | `sigm_psi_f.expand_as(x) * x` | `alpha * x` (broadcast-compatible) | Equivalent; cleaner code |
| 9 | 3D classes | `GridAttentionBlock3D` | Dropped | 2D dataset only |

### Gate placement vs original Oktay network

The original `unet_grid_attention_3D.py` gates encoder skip connections using the **previous decoder output** as the gating signal.
The first gating signal is the **bottleneck** (encoder layer 4 output in 3D U-Net).
Our implementation mirrors this: `gate4` uses `g=x5` (bottleneck), subsequent gates use the preceding decoder output.
This matches the original paper's Figure 1 exactly.

---

## Optimizer Coverage Assertion

`assert_optimizer_covers_all(model, param_groups)` verifies:
- Every `model.parameters()` id appears in exactly one param group.
- No extra parameters sneak in.
- No parameter is duplicated.

Encoder group uses `lr=1e-4`; decoder+head group uses `lr=1e-3` (10× higher).  
BatchNorm affine parameters (`weight`, `bias`) are in whichever group they belong to by prefix — never frozen.

---

## Training Conditions (identical for all three models)

| Setting | Value |
|---------|-------|
| Loss | Dice + BCE (equal weight) |
| Optimizer | AdamW |
| Encoder LR | 1e-4 |
| Decoder LR | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR with 5-epoch linear warmup |
| Epochs | 100 (max) |
| Min epochs | 20 |
| Early stop | 15 epochs patience, 3-epoch moving-avg val Dice |
| Batch size | 8 |
| Mixed precision | Yes (AMP) |
| Augmentation | HFlip, ShiftScaleRotate, BrightnessContrast, Gamma, ElasticTransform |
| Neg resampling | Per-epoch, seed = data.seed + epoch × 100003 |
| Val/test | All images kept (no negative subsampling) |
| Splits | Nested 5-fold, patient-grouped, stratified by lesion presence |
| Target | `mass` = Mass ROIs only |
| Pixel spacing | Read from DICOM PixelSpacing/ImagerPixelSpacing; fallback 0.070 mm |
