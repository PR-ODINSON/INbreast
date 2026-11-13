# INbreast Mammography Lesion Segmentation

A rigorous, reproducible PyTorch pipeline comparing **U-Net**, **Attention U-Net** (with vendored Oktay et al. attention gates), and **DeepLabV3** on the INbreast mammography dataset using identical ImageNet-pretrained **ResNet34** encoders.

---

### What This Repository Does / Does Not Do

**What it does:**
- Implements strict 5-fold nested patient-grouped cross-validation for lesion segmentation on INbreast.
- Guarantees zero patient leakage across train, validation, and test splits (108 unique patients).
- Compares U-Net, Attention U-Net (Oktay et al. gates in ResNet34), and DeepLabV3 under identical data, loss, optimization, and budget conditions.
- Keeps 100% of model parameters trainable (including BatchNorm affine parameters).
- Tunes binarization thresholds and minimum component sizes exclusively on validation folds.
- Evaluates boundary distances in millimeters using physical DICOM pixel spacing and resize scale tracking.
- Reports patient-level cluster bootstrap 95% confidence intervals (10,000 resamples) and paired permutation tests with Holm-Bonferroni correction.

**What it does not do:**
- Does not freeze any layers (no frozen encoders or frozen batch normalization layers).
- Does not touch or tune on test fold data during training or hyperparameter selection.
- Does not merge coarse elliptical "Spiculated Region" ROIs into the default mass segmentation mask.
- Does not invent or fabricate metadata, labels, or file paths.

---

## 2. Dataset and Target Definition

The pipeline targets the INbreast digital mammography database (Moreira et al., *Academic Radiology*, 2011), consisting of 410 Full-Field Digital Mammography (FFDM) DICOM images and 343 OsiriX XML annotation files across 108 unique patient cases.

### Target Definitions (`data.target` in `config.yaml`)
- `mass` (default): **Mass ROIs only** (116 lesions across 107 positive images).
- `mass_spiculated`: Mass ROIs + Spiculated Region ROIs.
- `calcification`: Calcification and Cluster ROIs.

### Why Spiculated Regions are Excluded by Default
In INbreast annotations, the `"Spiculated Region"` ROI is an idealized 200-point ellipse drawn in OsiriX around spicule extents (ellipse-fit residual = 0.0000; 3.5× larger mean area than mass contours). The true lesion margin is annotated by the `"Mass"` ROI. Merging the ellipse into the target mask artificially distorts and inflates lesion boundaries. Pectoral muscle ROIs are explicitly excluded.

---

## 3. Experimental Design

### Nested Patient-Grouped 5-Fold Cross-Validation
All 108 patients are partitioned into 5 stratified folds based on lesion presence. Patients never overlap across partitions within any fold, and each patient is evaluated in test exactly once across the 5 outer folds:

| Outer Fold | Test Set | Validation Set | Training Set |
|---|---|---|---|
| **Fold 0** | Fold 0 | Fold 1 | Folds 2, 3, 4 |
| **Fold 1** | Fold 1 | Fold 2 | Folds 3, 4, 0 |
| **Fold 2** | Fold 2 | Fold 3 | Folds 4, 0, 1 |
| **Fold 3** | Fold 3 | Fold 4 | Folds 0, 1, 2 |
| **Fold 4** | Fold 4 | Fold 0 | Folds 1, 2, 3 |

### Protocol Rules
- **Validation**: Monitored for early stopping (3-epoch moving average of positive validation Dice; patience = 15, minimum 20 epochs). Validation probability maps are used for post-processing grid searches (threshold and minimum component size).
- **Test**: Evaluated exactly once per fold with frozen thresholds and post-processing filters.
- **Dynamic Negative Resampling**: Training negatives are resampled each epoch via `resample_training_records(records, negative_ratio, seed, epoch)` using `seed + epoch * 100003`. Validation and test sets always evaluate 100% of images.
- **Identical Conditions**: All three models share the exact same splits (`splits.json`), loss (Dice + BCE), AdamW optimizer, warmup schedule, batch size, and epoch budget.

---

## 4. Model Architectures

All models utilize an ImageNet-pretrained **ResNet34** backbone (`torchvision` weights) with all parameters set to `requires_grad=True`. See [MODELS.md](MODELS.md) for detailed wiring diagrams and deviation logs.

| Model | Source | Head Architecture | Parameter Count |
|---|---|---|---|
| **U-Net** | `segmentation_models_pytorch.Unet` | `Conv2d(16, 1, k=3, pad=1)` | 24,436,369 |
| **Attention U-Net** | Oktay et al. gates + ResNet34 U-Net | `Conv2d(16, 1, k=3, pad=1)` | 24,794,485 |
| **DeepLabV3** | `segmentation_models_pytorch.DeepLabV3` | Atrous Spatial Pyramid Pooling (ASPP) | 26,007,105 |

- **Attention U-Net**: Integrates official Oktay et al. 2D grid attention blocks (vendored in `third_party/Attention-Gated-Networks`, commit `eee4881fdc31920efd873773e0b744df8dacbfb6`) into skip connections $x_1, x_2, x_3, x_4$. Coarser decoder stages provide gating signals.
- **DeepLabV3 Limitation**: Operates at output stride 8 via dilated convolutions without multi-scale skip connections, creating a trade-off in fine boundary delineation compared to encoder-decoder U-Net architectures.

---

## 5. Repository Structure

```
Avantika/
├── config.yaml              # Global dataset, augmentation, model & training configuration
├── requirements.txt         # Pinned Python package dependencies
├── splits.json              # 5-fold nested patient-grouped split definitions
├── steps.md                 # Copy-paste operational runbook (setup to report)
├── MODELS.md                # Model provenance, parameter table & upstream diff logs
├── README.md                # Project documentation & code architecture guide
├── INbreast_complete/       # User-provided raw dataset folder
│   ├── AllDICOMs/           # 410 DICOM FFDM mammography files (.dcm)
│   ├── AllXML/              # 343 OsiriX annotation XML files (.xml)
│   ├── INbreast.csv         # Clinical metadata (ACR density, BI-RADS)
│   └── PectoralMuscle/      # Pectoral muscle XML annotations
├── src/
│   ├── dataset.py           # DICOM parsing, XML rasterization, caching & DataLoader
│   ├── models.py            # Model factory, parameter counters & layer verification
│   ├── losses.py            # Dice, BCE, Focal, and Tversky loss functions
│   ├── metrics.py           # Pixel, Hausdorff (mm), ASSD, lesion-level & FROC metrics
│   ├── train.py             # Training loop with moving-avg early stopping & AMP
│   ├── evaluate.py          # Threshold tuning on val, test inference & bootstrap CIs
│   └── visualize.py         # Diagnostic overlays, FROC curves, heatmaps & grids
├── third_party/
│   └── Attention-Gated-Networks/ # Vendored official Oktay et al. grid attention block
├── tests/
│   ├── test_patient_leak.py # 3-way partition & patient leakage assertion test
│   ├── test_metrics.py      # Unit tests for pixel, boundary, and distance metrics
│   ├── test_rasterization.py# Polygon rasterization & spiculated exclusion tests
│   └── test_models.py       # 100% trainable parameter verification & forward/backward tests
└── results/
    └── params.csv           # Verified parameter counts per module group
```

---

## 6. Code Walkthrough

### Data Flow Diagram

```
Raw DICOM + XML Plist
         │
         ▼
[dataset.py: preprocess_sample] ──► Percentile Clip [0.5, 99.5]
         │                       ──► Crop Breast Bounding Box
         │                       ──► Aspect-Preserving Resize & Pad (512x512)
         │                       ──► Track scale factor & mm_per_pixel (0.070 fallback)
         ▼
[data/cache/*.npz] (Optional)
         │
         ▼
[dataset.py: INbreastDataset] ──► Epoch-specific negative resampling (train only)
         │                     ──► Albumentations data augmentation
         ▼
[train.py: train_one_epoch] ──► Mixed precision forward pass (AMP)
         │                   ──► Loss computation (DiceBCELoss)
         │                   ──► Gradient clipping & AdamW step
         ▼
[train.py: validate] ──► 3-epoch moving-average validation positive Dice
         │            ──► Checkpoint model weights (checkpoints/<model>_fold<k>/)
         │            ──► Export validation probabilities (val_probs_best.npz)
         ▼
[evaluate.py: tune_threshold] ──► Grid search threshold + min component size on VAL
         │
         ▼
[evaluate.py: test evaluation] ──► Inference on TEST set with frozen threshold
         │                      ──► Pixel metrics (Dice, IoU, Precision, Recall)
         │                      ──► Physical boundary distances (HD95 mm, ASSD mm, NSD)
         │                      ──► Lesion-level matching (IoU >= 0.2 and 0.5)
         │                      ──► Patient-level cluster bootstrap (10,000 resamples)
         │                      ──► Paired permutation test with Holm correction
         ▼
[visualize.py] ──► Diagnostic overlays, FROC curves, violin plots, attention maps
```

### Module Guide

#### `src/dataset.py`
- **Key Functions/Classes**: `read_dicom_image()`, `read_spacing_mm()`, `parse_xml_rois()`, `resize_and_pad()`, `preprocess_sample()`, `INbreastDataset`, `build_patient_stratified_splits()`, `resample_training_records()`, `precompute_and_cache_dataset()`.
- **Design Decisions**:
  - *Pixel Spacing*: Extracted from DICOM `PixelSpacing` or `ImagerPixelSpacing`. If absent, defaults to `0.070 mm/pixel` with a logged warning.
  - *Metric Conversion*: When images are resized with padding, the physical millimeter spacing scales as `mm_per_pixel = dicom_spacing / resize_scale`.
  - *Aspect-Preserving Resize*: Maintains lesion geometry by scaling the long axis and symmetric zero-padding to 512×512.

#### `src/models.py`
- **Key Functions/Classes**: `build_model()`, `AttentionUNet`, `verify_all_trainable()`, `assert_imagenet_conv1()`, `assert_optimizer_covers_all()`, `get_optimizer_param_groups()`, `export_parameter_counts()`.
- **Design Decisions**:
  - Enforces 100% parameter trainability at startup.
  - Compares encoder conv1 weights against torchvision ResNet34 `IMAGENET1K_V1` (asserting max diff $\le 10^{-5}$).
  - Strictly asserts that parameter groups partition `model.parameters()` without overlap.

#### `src/losses.py`
- **Key Functions/Classes**: `DiceLoss`, `DiceBCELoss`, `FocalLoss`, `TverskyLoss`, `get_loss_function()`.
- **Design Decisions**: Uses numerically stable soft Dice with additive epsilon ($10^{-6}$) combined with binary cross-entropy.

#### `src/metrics.py`
- **Key Functions/Classes**: `compute_pixel_metrics()`, `boundary_distances()`, `compute_lesion_level_metrics()`, `compute_case_metrics()`, `remove_small_components()`, `froc_operating_points()`, `sensitivity_at_fppi()`, `MetricTracker`.
- **Design Decisions**:
  - Boundary metrics use exact Euclidean distance transforms (`scipy.ndimage.distance_transform_edt`) sampled with physical `(row_mm, col_mm)` spacings.
  - Handles empty masks cleanly: when GT and Pred are empty, Dice=1.0 and distance=NaN; when GT is positive and Pred is empty, `penalized_hd95_mm` assigns image diagonal in mm.

#### `src/train.py`
- **Key Functions/Classes**: `train_one_epoch()`, `validate()`, `make_scheduler()`, `main()`.
- **Design Decisions**:
  - Early stopping triggers on a 3-epoch moving average of positive validation Dice to prevent halting on single-epoch noise.
  - Enforces `min_epochs` (default 20) before early stopping can activate.
  - Resamples negative training images each epoch while keeping validation/test intact.

#### `src/evaluate.py`
- **Key Functions/Classes**: `tune_threshold()`, `evaluate_model_fold()`, `patient_bootstrap_ci()`, `pairwise_permutation_test()`, `holm_correction()`, `main()`.
- **Design Decisions**:
  - Performs 2D grid search over thresholds (0.10–0.90) and minimum component sizes (0–256) on validation data only; test inference is run once.
  - Computes 10,000 patient cluster bootstrap resamples to account for multi-image correlation per patient.

#### `src/visualize.py`
- **Key Functions/Classes**: `plot_training_curves()`, `plot_boxplots()`, `plot_froc_curves()`, `plot_qualitative_grid()`, `plot_attention_maps()`, `save_debug_overlays()`.
- **Design Decisions**: Generates publication-ready vector/raster plots using headless Matplotlib (`Agg` backend).

---

## 7. Tests

Run the full test suite with:
```bash
pytest tests/
```

- `tests/test_patient_leak.py`: Validates strict 3-way partition (train $\cap$ val = $\emptyset$, train $\cap$ test = $\emptyset$, val $\cap$ test = $\emptyset$) and confirms all 108 patients appear in test exactly once across folds.
- `tests/test_metrics.py`: Verifies mathematical correctness of Dice, IoU, empty-mask handling, shifted Hausdorff distance, pixel-to-mm scaling, and component filtering.
- `tests/test_rasterization.py`: Checks XML coordinate parsing, mask binary values, aspect-preserving padding, and confirms Spiculated Region exclusion from the default `mass` target.
- `tests/test_models.py`: Verifies all model parameters have `requires_grad=True`, confirms forward and backward gradient flow, and checks attention map bounds $[0, 1]$.

---

## 8. Results

Results will be generated in results/summary.md after running steps.md

---

## 9. Limitations

1. **Dataset Scale**: INbreast contains 107 images with masses across 108 patients, representing a small single-center cohort.
2. **DeepLabV3 Output Stride**: The stride-8 atrous spatial pyramid pooling lacks the multi-scale skip connections of U-Net, limiting its resolution for small or thin lesion boundaries.
3. **Parameter Differences**: DeepLabV3 has ~26.0M parameters, Attention U-Net has ~24.8M parameters, and U-Net has ~24.4M parameters; differences in capacity must be considered when comparing metrics.
4. **Annotation Discrepancy**: 7,142 calcification ROIs across 313 images were identified in the XML files versus 6,880 in 299 images reported in Moreira et al. (2011).
5. **Non-Mass Lesions as Negatives**: Under `target: mass`, images exhibiting only calcifications or other benign findings are treated as negative backgrounds.
6. **Case vs Patient Count**: 108 unique anonymized patients are present versus the 115 cases described in the original publication due to follow-up examinations sharing patient IDs.

---

## 10. Citations

```bibtex
@article{moreira2011inbreast,
  title={INbreast: Toward a Full-field Digital Mammography Database},
  author={Moreira, In{\^e}s C and Amaral, Igor and Mendes, In{\^e}s and Gon{\c{c}}alves, Jo{\~a}o and Ramos, Jorge and Santos, Jo{\~a}o and Cardoso, Jaime S},
  journal={Academic Radiology},
  volume={19},
  number={2},
  pages={236--248},
  year={2012},
  publisher={Elsevier}
}

@inproceedings{oktay2018attention,
  title={Attention U-Net: Learning Where to Look for the Pancreas},
  author={Oktay, Ozan and Schlemper, Jo and Folgoc, Loic Le and Lee, Matthew and Heinrich, Mattias and Misawa, Kazunari and Mori, Kensaku and McDonagh, Steven and Hammerla, Nils Y and Kainz, Bernhard and Glocker, Ben and Rueckert, Daniel},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2018}
}

@article{chen2017rethinking,
  title={Rethinking Atrous Convolution for Semantic Image Segmentation},
  author={Chen, Liang-Chieh and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  journal={arXiv preprint arXiv:1706.05587},
  year={2017}
}

@inproceedings{ronneberger2015u,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages={234--241},
  year={2015},
  publisher={Springer}
}

@misc{yakubovskiy2019smp,
  author={Pavel Iakubovskii},
  title={Segmentation Models PyTorch},
  year={2019},
  publisher={GitHub},
  howpublished={\url{https://github.com/qubvel/segmentation_models.pytorch}}
}
```
