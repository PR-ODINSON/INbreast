# INbreast Lesion Segmentation: Execution Runbook (steps.md)

This runbook documents the exact, step-by-step procedure to set up, verify, train, evaluate, and visualize models in this repository.

---

## 1. Prerequisites and Environment Setup

- **Python**: 3.10+ (tested up to 3.14).
- **Hardware / CUDA**: An NVIDIA GPU with ≥ 8 GB VRAM is recommended for batch size 8 at 512×512 resolution (batch size can be reduced in `config.yaml` or via CLI for smaller GPUs). CPU execution is supported via `--device cpu`.
- **Virtual Environment**:

```bash
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Verification**:
```bash
python -c "import torch, smp, pydicom, albumentations, cv2; print('PyTorch CUDA:', torch.cuda.is_available())"
```
*Expected output*: Prints `PyTorch CUDA: True` (if GPU available) or `False` without import errors.

---

## 2. Dataset Preparation

The INbreast dataset must be acquired by the user (Moreira et al., 2011). Place the dataset files matching the paths configured in `config.yaml`:

```
INbreast_complete/
├── AllDICOMs/                          # Raw .dcm files (e.g. 20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm)
├── AllXML/                             # Annotation XML plists (e.g. 20586908.xml)
├── INbreast.csv                        # Metadata spreadsheet (semicolon-delimited)
└── PectoralMuscle/
    └── Pectoral Muscle XML/            # Pectoral muscle boundary XML plists
```

*Note*: 410 DICOM images and 343 annotation XMLs are expected. Patient IDs are parsed from the second filename token (e.g., `6c613a14b80a8591`).

---

## 3. Configuration (`config.yaml`)

Edit `config.yaml` to adjust execution parameters:

| Section | Key | Default | Description |
|---|---|---|---|
| `data` | `target` | `"mass"` | Target ROI: `"mass"` (Mass only), `"mass_spiculated"` (Mass + Spiculated Region), or `"calcification"`. |
| `data` | `image_size` | `512` | Square canvas dimension (aspect-preserving resize with zero padding). |
| `data` | `negative_ratio` | `1.0` | Ratio of negative to positive images resampled per epoch in training (-1 keeps all). Val/test keep 100%. |
| `data` | `fallback_spacing_mm` | `0.070` | Default pixel spacing when DICOM tags are missing. |
| `data` | `use_cache` | `true` | Load preprocessed `.npz` arrays from `data/cache`. |
| `training` | `batch_size` | `8` | Training and validation batch size. |
| `training` | `epochs` | `100` | Maximum training epochs. |
| `training` | `min_epochs` | `20` | Minimum epochs before early stopping can trigger. |
| `training` | `scheduler` | `"cosine_warmup"` | LR scheduler: `"cosine_warmup"` (5-epoch linear warmup + cosine) or `"plateau"`. |
| `training` | `device` | `"cuda"` | Compute device (`"cuda"` or `"cpu"`). |

---

## 4. Running the Tests

Verify patient partition integrity, rasterization logic, and metric mathematical correctness:

```bash
pytest tests/
```

Or run test files individually:
```bash
python tests/test_patient_leak.py
python tests/test_metrics.py
python tests/test_rasterization.py
```

- **Output / Verification**: All assertions pass. Output confirms:
  - 3-way partition verified: 108 unique patients, each appearing in test exactly once across 5 folds.
  - Spiculated Region excluded from default `mass` target.
  - Metric edge cases (empty masks, distance scaling) validated.

---

## 5. Generating Debug Overlays

Inspect preprocessed images and contour alignments visually:

```bash
python src/visualize.py --overlays --config config.yaml
```

- **Creates**: `debug_overlays/01_<id>.png` through `debug_overlays/10_<id>.png` (three panels: raw preprocessed mammogram, inferno target mask, and red lesion overlay).
- **Duration**: ~5–10 seconds.
- **Verification**: Check images in `debug_overlays/` to confirm lesion contours align with tissue masses.

---

## 6. Precomputing the Dataset Cache (Optional but Recommended)

Preprocess DICOMs and rasterize masks to compressed `.npz` files for fast training I/O:

```bash
python src/train.py --model unet --fold 0 --precache --epochs 0
```
*(Or pass `--precache` on your first training run).*

- **Creates**: `data/cache/<image_id>.npz` for all 410 images containing cropped images, masks, scale factors, and spacing metadata.
- **Duration**: 2–4 minutes (run once).

---

## 7. Smoke Test

Run a 2-epoch sanity check with batch size 2 on a tiny data subset:

```bash
python src/train.py --model unet --fold 0 --smoke_test
```

- **Checks**:
  - Unfrozen weight assertion passes (`verify_all_trainable`).
  - ImageNet backbone verification passes (`assert_imagenet_conv1`).
  - Disjoint optimizer parameter group coverage passes (`assert_optimizer_covers_all`).
  - Checkpoint and logs write to `checkpoints/unet_fold0/best_model.pth` and `training_log.csv`.
- **Duration**: ~10–30 seconds.

---

## 8. Full Training

### Train a single model and fold
```bash
python src/train.py --model unet --fold 0 --config config.yaml
```
Valid choices for `--model`: `unet`, `attunet`, `deeplabv3`.  
Valid choices for `--fold`: `0`, `1`, `2`, `3`, `4`.

### Train all models across all 5 folds
Run in Bash / Linux:
```bash
for model in unet attunet deeplabv3; do
  for fold in 0 1 2 3 4; do
    echo "=== Training $model fold $fold ==="
    python src/train.py --model $model --fold $fold --config config.yaml
  done
done
```

Run in Windows PowerShell:
```powershell
foreach ($model in @("unet", "attunet", "deeplabv3")) {
  foreach ($fold in 0..4) {
    Write-Host "=== Training $model fold $fold ==="
    python src/train.py --model $model --fold $fold --config config.yaml
  }
}
```

### Background Execution (Linux)
Use `nohup` or `tmux` to ensure long runs do not terminate if disconnected:
```bash
nohup bash -c 'for m in unet attunet deeplabv3; do for f in 0 1 2 3 4; do python src/train.py --model $m --fold $f; done; done' > train_all.log 2>&1 &
```

- **Output files created per fold**:
  - `checkpoints/<model>_fold<k>/best_model.pth`: Model weights at peak 3-epoch moving-average validation Dice.
  - `checkpoints/<model>_fold<k>/last_model.pth`: Checkpoint from the final epoch.
  - `checkpoints/<model>_fold<k>/val_probs_best.npz`: Validation prediction probabilities and ground truth used for threshold tuning.
  - `checkpoints/<model>_fold<k>/training_log.csv`: Epoch-by-epoch loss, Dice, IoU, and learning rate history.
  - `checkpoints/<model>_fold<k>/training_curves.png`: Loss, Dice, and IoU plots.
- **Duration**: Varies by GPU (typically 10–25 minutes per model fold on modern GPUs).

---

## 9. Evaluation and Reporting

After training completes, run test-set evaluation across all folds:

```bash
python src/evaluate.py --config config.yaml --models unet attunet deeplabv3 --folds 0 1 2 3 4
```

- **What it does**:
  1. For each model and fold, loads `val_probs_best.npz` and tunes the binarization threshold (0.10–0.90) and minimum component size filter (0, 16, 32, 64, 128, 256) on validation data only.
  2. Freezes optimal parameters and executes inference on the test fold.
  3. Computes pixel, distance (in mm), lesion-level, and FROC metrics.
  4. Runs 10,000 patient-level cluster bootstrap iterations for 95% confidence intervals.
  5. Computes paired permutation tests with Holm-Bonferroni correction between model pairs.
- **Output files created**:
  - `results/per_image_<model>_fold<k>.csv`: Per-test-image metrics, metadata, and runtime.
  - `results/prob_maps_<model>_fold<k>.npz`: Raw probability predictions on the test set.
  - `results/froc_<model>_fold<k>.csv`: Operating points across threshold sweeps.
  - `results/froc_aggregate.csv`: Mean and standard deviation of FROC points across folds.
  - `results/summary.csv`: Aggregated bootstrap means and 95% CIs per model and metric.
  - `results/stats_pairwise.csv`: Pairwise differences, permutation p-values, and Holm significance.
  - `results/REPORT.md`: Formatted Markdown evaluation report.
  - `results/params.csv`: Verified parameter breakdown per module group.

### Generate All Visualizations
```bash
python src/visualize.py --all --config config.yaml --fold 0
```
- **Creates**:
  - `visualizations/training_curves/<model>_training_curves.png`: Overlaid 5-fold training trajectories.
  - `visualizations/violin_dice.png`, `violin_iou.png`, `violin_hd95_mm.png`: Metric distributions on positive images.
  - `visualizations/froc_curves.png`: Mean FROC curves with ±1 std error bands.
  - `visualizations/qualitative/`: Best 3, median 3, and worst 3 segmentations across all models.
  - `visualizations/attention_maps/`: Attention coefficient heatmaps from Attention U-Net gating stages.

---

## 10. Interpreting Metrics and Conventions

All metrics adhere strictly to the definitions in `src/metrics.py`:

| Metric | Definition & Code Convention |
|---|---|
| **Dice / IoU** | Intersection over union and harmonic mean of precision/recall. |
| **Precision / Recall / Specificity** | Pixel-level classification rates. |
| **HD95 / HD100 (mm)** | 95th and 100th percentile symmetric directed Hausdorff distance in mm using DICOM spacing divided by resize scale. |
| **ASSD (mm)** | Average Symmetric Surface Distance in millimeters. |
| **NSD (1mm / 2mm)** | Normalized Surface Dice within 1.0 mm and 2.0 mm boundary tolerances. |
| **Lesion Recall / F1 (0.2 / 0.5)** | Connected-component lesion matching at IoU thresholds of 0.2 and 0.5. |
| **FPPI** | Average false positive lesion count per image. |
| **FROC Sensitivity** | Lesion detection sensitivity at 1.0 and 2.0 average FPPI. |
| **Image-level AUC** | Area under the ROC curve using per-image maximum predicted probability. |

### Empty-Mask Conventions
- **Both GT and Pred empty**: `Dice=1.0`, `IoU=1.0`, `Precision=1.0`, `Recall=1.0`, `HD95=NaN`.
- **GT empty, Pred non-empty**: `Dice=0.0`, `IoU=0.0`, `Precision=0.0`, `Recall=1.0`, `HD95=NaN`.
- **GT non-empty, Pred empty**: `Dice=0.0`, `IoU=0.0`, `Precision=0.0`, `Recall=0.0`, `HD95=NaN`, `penalized_hd95_mm = image_diagonal_mm`.

---

## 11. Troubleshooting

1. **CUDA Out of Memory**:
   - Lower `training.batch_size` in `config.yaml` to `4` or `2`, or run:
     `python src/train.py --model unet --fold 0 --batch_size 4`
2. **Missing DICOM or XML paths**:
   - Check directory layout in `config.yaml` (`data.dicom_dir` and `data.xml_dir`). Paths are relative to project root.
3. **Pydicom PixelSpacing missing / warning**:
   - Normal for some INbreast DICOMs. The pipeline logs a warning and automatically applies the standard fallback `0.070 mm/pixel`.
4. **Offline / Sandbox HuggingFace download errors**:
   - `segmentation-models-pytorch` downloads ResNet34 weights on first instantiation. Run on a machine with internet access or pre-populate torch/huggingface cache.
5. **Cache stale after changing target**:
   - If switching `data.target` (e.g. from `mass` to `mass_spiculated`), delete `data/cache` or run with `use_cache: false`. The cache checks target validity and will overwrite stale files.
6. **Resuming after an interrupted run**:
   - Training writes `checkpoints/<model>_fold<k>/last_model.pth`. Rerun the command for that specific fold.
7. **Invalid statement separator error on Windows PowerShell**:
   - Do not chain commands with `&&` in Windows PowerShell. Use `;` or separate lines.

---

## 12. Exact Reproduction Checklist

- [ ] **Dependencies**: `pip install -r requirements.txt`
- [ ] **Seed**: `data.seed: 42` in `config.yaml` (sets Python, NumPy, PyTorch CPU/CUDA deterministic flags).
- [ ] **Splits**: Verified `splits.json` with 108 patients in nested 5-fold structure.
- [ ] **Vendored Attention Gate**: Commit `eee4881fdc31920efd873773e0b744df8dacbfb6` in `third_party/Attention-Gated-Networks`.
- [ ] **Model Backbones**: Pretrained torchvision ResNet34 (`IMAGENET1K_V1`), verified by conv1 assertion.

