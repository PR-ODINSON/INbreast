# INbreast Lesion Segmentation Pipeline: U-Net, Attention U-Net & DeepLabV3

A complete, reproducible PyTorch training and evaluation pipeline for breast lesion segmentation on the **INbreast** mammography database (Moreira et al., Academic Radiology 2011), comparing **U-Net**, **Attention U-Net** (with original Oktay et al. attention gates), and **DeepLabV3** using identical ImageNet-pretrained **ResNet34** encoders.

---

## 1. Project Directory Structure

```
Avantika/
├── config.yaml              # Master configuration (paths, hyperparameters, splits)
├── requirements.txt         # Pinned Python package dependencies
├── MODELS.md                # Architecture provenance, wiring diagrams & minimal changes
├── README.md                # End-to-end instructions
├── INbreast_complete/       # Original dataset (read-only)
│   ├── AllDICOMs/           # 410 Full-Field Digital Mammograms (.dcm)
│   ├── AllXML/              # 343 OsiriX XML annotations (.xml)
│   ├── INbreast.csv         # Clinical metadata (ACR, BI-RADS)
│   └── PectoralMuscle/      # Pectoral muscle boundary annotations
├── src/
│   ├── dataset.py           # DICOM parsing, XML rasterization, cropping, Dataset class
│   ├── models.py            # U-Net, Attention U-Net, DeepLabV3 (100% unfrozen assertion)
│   ├── losses.py            # Combined Dice + BCE, Focal, Tversky losses
│   ├── metrics.py           # Pixel Dice/IoU + Lesion-level recall & FP/image
│   ├── train.py             # CLI training runner (AMP, early stopping, AdamW)
│   ├── evaluate.py          # 5-fold evaluation, validation threshold tuning, paired t-tests
│   └── visualize.py         # 10 debug overlays, qualitative figures, attention heatmaps
├── third_party/
│   └── Attention-Gated-Networks/ # Vendored official Oktay et al. Grid Attention Block
└── tests/
    ├── test_patient_leak.py # Verifies zero patient leakage across all 5 folds
    ├── test_rasterization.py# Tests polygon coordinate to binary mask rasterization
    └── test_models.py       # Asserts 100% trainable params & forward/backward flow
```

---

## 2. Server Environment Setup

1. **Clone or transfer the project folder to your GPU server:**
   ```bash
   cd Avantika
   ```

2. **Create and activate a clean Python environment (Python 3.10 or 3.11 recommended):**
   ```bash
   conda create -n inbreast python=3.11 -y
   conda activate inbreast
   ```

3. **Install PyTorch with CUDA support:**
   ```bash
   # For CUDA 11.8:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   # Or for CUDA 12.1+:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install all remaining requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Step-by-Step Execution Guide

### Step 1: Run Unit Tests
Run the automated test suite to verify zero patient leakage, mask rasterization, and 100% unfrozen model architectures:
```bash
pytest tests/ -v
```
All unit tests should pass with 100% success.

### Step 2: Generate 10 Debug Contour Overlays
Inspect 10 sample image + ground-truth contour overlays in `debug_overlays/` to verify that contours are perfectly aligned with lesion tissue:
```bash
python src/visualize.py --overlays
```

### Step 3: (Recommended) Precompute Dataset Cache
Precomputing the cropped, padded $512 \times 512$ tensors to `data/cache/` provides a **20–50x speedup** during training:
```bash
python -c "import yaml; from src.dataset import precompute_and_cache_dataset; cfg = yaml.safe_load(open('config.yaml')); precompute_and_cache_dataset(cfg)"
```

### Step 4: Run Smoke Test
Run a quick 2-epoch test run on Fold 0 to verify end-to-end forward pass, backward pass, AMP scaling, and checkpointing:
```bash
python src/train.py --model unet --fold 0 --epochs 2 --batch_size 4
python src/train.py --model attunet --fold 0 --epochs 2 --batch_size 4
python src/train.py --model deeplabv3 --fold 0 --epochs 2 --batch_size 4
```

### Step 5: Full 5-Fold Training Across All Models
Train all three models across all 5 cross-validation folds (100 epochs with early stopping patience 15):

```bash
# U-Net (Folds 0 to 4)
for fold in 0 1 2 3 4; do
    python src/train.py --model unet --fold $fold
done

# Attention U-Net (Folds 0 to 4)
for fold in 0 1 2 3 4; do
    python src/train.py --model attunet --fold $fold
done

# DeepLabV3 (Folds 0 to 4)
for fold in 0 1 2 3 4; do
    python src/train.py --model deeplabv3 --fold $fold
done
```

### Step 6: 5-Fold Cross Validation Evaluation
Evaluate all models across all 5 folds with validation-only threshold tuning, generate the comparison summary table, and run paired statistical significance tests:
```bash
python src/evaluate.py
```
This produces:
- `results/per_fold_results.csv`: Complete metrics per fold and model.
- `results/summary.csv`: Aggregated **Mean $\pm$ Std** across all 5 folds.
- Printed Markdown summary table comparing U-Net, Attention U-Net, and DeepLabV3.
- Paired $t$-test and Wilcoxon signed-rank test outputs.

### Step 7: Generate Qualitative Comparisons & Attention Maps
1. **Qualitative Figure (Image, Ground Truth, Predictions across models):**
   ```bash
   python src/visualize.py --qualitative --fold 0
   ```
2. **Attention Map Visualizations (Attention Gate heatmaps $\alpha$):**
   ```bash
   python src/visualize.py --attention --fold 0
   ```
Visualizations are saved to `visualizations/`.

---

## 4. Hard Rules Enforcement Summary

1. **100% Unfrozen Parameters:**
   - Every model has zero frozen layers.
   - Verified via `verify_all_trainable()` startup assertion:
     $$\text{total\_params} == \text{trainable\_params}$$
   - AdamW optimizer trains both encoder (`lr=1e-4`) and decoder/head (`lr=1e-3`) parameter groups.
2. **Original Model Code Provenance:**
   - U-Net and DeepLabV3 imported from `segmentation_models_pytorch`.
   - Attention U-Net built with the official Oktay et al. grid attention block (`third_party/Attention-Gated-Networks`).
   - Detailed wiring and minimal adaptations documented in `MODELS.md`.
3. **Identical Experimental Conditions:**
   - All models use the exact same `splits.json` (zero patient leakage).
   - Identical preprocessing, aspect-preserving padding/resizing ($512 \times 512$), ImageNet normalization, Dice+BCE loss, AdamW optimizer, and evaluation protocols.

