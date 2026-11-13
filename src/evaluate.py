"""Evaluation script: threshold tuning on VAL, final metrics on TEST only.

Usage:
  python src/evaluate.py
  python src/evaluate.py --models unet attunet deeplabv3 --smoke_test
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import INbreastDataset
from src.metrics import (
    PER_IMAGE_COLUMNS,
    MetricTracker,
    compute_case_metrics,
    compute_lesion_level_metrics,
    compute_pixel_metrics,
    froc_operating_points,
    image_level_auc,
    remove_small_components,
    sensitivity_at_fppi,
)
from src.models import build_model, export_parameter_counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_name: str, fold: int, config: dict, device: torch.device):
    ckpt_path = os.path.join("checkpoints", f"{model_name}_fold{fold}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [skip] {ckpt_path} not found")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = config["models"].get("backbone", "resnet34")
    classes  = int(config["models"].get("classes", 1))
    model = build_model(model_name, backbone=backbone,
                        encoder_weights=None,  # weights already in checkpoint
                        in_channels=3, classes=classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model


@torch.no_grad()
def _infer(model, loader, device):
    """Return (probs, gt_masks, image_ids, mm_rows, mm_cols, inference_ms_list)."""
    all_probs, all_gt, all_ids, all_mmr, all_mmc, all_ms = [], [], [], [], [], []
    for batch in tqdm(loader, desc="  Inference", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_img_ms = elapsed_ms / images.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        gt    = batch["mask"].numpy().astype(np.uint8)

        for i in range(images.size(0)):
            all_probs.append(probs[i, 0])
            all_gt.append(gt[i, 0])
            all_ids.append(batch["image_id"][i])
            all_mmr.append(float(batch["mm_per_px_row"][i]))
            all_mmc.append(float(batch["mm_per_px_col"][i]))
            all_ms.append(per_img_ms)

    return (
        np.stack(all_probs, 0).astype(np.float32),
        np.stack(all_gt, 0).astype(np.uint8),
        all_ids,
        all_mmr,
        all_mmc,
        all_ms,
    )


# ---------------------------------------------------------------------------
# Threshold + min_component_size tuning (VAL only)
# ---------------------------------------------------------------------------

def tune_threshold(val_probs, val_gt, config: dict, smoke_test: bool = False):
    """Grid search on VAL prob maps. Returns (best_thresh, best_min_size, best_dice)."""
    eval_cfg = config.get("evaluation", {})
    thresholds = np.arange(
        float(eval_cfg.get("threshold_search_start", 0.10)),
        float(eval_cfg.get("threshold_search_end",   0.91)),
        float(eval_cfg.get("threshold_search_step",  0.05)),
    )
    min_sizes = eval_cfg.get("min_component_sizes", [0, 16, 32, 64, 128, 256])
    if smoke_test:
        thresholds = np.array([0.4, 0.5, 0.6])
        min_sizes  = [0, 32]

    best_dice, best_thresh, best_min = -1.0, 0.5, 0
    for th in thresholds:
        for ms in min_sizes:
            dices = []
            for prob, gt in zip(val_probs, val_gt):
                pred = (prob > th).astype(np.uint8)
                if ms > 0:
                    pred = remove_small_components(pred, ms)
                gt_sum = int(gt.sum())
                if gt_sum == 0:
                    continue  # only optimise on positive images
                px = compute_pixel_metrics(pred, gt)
                dices.append(px["dice"])
            if not dices:
                continue
            mean_dice = float(np.mean(dices))
            # Tie-break: higher threshold wins (more conservative)
            if mean_dice > best_dice or (
                abs(mean_dice - best_dice) < 1e-6 and float(th) > best_thresh
            ):
                best_dice, best_thresh, best_min = mean_dice, float(th), int(ms)

    return best_thresh, best_min, best_dice


# ---------------------------------------------------------------------------
# FROC sweep
# ---------------------------------------------------------------------------

def compute_froc_sweep(prob_maps, gt_masks, patient_ids, config: dict, fold: int, model_name: str):
    eval_cfg = config.get("evaluation", {})
    thresholds = np.arange(
        float(eval_cfg.get("froc_threshold_start", 0.02)),
        float(eval_cfg.get("froc_threshold_end",   0.99)),
        float(eval_cfg.get("froc_threshold_step",  0.02)),
    )
    n_images = len(prob_maps)
    # unique positive images (GT has at least 1 pixel)
    pos_indices = [i for i, gt in enumerate(gt_masks) if gt.sum() > 0]
    n_pos = len(pos_indices)

    rows = []
    for th in thresholds:
        tp_total = fp_total = gt_total = fp_pos = 0
        for i, (prob, gt) in enumerate(zip(prob_maps, gt_masks)):
            pred = (prob > th).astype(np.uint8)
            les = compute_lesion_level_metrics(pred, gt, iou_threshold=0.2)
            tp_total += les["tp_lesions"]
            fp_total += les["fp_lesions"]
            gt_total += les["gt_lesions"]
            if i in pos_indices:
                fp_pos += les["fp_lesions"]
        fppi    = fp_total / n_images if n_images else 0.0
        sens    = tp_total / gt_total if gt_total else 0.0
        fppi_pos = fp_pos / n_pos if n_pos else 0.0
        rows.append({
            "model": model_name, "fold": fold, "threshold": float(th),
            "tp": tp_total, "fp": fp_total, "gt": gt_total,
            "fppi": fppi, "sensitivity": sens,
            "fppi_pos": fppi_pos,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap CI (patient-level cluster)
# ---------------------------------------------------------------------------

def patient_bootstrap_ci(df: pd.DataFrame, metric: str, n_resamples: int = 10000, seed: int = 42):
    """Resample patients with replacement; take all images per patient."""
    if metric not in df.columns:
        return float("nan"), float("nan"), float("nan")
    patients = df["patient_id"].unique()
    rng = np.random.RandomState(seed)
    boot_vals = []
    for _ in range(n_resamples):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        rows = pd.concat([df[df["patient_id"] == p] for p in chosen], ignore_index=True)
        v = rows[metric].dropna()
        if len(v) > 0:
            boot_vals.append(float(v.mean()))
    if not boot_vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(boot_vals)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def pairwise_permutation_test(df_a: pd.DataFrame, df_b: pd.DataFrame,
                               metric: str, n_resamples: int = 10000, seed: int = 42):
    """Patient-level paired permutation test. Returns p-value."""
    # Match on patient_id
    pids = set(df_a["patient_id"].unique()) & set(df_b["patient_id"].unique())
    if not pids:
        return float("nan")

    def _mean_per_patient(df, pids_):
        return np.array([df[df["patient_id"] == p][metric].dropna().mean() for p in sorted(pids_)])

    scores_a = _mean_per_patient(df_a, pids)
    scores_b = _mean_per_patient(df_b, pids)
    obs_diff = float(np.mean(scores_a) - np.mean(scores_b))
    rng = np.random.RandomState(seed)
    perm_diffs = []
    for _ in range(n_resamples):
        swap = rng.rand(len(pids)) > 0.5
        a_ = np.where(swap, scores_b, scores_a)
        b_ = np.where(swap, scores_a, scores_b)
        perm_diffs.append(float(np.mean(a_) - np.mean(b_)))
    p_val = float(np.mean(np.abs(perm_diffs) >= abs(obs_diff)))
    return p_val, obs_diff


def holm_correction(p_values: list):
    """Holm-Bonferroni correction. Returns corrected p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected[orig_idx] = min(1.0, p * (n - rank))
    # Enforce monotonicity
    running_max = 0.0
    for _, (orig_idx, _) in enumerate(indexed):
        corrected[orig_idx] = max(corrected[orig_idx], running_max)
        running_max = corrected[orig_idx]
    return corrected


# ---------------------------------------------------------------------------
# FLOPs + inference time
# ---------------------------------------------------------------------------

def compute_flops(model, device):
    try:
        from thop import profile as thop_profile
        dummy = torch.zeros(1, 3, 512, 512).to(device)
        flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return int(flops)
    except Exception:
        pass
    return None


def compute_inference_time_ms(model, device, n_runs: int = 50):
    model.eval()
    dummy = torch.zeros(1, 3, 512, 512).to(device)
    with torch.no_grad():
        # Warm up
        for _ in range(5):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
    return elapsed / n_runs


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model_fold(model_name: str, fold: int, config: dict, device: torch.device,
                        meta_df: pd.DataFrame, smoke_test: bool = False):
    """Full evaluation for one (model, fold). Returns list of per-image row dicts."""
    print(f"\n  → {model_name} fold {fold}")

    # 1. Load model
    model = _load_model(model_name, fold, config, device)
    if model is None:
        return [], None

    # 2. Load VAL prob maps (from training) or run inference
    val_probs_path = os.path.join("checkpoints", f"{model_name}_fold{fold}", "val_probs_best.npz")
    splits_path = config["data"].get("splits_path", "splits.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    fold_info = splits_data["folds"][f"fold_{fold}"]

    if os.path.exists(val_probs_path):
        d = np.load(val_probs_path, allow_pickle=True)
        val_probs_arr = d["probs"]
        val_gt_arr    = d["gt_masks"]
    else:
        print("    val_probs_best.npz not found — running val inference for tuning")
        val_records = fold_info["val_samples"]
        if smoke_test:
            val_records = val_records[:4]
        val_ds = INbreastDataset(val_records, config=config, is_train=False)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
        val_probs_arr, val_gt_arr, _, _, _, _ = _infer(model, val_loader, device)

    # 3. Tune threshold + min_size on VAL
    best_thresh, best_min_size, best_val_dice = tune_threshold(
        val_probs_arr, val_gt_arr, config, smoke_test=smoke_test
    )
    print(f"    Tuned: threshold={best_thresh:.2f}, min_component_size={best_min_size}  "
          f"(val pos dice={best_val_dice:.4f})")

    # 4. TEST inference
    test_records = fold_info["test_samples"]
    if smoke_test:
        test_records = test_records[:4]

    test_ds = INbreastDataset(test_records, config=config, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                             num_workers=0, pin_memory=False)
    test_probs, test_gt, test_ids, mmr_list, mmc_list, ms_list = _infer(model, test_loader, device)

    # Build patient_id lookup from records
    id_to_patient = {r["image_id"]: r["patient_id"] for r in test_records}
    # ACR / BI-RADS lookup from INbreast.csv
    if meta_df is not None and len(meta_df):
        id_to_meta = {row["image_id"]: row for _, row in meta_df.iterrows()}
    else:
        id_to_meta = {}

    # 5. Compute per-image metrics
    rows = []
    n_images = len(test_probs)
    for idx in range(n_images):
        prob  = test_probs[idx]
        gt    = test_gt[idx]
        img_id = test_ids[idx]
        mmr   = mmr_list[idx]
        mmc   = mmc_list[idx]
        inf_ms = ms_list[idx]

        pred  = (prob > best_thresh).astype(np.uint8)
        if best_min_size > 0:
            pred = remove_small_components(pred, best_min_size)

        m = compute_case_metrics(pred, gt, spacing_mm=(mmr, mmc),
                                 tolerances_mm=config.get("evaluation", {}).get("nsd_tolerances_mm", [1.0, 2.0]))

        meta = id_to_meta.get(img_id, {})
        row = {
            "model":    model_name,
            "fold":     fold,
            "image_id": img_id,
            "patient_id": id_to_patient.get(img_id, ""),
            "is_positive": bool(gt.sum() > 0),
            **m,
            "max_prob": float(prob.max()),
            "threshold": best_thresh,
            "min_component_size": best_min_size,
            "inference_ms": inf_ms,
            "acr":    meta.get("acr",    ""),
            "birads": meta.get("birads", ""),
        }
        rows.append(row)

    # Save per-image CSV and prob maps
    results_dir = config["evaluation"].get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    df_rows = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, f"per_image_{model_name}_fold{fold}.csv")
    df_rows.to_csv(csv_path, index=False)
    print(f"    Saved {csv_path}  ({len(df_rows)} rows)")

    npz_path = os.path.join(results_dir, f"prob_maps_{model_name}_fold{fold}.npz")
    np.savez_compressed(npz_path, probs=test_probs, gt_masks=test_gt, image_ids=np.array(test_ids))

    # FROC data
    froc_df = compute_froc_sweep(
        test_probs, test_gt,
        [id_to_patient.get(iid, "") for iid in test_ids],
        config, fold=fold, model_name=model_name
    )
    froc_path = os.path.join(results_dir, f"froc_{model_name}_fold{fold}.csv")
    froc_df.to_csv(froc_path, index=False)

    return rows, froc_df


def load_meta_df(config: dict) -> pd.DataFrame:
    """Load INbreast.csv and produce a lookup by image_id."""
    csv_path = config["data"].get("csv_path", "INbreast_complete/INbreast.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="latin1")
        # Normalise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        # file_name column → image_id (first token)
        fname_col = next((c for c in df.columns if "file" in c and "name" in c), None)
        if fname_col is None:
            return pd.DataFrame()
        df["image_id"] = df[fname_col].astype(str).apply(lambda x: x.split("_")[0].strip())
        acr_col    = next((c for c in df.columns if "acr" in c), None)
        birads_col = next((c for c in df.columns if "bi" in c or "birad" in c), None)
        if acr_col:
            df["acr"] = df[acr_col]
        if birads_col:
            df["birads"] = df[birads_col]
        return df[["image_id"] + (["acr"] if "acr" in df else []) + (["birads"] if "birads" in df else [])]
    except Exception as e:
        print(f"  [warn] Could not load INbreast.csv: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Summary tables and stats
# ---------------------------------------------------------------------------

SUMMARY_METRICS = [
    "dice", "iou", "precision", "recall", "specificity",
    "hd95_mm", "assd_mm", "nsd_1mm",
    "lesion_recall_02", "lesion_precision_02", "lesion_f1_02",
    "lesion_recall_05", "lesion_f1_05",
]


def build_summary(all_rows: list, config: dict, n_resamples: int = 10000) -> pd.DataFrame:
    df_all = pd.DataFrame(all_rows)
    df_pos = df_all[df_all["is_positive"].astype(bool)]

    ci_rows = []
    for model_name, grp in df_all.groupby("model"):
        grp_pos = grp[grp["is_positive"].astype(bool)]
        for metric in SUMMARY_METRICS:
            use_df = grp_pos if metric not in ("specificity",) else grp
            if metric not in use_df.columns:
                continue
            mean_, lo, hi = patient_bootstrap_ci(use_df, metric, n_resamples=n_resamples)
            ci_rows.append({"model": model_name, "metric": metric,
                            "mean": mean_, "ci_lower": lo, "ci_upper": hi})

        # AUC-ROC / AUC-PR (image-level, all images)
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            y = grp["is_positive"].astype(int).values
            s = grp["max_prob"].values
            if y.min() != y.max():
                auc_roc = float(roc_auc_score(y, s))
                auc_pr  = float(average_precision_score(y, s))
            else:
                auc_roc = auc_pr = float("nan")
        except Exception:
            auc_roc = auc_pr = float("nan")
        ci_rows.append({"model": model_name, "metric": "auc_roc", "mean": auc_roc,
                        "ci_lower": float("nan"), "ci_upper": float("nan")})
        ci_rows.append({"model": model_name, "metric": "auc_pr", "mean": auc_pr,
                        "ci_lower": float("nan"), "ci_upper": float("nan")})

    return pd.DataFrame(ci_rows)


def build_pairwise_stats(all_rows: list, models: list, n_resamples: int = 10000) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pairs.append((models[i], models[j]))

    p_vals = []
    pair_rows = []
    for m_a, m_b in pairs:
        da = df[df["model"] == m_a]
        db = df[df["model"] == m_b]
        da_pos = da[da["is_positive"].astype(bool)]
        db_pos = db[db["is_positive"].astype(bool)]
        metric = "dice"
        result = pairwise_permutation_test(da_pos, db_pos, metric, n_resamples=n_resamples)
        if result == float("nan") or not isinstance(result, tuple):
            p_val, obs_diff = float("nan"), float("nan")
        else:
            p_val, obs_diff = result
        p_vals.append(p_val)
        pair_rows.append({"model_a": m_a, "model_b": m_b, "metric": metric,
                           "obs_diff": obs_diff, "p_perm": p_val})

    # Holm correction (skip NaN)
    finite_mask = [not np.isnan(p) for p in p_vals]
    finite_ps = [p for p, ok in zip(p_vals, finite_mask) if ok]
    if finite_ps:
        corrected = holm_correction(finite_ps)
        ci = 0
        for row in pair_rows:
            if not np.isnan(row["p_perm"]):
                row["p_holm"] = corrected[ci]
                row["significant"] = corrected[ci] < 0.05
                ci += 1
            else:
                row["p_holm"] = float("nan")
                row["significant"] = False
    return pd.DataFrame(pair_rows)


def build_froc_aggregate(results_dir: str, models: list) -> pd.DataFrame:
    frames = []
    for m in models:
        for f in range(5):
            path = os.path.join(results_dir, f"froc_{m}_fold{f}.csv")
            if os.path.exists(path):
                frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.groupby(["model", "threshold"]).agg(
        fppi_mean=("fppi", "mean"), fppi_std=("fppi", "std"),
        sensitivity_mean=("sensitivity", "mean"), sensitivity_std=("sensitivity", "std"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--models",  nargs="+", default=["unet", "attunet", "deeplabv3"])
    parser.add_argument("--folds",   nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--device",  default=None)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device_name = args.device or config["training"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Evaluating on device: {device}")

    n_resamples = 10 if args.smoke_test else int(
        config["evaluation"].get("bootstrap_resamples", 10000)
    )
    results_dir = config["evaluation"].get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load INbreast metadata
    meta_df = load_meta_df(config)

    # ---- params.csv ----
    params_path = os.path.join(results_dir, "params.csv")
    if not os.path.exists(params_path):
        print("\nGenerating params.csv…")
        try:
            export_parameter_counts(path=params_path, encoder_weights="imagenet")
        except Exception as e:
            print(f"  [warn] params.csv failed: {e}")

    # ---- Per-model, per-fold evaluation ----
    all_rows = []
    for model_name in args.models:
        print(f"\n=== Evaluating {model_name.upper()} ===")
        for fold in args.folds:
            rows, _ = evaluate_model_fold(
                model_name, fold, config, device, meta_df, smoke_test=args.smoke_test
            )
            if rows:
                all_rows.extend(rows)

    if not all_rows:
        print("\nNo results collected. Train models first (src/train.py).")
        return

    # Smoke-test column check
    if args.smoke_test:
        df_check = pd.DataFrame(all_rows)
        expected = [
            "model", "fold", "image_id", "patient_id", "is_positive",
            "dice", "iou", "precision", "recall", "specificity",
            "hd95_mm", "hd100_mm", "assd_mm", "nsd_1mm", "nsd_2mm",
            "penalized_hd95_mm", "mm_per_px_row", "mm_per_px_col",
            "gt_lesions", "pred_lesions",
            "tp_lesions_02", "fp_lesions_02", "fn_lesions_02",
            "lesion_recall_02", "lesion_precision_02", "lesion_f1_02",
            "tp_lesions_05", "fp_lesions_05", "fn_lesions_05",
            "lesion_recall_05", "lesion_precision_05", "lesion_f1_05",
            "max_prob", "threshold", "min_component_size", "inference_ms",
        ]
        present = set(df_check.columns)
        missing = [c for c in expected if c not in present]
        print("\n--- SMOKE TEST: Column check ---")
        print(f"Present:  {sorted(present)}")
        if missing:
            print(f"MISSING:  {missing}")
        else:
            print("ALL REQUIRED COLUMNS PRESENT ✓")
        print(df_check.head(2).to_string())
        return

    # ---- FROC aggregate ----
    froc_agg = build_froc_aggregate(results_dir, args.models)
    if len(froc_agg):
        froc_agg.to_csv(os.path.join(results_dir, "froc_aggregate.csv"), index=False)
        # Compute sens@1FPPI and sens@2FPPI per model
        for model_name, grp in froc_agg.groupby("model"):
            s1 = sensitivity_at_fppi(grp["fppi_mean"].values, grp["sensitivity_mean"].values, 1.0)
            s2 = sensitivity_at_fppi(grp["fppi_mean"].values, grp["sensitivity_mean"].values, 2.0)
            print(f"  {model_name}: sens@1FPPI={s1:.3f}  sens@2FPPI={s2:.3f}")

    # ---- Bootstrap CI summary ----
    print("\nComputing bootstrap CIs (may take a few minutes)…")
    summary_df = build_summary(all_rows, config, n_resamples=n_resamples)
    summary_df.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
    print(f"Saved {results_dir}/summary.csv")

    # ---- Pairwise stats ----
    print("Running pairwise permutation tests…")
    stats_df = build_pairwise_stats(all_rows, args.models, n_resamples=n_resamples)
    stats_df.to_csv(os.path.join(results_dir, "stats_pairwise.csv"), index=False)
    print(stats_df.to_string(index=False))

    # ---- Markdown summary table ----
    md_lines = [
        "# Evaluation Results\n",
        "## Per-metric summary (mean [95% CI], positive test images)\n",
        "| Model | Metric | Mean | 95% CI Lower | 95% CI Upper |",
        "|-------|--------|------|-------------|-------------|",
    ]
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| {row['model']} | {row['metric']} | {row['mean']:.4f} | "
            f"{row['ci_lower']:.4f} | {row['ci_upper']:.4f} |"
        )
    md_lines += [
        "\n## Pairwise significance (Dice, positive images, Holm-corrected permutation test)\n",
        "| Model A | Model B | Obs Diff | p (permutation) | p (Holm) | Significant |",
        "|---------|---------|----------|----------------|---------|-------------|",
    ]
    for _, row in stats_df.iterrows():
        md_lines.append(
            f"| {row['model_a']} | {row['model_b']} | {row['obs_diff']:.4f} | "
            f"{row['p_perm']:.4f} | {row.get('p_holm', float('nan')):.4f} | "
            f"{'Yes' if row.get('significant', False) else 'No'} |"
        )
    md_lines.append(
        "\n> **Note (108 vs 115 patients):** The paper reports 115 cases; "
        "we find 108 unique patient IDs from DICOM filenames. "
        "Medical reports show patients with multiple exam dates "
        "(e.g. `069212ec65a94339_2009_01.txt` and `_2010_01.txt`), "
        "indicating follow-up exams counted as separate cases in the paper "
        "but sharing one patient ID in the anonymised data. "
        "We split by the 108 patient IDs to prevent cross-exam leakage.\n"
    )
    md_lines.append(
        "> **Note (target=mass):** Images that contain only non-mass lesion types "
        "(e.g. calcifications, spiculated regions) count as negatives.\n"
    )
    md_lines.append(
        "> **Calcification count discrepancy:** We find 7 142 calcification ROIs in 313 images; "
        "the paper reports 6 880 in 299. We do not claim a match — "
        "version/annotation differences are possible.\n"
    )
    report_path = os.path.join(results_dir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
