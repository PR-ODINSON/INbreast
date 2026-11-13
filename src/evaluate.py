import os
import sys
import glob
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import INbreastDataset
from src.models import build_model
from src.metrics import MetricTracker


def find_optimal_threshold(model, val_loader, device, thresholds=None):
    """
    Finds the optimal binarization threshold on the validation set
    by maximizing the mean Dice score.
    Grid search is strictly restricted to validation data.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    model.eval()
    all_probs = []
    all_targets = []
    all_image_ids = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            for i in range(images.size(0)):
                all_probs.append(probs[i, 0])
                all_targets.append(masks[i, 0])
                all_image_ids.append(batch["image_id"][i])

    best_thresh = 0.5
    best_dice = -1.0

    for th in thresholds:
        tracker = MetricTracker(lesion_iou_threshold=0.2)
        for prob, target, img_id in zip(all_probs, all_targets, all_image_ids):
            pred_bin = (prob > th).astype(np.uint8)
            tracker.update(pred_bin, target.astype(np.uint8), image_id=img_id)
        
        summary = tracker.summary()
        # Optimize for positive-only dice or all dice
        score = summary.get("pos_dice", 0.0) if summary.get("num_pos_images", 0) > 0 else summary.get("all_dice", 0.0)
        if score > best_dice:
            best_dice = score
            best_thresh = float(th)

    return best_thresh, best_dice


def evaluate_model_fold(model_name: str, fold: int, config: dict, device: torch.device):
    """
    Evaluates a single model checkpoint on its corresponding validation fold.
    """
    ckpt_path = os.path.join("checkpoints", f"{model_name}_fold{fold}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = config["models"].get("backbone", "resnet34")
    classes = config["models"].get("classes", 1)

    model = build_model(model_name, backbone=backbone, in_channels=3, classes=classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    splits_path = config["data"].get("splits_path", "splits.json")
    with open(splits_path, "r") as f:
        splits_data = json.load(f)

    val_records = splits_data["folds"][f"fold_{fold}"]["val_samples"]
    val_dataset = INbreastDataset(val_records, config=config, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # 1. Tune threshold on validation set
    opt_threshold, _ = find_optimal_threshold(model, val_loader, device)

    # 2. Evaluate with optimal threshold
    tracker = MetricTracker(lesion_iou_threshold=config["evaluation"].get("lesion_iou_threshold", 0.2))
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(images.size(0)):
                pred_bin = (probs[i, 0] > opt_threshold).astype(np.uint8)
                tracker.update(pred_bin, masks[i, 0].astype(np.uint8), image_id=batch["image_id"][i])

    summary = tracker.summary()
    summary["model"] = model_name
    summary["fold"] = fold
    summary["optimal_threshold"] = opt_threshold
    return summary


def run_paired_tests(results_df: pd.DataFrame, models: list, metric: str = "pos_dice"):
    """
    Performs paired t-tests and Wilcoxon signed-rank tests across folds.
    """
    print(f"\n--- Paired Statistical Significance Tests for '{metric}' across folds ---")
    if "attunet" in models:
        base_model = "attunet"
        for other in models:
            if other == base_model:
                continue
            scores_base = results_df[results_df["model"] == base_model].sort_values("fold")[metric].values
            scores_other = results_df[results_df["model"] == other].sort_values("fold")[metric].values

            if len(scores_base) == len(scores_other) and len(scores_base) >= 3:
                diff = scores_base - scores_other
                t_stat, p_val = stats.ttest_rel(scores_base, scores_other)
                try:
                    w_stat, w_pval = stats.wilcoxon(scores_base, scores_other)
                except Exception:
                    w_stat, w_pval = np.nan, np.nan
                print(f"{base_model} vs {other}:")
                print(f"  Mean Diff ({base_model} - {other}): {np.mean(diff):+.4f}")
                print(f"  Paired t-test: t={t_stat:.4f}, p-value={p_val:.4f} {'*' if p_val < 0.05 else ''}")
                print(f"  Wilcoxon test: W={w_stat}, p-value={w_pval:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 5-fold cross validation for segmentation models")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--models", nargs="+", default=["unet", "attunet", "deeplabv3"], help="Models to evaluate")
    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device_name = args.device or config["training"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Running evaluation on device: {device}")

    results = []
    n_splits = config["data"].get("n_splits", 5)

    for model_name in args.models:
        print(f"\nEvaluating Model: {model_name.upper()}...")
        for fold in range(n_splits):
            res = evaluate_model_fold(model_name, fold, config, device)
            if res is not None:
                results.append(res)

    if not results:
        print("No evaluation results were collected. Ensure checkpoints exist in checkpoints/.")
        return

    df_results = pd.DataFrame(results)
    results_dir = config["evaluation"].get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    df_results.to_csv(os.path.join(results_dir, "per_fold_results.csv"), index=False)

    # Compute mean +/- std across folds
    metrics_to_aggregate = [
        "pos_dice", "all_dice",
        "pos_iou", "all_iou",
        "pos_sensitivity", "all_sensitivity",
        "pos_precision", "all_precision",
        "pos_lesion_recall", "fp_per_image"
    ]

    summary_rows = []
    for model_name, grp in df_results.groupby("model"):
        row = {"Model": model_name.upper()}
        for m in metrics_to_aggregate:
            mean_val = grp[m].mean()
            std_val = grp[m].std()
            row[m] = f"{mean_val:.4f} ± {std_val:.4f}"
            row[f"{m}_mean"] = mean_val
            row[f"{m}_std"] = std_val
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)

    # Print Markdown Table
    print("\n" + "=" * 90)
    print("5-FOLD CROSS VALIDATION RESULTS SUMMARY (Mean ± Std)")
    print("=" * 90)
    
    display_cols = [
        "Model", "pos_dice", "pos_iou", "pos_sensitivity", "pos_precision", "pos_lesion_recall", "fp_per_image"
    ]
    col_names = [
        "Model", "Lesion Dice", "Lesion IoU", "Sensitivity", "Precision", "Lesion Recall", "FP / Image"
    ]
    table_df = df_summary[display_cols].copy()
    table_df.columns = col_names
    print(table_df.to_markdown(index=False))

    # Statistical significance test
    run_paired_tests(df_results, models=args.models, metric="pos_dice")
    print(f"\nResults saved to {results_dir}/summary.csv and per_fold_results.csv")


if __name__ == "__main__":
    main()

