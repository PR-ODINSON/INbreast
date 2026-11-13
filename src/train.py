"""Training script for INbreast lesion segmentation.

Usage:
  python src/train.py --model unet --fold 0
  python src/train.py --model attunet --fold 0 --smoke_test
  python src/train.py --model deeplabv3 --fold 0 --precache
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import (
    INbreastDataset,
    build_patient_stratified_splits,
    precompute_and_cache_dataset,
    resample_training_records,
)
from src.losses import get_loss_function
from src.metrics import MetricTracker, compute_pixel_metrics
from src.models import build_model, get_optimizer_param_groups, verify_all_trainable


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, config: dict, epochs: int):
    """Return (scheduler, is_plateau). Plateau schedulers need a metric."""
    sched_type = config["training"].get("scheduler", "cosine_warmup").lower()
    if sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config["training"].get("plateau_factor", 0.5),
            patience=config["training"].get("plateau_patience", 5),
            min_lr=config["training"].get("min_lr", 1e-6),
        )
        return scheduler, True
    else:  # cosine_warmup (default)
        warmup = int(config["training"].get("warmup_epochs", 5))
        warmup = min(warmup, epochs - 1)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup),
            eta_min=config["training"].get("min_lr", 1e-6),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
        )
        return scheduler, False


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        bsz = images.size(0)

        optimizer.zero_grad()
        use_amp = device.type == "cuda" and scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        pred_masks = (probs > 0.5).astype(np.uint8)
        gt_masks = masks.detach().cpu().numpy().astype(np.uint8)
        batch_dice = sum(
            compute_pixel_metrics(pred_masks[i, 0], gt_masks[i, 0])["dice"]
            for i in range(bsz)
        )

        running_loss += loss.item() * bsz
        running_dice += batch_dice
        total_samples += bsz
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{running_dice/total_samples:.4f}"})

    return running_loss / total_samples, running_dice / total_samples


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    """Returns (summary_dict, probs_arr, gt_arr, image_ids)."""
    model.eval()
    running_loss = 0.0
    tracker = MetricTracker(lesion_iou_threshold=0.2)
    total_samples = 0
    all_probs, all_gt, all_ids = [], [], []

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        bsz = images.size(0)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        probs = torch.sigmoid(logits).cpu().numpy()   # (B,1,H,W)
        gt = masks.cpu().numpy().astype(np.uint8)
        pred_masks = (probs > threshold).astype(np.uint8)

        for i in range(bsz):
            tracker.update(pred_masks[i, 0], gt[i, 0], image_id=batch["image_id"][i])
            all_probs.append(probs[i, 0])
            all_gt.append(gt[i, 0])
            all_ids.append(batch["image_id"][i])

        running_loss += loss.item() * bsz
        total_samples += bsz

    summary = tracker.summary()
    summary["val_loss"] = running_loss / total_samples
    return (
        summary,
        np.stack(all_probs, axis=0).astype(np.float32),
        np.stack(all_gt, axis=0).astype(np.uint8),
        all_ids,
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_curves(history: pd.DataFrame, save_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["epoch"], history["train_dice"], label="Train Dice")
    axes[1].plot(history["epoch"], history["val_all_dice"], label="Val All Dice")
    axes[1].plot(history["epoch"], history["val_pos_dice"], label="Val Pos Dice")
    axes[1].set_title("Dice"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(history["epoch"], history["val_all_iou"], label="Val All IoU")
    axes[2].plot(history["epoch"], history["val_pos_iou"], label="Val Pos IoU")
    axes[2].set_title("IoU"); axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train segmentation models on INbreast")
    parser.add_argument("--model", required=True, choices=["unet", "attunet", "deeplabv3"])
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--precache", action="store_true")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Limit to tiny data subset for quick sanity check")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Smoke-test overrides
    if args.smoke_test:
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 2
        config["training"]["min_epochs"] = 1
        config["training"]["patience"] = 99
        config["data"]["use_cache"] = False
        print("[SMOKE TEST] epochs=2, batch_size=2, cache disabled")

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["training"]["device"] = args.device

    seed = int(config["data"].get("seed", 42))
    set_seed(seed + args.fold)

    device_name = config["training"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Device: {device}")

    # ---- Splits ----
    splits_path = config["data"].get("splits_path", "splits.json")
    if not os.path.exists(splits_path):
        print(f"{splits_path} not found — generating splits…")
        splits_data = build_patient_stratified_splits(config, output_path=splits_path)
    else:
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

    # Validate splits scheme
    scheme = splits_data.get("scheme", "")
    if "nested" not in scheme:
        print(f"WARNING: splits.json scheme='{scheme}' — expected nested folds. "
              "Regenerating with correct scheme…")
        splits_data = build_patient_stratified_splits(config, output_path=splits_path)

    fold_key = f"fold_{args.fold}"
    if fold_key not in splits_data["folds"]:
        raise ValueError(f"Fold '{fold_key}' not found in {splits_path}")

    fold_info = splits_data["folds"][fold_key]

    # Require 3-way split
    for key in ("train_patients", "val_patients", "test_patients",
                "train_samples", "val_samples", "test_samples"):
        if key not in fold_info:
            raise KeyError(f"splits.json fold_{args.fold} missing key '{key}'. "
                           "Re-run dataset.py to regenerate splits.")

    train_pats = set(fold_info["train_patients"])
    val_pats   = set(fold_info["val_patients"])
    test_pats  = set(fold_info["test_patients"])
    assert train_pats & val_pats  == set(), "train∩val patient overlap!"
    assert train_pats & test_pats == set(), "train∩test patient overlap!"
    assert val_pats  & test_pats  == set(), "val∩test patient overlap!"

    all_train_records = fold_info["train_samples"]
    val_records       = fold_info["val_samples"]

    n_test = len(fold_info["test_samples"])
    print(
        f"Fold {args.fold}: train={len(all_train_records)} imgs ({len(train_pats)} pts), "
        f"val={len(val_records)} imgs ({len(val_pats)} pts), "
        f"test={n_test} imgs ({len(test_pats)} pts) — test kept for evaluate.py only"
    )

    if args.smoke_test:
        all_train_records = sorted(all_train_records, key=lambda r: r["image_id"])[:8]
        val_records       = sorted(val_records,       key=lambda r: r["image_id"])[:4]

    if args.precache:
        precompute_and_cache_dataset(config)

    # ---- Model ----
    backbone = config["models"].get("backbone", "resnet34")
    weights  = config["models"].get("encoder_weights", "imagenet")
    classes  = int(config["models"].get("classes", 1))
    model = build_model(args.model, backbone=backbone, encoder_weights=weights,
                        in_channels=3, classes=classes)
    model = model.to(device)

    # ---- Optimizer ----
    enc_lr = float(config["training"].get("encoder_lr", 1e-4))
    dec_lr = float(config["training"].get("decoder_lr", 1e-3))
    wd     = float(config["training"].get("weight_decay", 1e-4))
    param_groups = get_optimizer_param_groups(model, encoder_lr=enc_lr, decoder_lr=dec_lr, weight_decay=wd)
    optimizer = torch.optim.AdamW(param_groups)

    epochs     = int(config["training"].get("epochs", 100))
    min_epochs = int(config["training"].get("min_epochs", 20))
    patience   = int(config["training"].get("patience", 15))
    win_size   = int(config["training"].get("early_stop_window", 3))
    grad_clip  = float(config["training"].get("grad_clip", 1.0))
    neg_ratio  = config["data"].get("negative_ratio", 1.0)
    batch_size = int(config["training"].get("batch_size", 8))
    num_workers = int(config["training"].get("num_workers", 4))
    if os.name == "nt" or device.type == "cpu":
        num_workers = min(num_workers, 2)

    scheduler, is_plateau = make_scheduler(optimizer, config, epochs)
    criterion = get_loss_function(config)
    scaler = (
        torch.cuda.amp.GradScaler()
        if device.type == "cuda" and config["training"].get("mixed_precision", True)
        else None
    )

    checkpoint_dir = os.path.join("checkpoints", f"{args.model}_fold{args.fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Validation DataLoader (fixed, all val images)
    val_dataset = INbreastDataset(val_records, config=config, is_train=False)
    val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # Training state
    best_moving_avg = -1.0
    patience_counter = 0
    dice_window = []
    history = []

    print(f"\n--- {args.model.upper()} | Fold {args.fold} | {epochs} epochs ---")
    for epoch in range(1, epochs + 1):
        # Per-epoch negative resampling (same seed sequence for all models)
        epoch_records = resample_training_records(all_train_records, neg_ratio, seed, epoch)
        train_dataset = INbreastDataset(epoch_records, config=config, is_train=True)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=(device.type == "cuda"))

        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, grad_clip=grad_clip
        )
        val_summary, val_probs, val_gt, val_ids = validate(
            model, val_loader, criterion, device
        )

        val_pos_dice = val_summary.get("pos_dice", 0.0)
        val_all_dice = val_summary.get("all_dice", 0.0)
        val_loss     = val_summary["val_loss"]

        # 3-epoch moving average
        dice_window.append(val_pos_dice if val_summary.get("num_pos_images", 0) > 0 else val_all_dice)
        if len(dice_window) > win_size:
            dice_window.pop(0)
        moving_avg = float(np.mean(dice_window))

        # Scheduler step
        if is_plateau:
            scheduler.step(moving_avg)
        else:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_all_dice": val_all_dice,
            "val_pos_dice": val_pos_dice,
            "val_all_iou": val_summary.get("all_iou", 0.0),
            "val_pos_iou": val_summary.get("pos_iou", 0.0),
            "val_moving_avg": moving_avg,
            "val_all_sensitivity": val_summary.get("all_sensitivity", 0.0),
            "val_all_precision": val_summary.get("all_precision", 0.0),
            "val_fp_per_image": val_summary.get("fp_per_image", 0.0),
            "val_lesion_recall": val_summary.get("overall_lesion_recall", 0.0),
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "decoder_lr": optimizer.param_groups[1]["lr"],
            "n_train_samples": len(epoch_records),
        }
        history.append(record)

        print(
            f"Ep {epoch:03d}/{epochs} | "
            f"TrLoss={train_loss:.4f} TrDice={train_dice:.4f} | "
            f"ValLoss={val_loss:.4f} ValDice(pos/all)={val_pos_dice:.4f}/{val_all_dice:.4f} | "
            f"MovAvg={moving_avg:.4f}"
        )

        # Save best checkpoint
        improved = moving_avg > best_moving_avg
        if improved:
            best_moving_avg = moving_avg
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": best_moving_avg,
                    "val_summary": val_summary,
                    "config": config,
                    "model_name": args.model,
                    "fold": args.fold,
                    "threshold": 0.5,  # placeholder; tuned in evaluate.py
                },
                best_path,
            )
            # Save val prob maps for threshold tuning in evaluate.py
            probs_path = os.path.join(checkpoint_dir, "val_probs_best.npz")
            np.savez_compressed(
                probs_path,
                probs=val_probs,
                gt_masks=val_gt,
                image_ids=np.array(val_ids),
            )
            print(f"  [*] New best moving avg={best_moving_avg:.4f}  → saved {best_path}")

        # Early stopping (only after min_epochs)
        if epoch >= min_epochs:
            if not improved:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[!] Early stop at epoch {epoch} (patience={patience})")
                    break
            else:
                patience_counter = 0

        # Save last checkpoint + log
        last_path = os.path.join(checkpoint_dir, "last_model.pth")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "config": config, "model_name": args.model, "fold": args.fold}, last_path)

        df_history = pd.DataFrame(history)
        df_history.to_csv(os.path.join(checkpoint_dir, "training_log.csv"), index=False)
        plot_curves(df_history, os.path.join(checkpoint_dir, "training_curves.png"))

    print(f"\nDone: {args.model} fold {args.fold}. Best moving-avg val Dice = {best_moving_avg:.4f}")


if __name__ == "__main__":
    main()
