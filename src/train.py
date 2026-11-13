import os
import sys
import argparse
import json
import random
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import INbreastDataset, build_patient_stratified_splits, precompute_and_cache_dataset
from src.models import build_model, get_optimizer_param_groups, verify_all_trainable
from src.losses import get_loss_function
from src.metrics import compute_pixel_metrics, MetricTracker


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Mixed precision training
        use_amp = (device.type == "cuda") and (scaler is not None)
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

        # Compute batch metrics
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        pred_masks = (probs > 0.5).astype(np.uint8)
        gt_masks = masks.detach().cpu().numpy().astype(np.uint8)

        batch_dice = sum(compute_pixel_metrics(pred_masks[i, 0], gt_masks[i, 0])["dice"] for i in range(batch_size))

        running_loss += loss.item() * batch_size
        running_dice += batch_dice
        total_samples += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{(running_dice / total_samples):.4f}"})

    epoch_loss = running_loss / total_samples
    epoch_dice = running_dice / total_samples
    return epoch_loss, epoch_dice


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    tracker = MetricTracker(lesion_iou_threshold=0.2)
    total_samples = 0

    pbar = tqdm(loader, desc="Validation", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        probs = torch.sigmoid(logits).cpu().numpy()
        pred_masks = (probs > threshold).astype(np.uint8)
        gt_masks = masks.cpu().numpy().astype(np.uint8)
        image_ids = batch["image_id"]

        for i in range(batch_size):
            tracker.update(pred_masks[i, 0], gt_masks[i, 0], image_id=image_ids[i])

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    summary = tracker.summary()
    summary["val_loss"] = epoch_loss
    return summary


def plot_curves(history: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", color="red")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Dice
    axes[1].plot(history["epoch"], history["train_dice"], label="Train Dice", color="blue")
    axes[1].plot(history["epoch"], history["val_all_dice"], label="Val All Dice", color="green")
    axes[1].plot(history["epoch"], history["val_pos_dice"], label="Val Pos Dice", color="orange")
    axes[1].set_title("Dice Metric")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].legend()
    axes[1].grid(True)

    # IoU
    axes[2].plot(history["epoch"], history["val_all_iou"], label="Val All IoU", color="purple")
    axes[2].plot(history["epoch"], history["val_pos_iou"], label="Val Pos IoU", color="brown")
    axes[2].set_title("IoU Metric")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU Score")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune segmentation models on INbreast dataset")
    parser.add_argument("--model", type=str, required=True, choices=["unet", "attunet", "deeplabv3"], help="Model architecture")
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4], help="Fold index (0 to 4)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cuda' or 'cpu')")
    parser.add_argument("--precache", action="store_true", help="Precompute and cache preprocessed images before training")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Command-line overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["training"]["device"] = args.device

    # Set random seed
    seed = config["data"].get("seed", 42)
    set_seed(seed + args.fold)

    # Device selection
    device_name = config["training"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Splits management
    splits_path = config["data"].get("splits_path", "splits.json")
    if not os.path.exists(splits_path):
        print(f"Splits file {splits_path} not found. Generating 5-fold patient-stratified splits...")
        splits_data = build_patient_stratified_splits(config, output_path=splits_path)
    else:
        with open(splits_path, "r") as f:
            splits_data = json.load(f)

    # Optional dataset pre-caching
    if args.precache:
        precompute_and_cache_dataset(config)

    # Get fold data
    fold_key = f"fold_{args.fold}"
    if fold_key not in splits_data["folds"]:
        raise ValueError(f"Fold '{fold_key}' not found in {splits_path}")

    fold_info = splits_data["folds"][fold_key]
    train_records = fold_info["train_samples"]
    val_records = fold_info["val_samples"]

    # Hard Rule check: assert zero patient overlap
    train_patients = set(fold_info["train_patients"])
    val_patients = set(fold_info["val_patients"])
    overlap = train_patients.intersection(val_patients)
    assert len(overlap) == 0, f"PATIENT LEAKAGE DETECTED in {fold_key}: {overlap}"
    print(f"Fold {args.fold}: {len(train_records)} train samples ({len(train_patients)} patients), {len(val_records)} val samples ({len(val_patients)} patients). Zero leakage verified.")

    # Datasets and Loaders
    train_dataset = INbreastDataset(train_records, config=config, is_train=True)
    val_dataset = INbreastDataset(val_records, config=config, is_train=False)

    batch_size = config["training"].get("batch_size", 8)
    num_workers = config["training"].get("num_workers", 4)
    # If on Windows/CPU, keep num_workers conservative
    if os.name == "nt" or device.type == "cpu":
        num_workers = min(num_workers, 2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # Build Model
    model_name = args.model
    backbone = config["models"].get("backbone", "resnet34")
    weights = config["models"].get("encoder_weights", "imagenet")
    classes = config["models"].get("classes", 1)

    model = build_model(model_name=model_name, backbone=backbone, encoder_weights=weights, in_channels=3, classes=classes)
    model = model.to(device)

    # Optimizer with differential learning rates (100% unfrozen parameter groups)
    enc_lr = config["training"].get("encoder_lr", 1e-4)
    dec_lr = config["training"].get("decoder_lr", 1e-3)
    wd = config["training"].get("weight_decay", 1e-4)
    param_groups = get_optimizer_param_groups(model, encoder_lr=enc_lr, decoder_lr=dec_lr, weight_decay=wd)
    optimizer = torch.optim.AdamW(param_groups)

    # Scheduler and Criterion
    epochs = config["training"].get("epochs", 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = get_loss_function(config)

    # GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and config["training"].get("mixed_precision", True)) else None

    # Checkpoint directories
    checkpoint_dir = os.path.join("checkpoints", f"{model_name}_fold{args.fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training state
    best_val_dice = -1.0
    patience = config["training"].get("patience", 15)
    patience_counter = 0
    history = []

    print(f"\n--- Starting Training: {model_name.upper()} | Fold {args.fold} | Epochs: {epochs} ---")
    for epoch in range(1, epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=config["training"].get("grad_clip", 1.0)
        )

        val_summary = validate(model=model, loader=val_loader, criterion=criterion, device=device)
        scheduler.step()

        val_loss = val_summary["val_loss"]
        val_all_dice = val_summary["all_dice"]
        val_pos_dice = val_summary["pos_dice"]
        val_all_iou = val_summary["all_iou"]
        val_pos_iou = val_summary["pos_iou"]

        # Log history
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_all_dice": val_all_dice,
            "val_pos_dice": val_pos_dice,
            "val_all_iou": val_all_iou,
            "val_pos_iou": val_pos_iou,
            "val_all_sensitivity": val_summary["all_sensitivity"],
            "val_all_precision": val_summary["all_precision"],
            "val_fp_per_image": val_summary["fp_per_image"],
            "val_lesion_recall": val_summary["overall_lesion_recall"],
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "decoder_lr": optimizer.param_groups[1]["lr"]
        }
        history.append(record)

        print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice (Pos/All): {val_pos_dice:.4f} / {val_all_dice:.4f} | Val IoU: {val_pos_iou:.4f}")

        # Checkpointing
        # We monitor validation positive dice (or all dice)
        current_monitor = val_pos_dice if val_summary["num_pos_images"] > 0 else val_all_dice
        if current_monitor > best_val_dice:
            best_val_dice = current_monitor
            patience_counter = 0
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": current_monitor,
                "val_summary": val_summary,
                "config": config,
                "model_name": model_name,
                "fold": args.fold
            }, best_path)
            print(f"[*] New best validation Dice: {best_val_dice:.4f}! Saved to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch} (patience={patience})")
                break

        # Save last checkpoint
        last_path = os.path.join(checkpoint_dir, "last_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "model_name": model_name,
            "fold": args.fold
        }, last_path)

        # Save training log and plots
        df_history = pd.DataFrame(history)
        df_history.to_csv(os.path.join(checkpoint_dir, "training_log.csv"), index=False)
        plot_curves(df_history, os.path.join(checkpoint_dir, "training_curves.png"))

    print(f"\nTraining completed for {model_name} Fold {args.fold}. Best Val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()

