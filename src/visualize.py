import os
import sys
import glob
import json
import yaml
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import preprocess_sample, INbreastDataset, IMAGENET_MEAN, IMAGENET_STD
from src.models import build_model


def save_debug_overlays(config: dict, num_samples: int = 10, output_dir: str = "debug_overlays"):
    """
    Generates and saves image + mask contour overlays for raw DICOMs and XMLs
    to visually confirm precise contour alignment on lesion tissue.
    """
    os.makedirs(output_dir, exist_ok=True)
    data_cfg = config.get("data", {})
    dicom_dir = data_cfg.get("dicom_dir", "INbreast_complete/AllDICOMs")
    xml_dir = data_cfg.get("xml_dir", "INbreast_complete/AllXML")
    target = data_cfg.get("target", "mass")

    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    positive_samples = []

    for dpath in dicom_files:
        img_id = os.path.basename(dpath).split("_")[0]
        xml_path = os.path.join(xml_dir, f"{img_id}.xml")
        if os.path.exists(xml_path):
            img, mask = preprocess_sample(dpath, xml_path, config=config)
            if mask.sum() > 0:
                positive_samples.append((img_id, dpath, xml_path, img, mask))
                if len(positive_samples) >= num_samples:
                    break

    print(f"Generating {len(positive_samples)} debug overlays in '{output_dir}'...")
    for idx, (img_id, dpath, xml_path, img, mask) in enumerate(positive_samples):
        # Convert float [0, 1] image to uint8 [0, 255] RGB
        img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
        rgb_img = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

        # Draw ground-truth contour in bright green
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = rgb_img.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=2)

        # Plot side-by-side comparison: Raw Preprocessed, Mask, and Overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"Preprocessed Mammogram ({img_id})")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="inferno")
        axes[1].set_title(f"Target Mask ({target})")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Aligned Contour Overlay (Green)")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{idx+1:02d}_{img_id}_alignment_check.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Debug overlays successfully saved to '{output_dir}'.")


def generate_qualitative_comparison(models: list, fold: int, config: dict, device: torch.device, output_dir: str = "visualizations"):
    """
    Generates comparison figures across models for selected test/val cases,
    showing successes (high Dice) and failure modes (false positives / false negatives).
    """
    os.makedirs(output_dir, exist_ok=True)
    splits_path = config["data"].get("splits_path", "splits.json")
    with open(splits_path, "r") as f:
        splits_data = json.load(f)

    val_records = splits_data["folds"][f"fold_{fold}"]["val_samples"]
    val_dataset = INbreastDataset(val_records, config=config, is_train=False)

    # Load all models
    loaded_models = {}
    for mname in models:
        ckpt_path = os.path.join("checkpoints", f"{mname}_fold{fold}", "best_model.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            m = build_model(mname, backbone=config["models"].get("backbone", "resnet34"), classes=1)
            m.load_state_dict(ckpt["model_state_dict"])
            m = m.to(device).eval()
            loaded_models[mname] = m

    if not loaded_models:
        print("No trained models found for qualitative comparison.")
        return

    # Select positive samples
    samples = [val_dataset[i] for i in range(len(val_dataset)) if val_dataset[i]["is_positive"]]
    if len(samples) > 6:
        samples = samples[:6]

    for s_idx, sample in enumerate(samples):
        img_t = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["mask"].squeeze().numpy().astype(np.uint8)
        img_id = sample["image_id"]

        # Reconstruct display image from ImageNet normalized tensor
        disp_img = sample["image"].permute(1, 2, 0).numpy()
        disp_img = (disp_img * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)[:, :, 0]

        predictions = {}
        for mname, m in loaded_models.items():
            with torch.no_grad():
                logits = m(img_t)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                pred_bin = (prob > 0.5).astype(np.uint8)
                predictions[mname] = pred_bin

        # Plot qualitative row
        n_plots = 2 + len(predictions)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

        axes[0].imshow(disp_img, cmap="gray")
        axes[0].set_title(f"Image: {img_id}")
        axes[0].axis("off")

        # Ground truth overlay
        gt_overlay = cv2.cvtColor((disp_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        gt_cnts, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_overlay, gt_cnts, -1, (0, 255, 0), 2)
        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth (Green)")
        axes[1].axis("off")

        # Predictions overlays
        for p_idx, (mname, p_mask) in enumerate(predictions.items()):
            ax = axes[2 + p_idx]
            pred_overlay = cv2.cvtColor((disp_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            cv2.drawContours(pred_overlay, gt_cnts, -1, (0, 255, 0), 1)  # GT in thin green
            pred_cnts, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pred_overlay, pred_cnts, -1, (255, 0, 0), 2)  # Pred in red
            ax.imshow(pred_overlay)
            ax.set_title(f"{mname.upper()} Pred (Red)")
            ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"qualitative_{s_idx+1}_{img_id}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Qualitative comparisons saved to '{output_dir}'.")


def visualize_attention_maps(fold: int, config: dict, device: torch.device, output_dir: str = "visualizations/attention_maps"):
    """
    Extracts and visualizes the attention coefficient maps from the Oktay et al.
    grid attention gates in Attention U-Net.
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"attunet_fold{fold}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("Attention U-Net checkpoint not found. Cannot visualize attention maps.")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model("attunet", backbone=config["models"].get("backbone", "resnet34"), classes=1)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    splits_path = config["data"].get("splits_path", "splits.json")
    with open(splits_path, "r") as f:
        splits_data = json.load(f)

    val_records = splits_data["folds"][f"fold_{fold}"]["val_samples"]
    val_dataset = INbreastDataset(val_records, config=config, is_train=False)

    pos_samples = [val_dataset[i] for i in range(len(val_dataset)) if val_dataset[i]["is_positive"]][:4]

    for idx, sample in enumerate(pos_samples):
        img_t = sample["image"].unsqueeze(0).to(device)
        img_id = sample["image_id"]

        with torch.no_grad():
            logits, att_maps = model(img_t, return_attention=True)

        disp_img = sample["image"].permute(1, 2, 0).numpy()
        disp_img = (disp_img * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)[:, :, 0]

        # 4 attention gates: gate1, gate2, gate3, gate4
        fig, axes = plt.subplots(1, 6, figsize=(24, 4))
        axes[0].imshow(disp_img, cmap="gray")
        axes[0].set_title(f"Image: {img_id}")
        axes[0].axis("off")

        axes[1].imshow(sample["mask"].squeeze().numpy(), cmap="inferno")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        for g_idx, g_name in enumerate(["gate4", "gate3", "gate2", "gate1"]):
            alpha = att_maps[g_name].squeeze().cpu().numpy()
            ax = axes[2 + g_idx]
            im = ax.imshow(alpha, cmap="jet", vmin=0, vmax=1)
            ax.set_title(f"Attention Gate {4 - g_idx} (Level {g_name[-1]})")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"attention_map_{idx+1}_{img_id}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Attention map visualizations saved to '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Visualization utilities for INbreast lesion segmentation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--overlays", action="store_true", help="Generate debug alignment overlays")
    parser.add_argument("--qualitative", action="store_true", help="Generate qualitative comparison across models")
    parser.add_argument("--attention", action="store_true", help="Generate attention map heatmaps")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to visualize")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.overlays or (not args.qualitative and not args.attention):
        save_debug_overlays(config, num_samples=10, output_dir=config["evaluation"].get("debug_overlay_dir", "debug_overlays"))

    if args.qualitative:
        generate_qualitative_comparison(models=["unet", "attunet", "deeplabv3"], fold=args.fold, config=config, device=device)

    if args.attention:
        visualize_attention_maps(fold=args.fold, config=config, device=device)


if __name__ == "__main__":
    main()

