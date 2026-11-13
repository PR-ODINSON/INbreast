"""Visualization utilities for INbreast lesion segmentation results."""

import glob
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

try:
    import pandas as pd
    _PD = True
except ImportError:
    _PD = False

import torch

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, INbreastDataset, preprocess_sample
from src.models import build_model

MODEL_COLORS = {"unet": "#1f77b4", "attunet": "#ff7f0e", "deeplabv3": "#2ca02c"}
MODEL_LABELS = {"unet": "U-Net", "attunet": "Attn U-Net", "deeplabv3": "DeepLabV3"}


def _require_mpl():
    if not _MPL:
        raise ImportError("matplotlib is required for visualization.")


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(checkpoint_dir: str = "checkpoints",
                         output_dir: str = "visualizations/training_curves"):
    _require_mpl()
    os.makedirs(output_dir, exist_ok=True)
    for model_name in ("unet", "attunet", "deeplabv3"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{MODEL_LABELS.get(model_name, model_name)} Training Curves", fontsize=14)
        found = False
        for fold in range(5):
            log = os.path.join(checkpoint_dir, f"{model_name}_fold{fold}", "training_log.csv")
            if not os.path.exists(log):
                continue
            df = pd.read_csv(log)
            found = True
            lbl = f"Fold {fold}"
            axes[0].plot(df["epoch"], df["train_loss"], alpha=0.7, label=f"{lbl} train")
            if "val_loss" in df:
                axes[0].plot(df["epoch"], df["val_loss"], "--", alpha=0.5, label=f"{lbl} val")
            if "val_pos_dice" in df:
                axes[1].plot(df["epoch"], df["val_pos_dice"], alpha=0.7, label=f"{lbl} pos")
            if "val_all_dice" in df:
                axes[1].plot(df["epoch"], df["val_all_dice"], "--", alpha=0.5, label=f"{lbl} all")
            if "encoder_lr" in df:
                axes[2].plot(df["epoch"], df["encoder_lr"], alpha=0.7, label=f"{lbl} enc-lr")
        for ax, title, ylabel in zip(axes, ["Loss", "Val Dice", "LR"], ["Loss", "Dice", "LR"]):
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{model_name}_training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        if found:
            print(f"Saved {path}")


# ---------------------------------------------------------------------------
# 2. Box / violin plots
# ---------------------------------------------------------------------------

def plot_boxplots(results_dir: str = "results", output_dir: str = "visualizations"):
    _require_mpl()
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    for f in glob.glob(os.path.join(results_dir, "per_image_*.csv")):
        frames.append(pd.read_csv(f))
    if not frames:
        print("No per_image CSV files found. Skipping boxplots.")
        return
    df = pd.concat(frames, ignore_index=True)
    df_pos = df[df["is_positive"].astype(bool)]

    for metric, ylabel, title in [
        ("dice",    "Dice Score",  "Dice (positive test images)"),
        ("iou",     "IoU",         "IoU (positive test images)"),
        ("hd95_mm", "HD95 (mm)",   "HD95 (positive test images)"),
    ]:
        if metric not in df_pos.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        model_names = sorted(df_pos["model"].unique())
        data = [df_pos[df_pos["model"] == m][metric].dropna().values for m in model_names]
        # Violin + box overlay
        parts = ax.violinplot(data, positions=range(len(model_names)),
                              showmedians=True, showextrema=True)
        for pc, m in zip(parts["bodies"], model_names):
            pc.set_facecolor(MODEL_COLORS.get(m, "gray"))
            pc.set_alpha(0.65)
        ax.boxplot(data, positions=range(len(model_names)), widths=0.08,
                   patch_artist=False, manage_ticks=False)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in model_names])
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = os.path.join(output_dir, f"violin_{metric}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


# ---------------------------------------------------------------------------
# 3. FROC curves
# ---------------------------------------------------------------------------

def plot_froc_curves(results_dir: str = "results", output_dir: str = "visualizations"):
    _require_mpl()
    os.makedirs(output_dir, exist_ok=True)
    agg_path = os.path.join(results_dir, "froc_aggregate.csv")
    if not os.path.exists(agg_path):
        print("froc_aggregate.csv not found. Skipping FROC plot.")
        return
    df = pd.read_csv(agg_path)
    fig, ax = plt.subplots(figsize=(9, 6))
    any_plotted = False
    for model_name, grp in df.groupby("model"):
        grp = grp.sort_values("fppi_mean")
        x  = grp["fppi_mean"].values
        y  = grp["sensitivity_mean"].values
        ye = grp["sensitivity_std"].fillna(0).values
        c  = MODEL_COLORS.get(model_name, "gray")
        ax.plot(x, y, label=MODEL_LABELS.get(model_name, model_name), color=c, linewidth=2)
        ax.fill_between(x, np.clip(y - ye, 0, 1), np.clip(y + ye, 0, 1), alpha=0.18, color=c)
        any_plotted = True
    if not any_plotted:
        plt.close(); return
    for x_ref in (1.0, 2.0):
        ax.axvline(x_ref, color="gray", linestyle=":", alpha=0.6, linewidth=1)
    ax.set_xlabel("Average False Positives per Image (FPPI)")
    ax.set_ylabel("Lesion Sensitivity")
    ax.set_title("FROC Curve (5-fold mean ± 1 std)")
    ax.set_xlim(0, 4); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "froc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# 4. Qualitative grid  (best 3 / median 3 / worst 3)
# ---------------------------------------------------------------------------

def plot_qualitative_grid(config: dict, results_dir: str = "results",
                          output_dir: str = "visualizations/qualitative", fold: int = 0):
    _require_mpl()
    os.makedirs(output_dir, exist_ok=True)
    frames = {}
    for model_name in ("unet", "attunet", "deeplabv3"):
        path = os.path.join(results_dir, f"per_image_{model_name}_fold{fold}.csv")
        if os.path.exists(path):
            frames[model_name] = pd.read_csv(path)
    if not frames:
        print("No per_image test CSVs found. Skipping qualitative grid."); return

    ref_model = next(iter(frames))
    ref_df = frames[ref_model]
    pos = ref_df[ref_df["is_positive"].astype(bool)].sort_values("dice")
    if len(pos) < 3:
        print("Too few positive images for qualitative grid."); return

    n = len(pos)
    group_ids = {
        "best":   list(pos.tail(3)["image_id"]),
        "median": list(pos.iloc[max(0, n//2 - 1):n//2 + 2]["image_id"]),
        "worst":  list(pos.head(3)["image_id"]),
    }

    # Load prob maps
    prob_maps = {}
    for model_name in frames:
        npz = os.path.join(results_dir, f"prob_maps_{model_name}_fold{fold}.npz")
        if os.path.exists(npz):
            d = np.load(npz, allow_pickle=True)
            ids = list(d["image_ids"])
            prob_maps[model_name] = {
                img_id: (d["probs"][i], d["gt_masks"][i])
                for i, img_id in enumerate(ids)
            }

    model_order = [m for m in ("unet", "attunet", "deeplabv3") if m in prob_maps]
    n_cols = 1 + len(model_order)   # GT + each model pred

    for group_name, img_ids in group_ids.items():
        for img_id in img_ids:
            avail = [m for m in model_order if img_id in prob_maps.get(m, {})]
            if not avail:
                continue
            _, gt_mask = prob_maps[avail[0]][img_id]

            fig, axes = plt.subplots(1, 1 + len(avail), figsize=(4 * (1 + len(avail)), 4))
            axes[0].imshow(gt_mask, cmap="gray")
            axes[0].set_title(f"GT mask\n{img_id[:14]}")
            axes[0].axis("off")

            for j, m in enumerate(avail):
                prob, _ = prob_maps[m][img_id]
                pred = (prob > 0.5).astype(np.float32)
                # RGB composite: GT=green, Pred=red
                r = np.clip(pred * 0.9, 0, 1)
                g = np.clip(gt_mask.astype(np.float32) * 0.9, 0, 1)
                b = np.zeros_like(r)
                composite = np.stack([r, g, b], axis=-1)
                ax = axes[1 + j]
                ax.imshow(composite)
                drow = frames.get(m, pd.DataFrame()).query(f"image_id == '{img_id}'")
                dval = f" D={drow['dice'].values[0]:.3f}" if len(drow) > 0 else ""
                ax.set_title(f"{MODEL_LABELS.get(m, m)}{dval}\n(red=pred,green=GT)")
                ax.axis("off")

            plt.suptitle(f"{group_name.capitalize()} case: {img_id}", fontsize=10)
            plt.tight_layout()
            fname = os.path.join(output_dir, f"{group_name}_{img_id[:16]}.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
    print(f"Qualitative grid saved to {output_dir}")


# ---------------------------------------------------------------------------
# 5. Attention maps
# ---------------------------------------------------------------------------

def plot_attention_maps(config: dict, fold: int = 0,
                        output_dir: str = "visualizations/attention_maps"):
    _require_mpl()
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"attunet_fold{fold}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("AttentionUNet checkpoint not found."); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)
    model  = build_model("attunet", backbone=config["models"].get("backbone", "resnet34"),
                         encoder_weights=None, classes=1)
    model.load_state_dict(ckpt["model_state_dict"])
    model  = model.to(device).eval()

    splits_path = config["data"].get("splits_path", "splits.json")
    with open(splits_path) as f:
        splits_data = json.load(f)

    test_records = splits_data["folds"][f"fold_{fold}"]["test_samples"]
    test_ds      = INbreastDataset(test_records, config=config, is_train=False)
    pos_samples  = [test_ds[i] for i in range(min(len(test_ds), 30))
                    if test_ds[i]["is_positive"]][:4]

    for idx, sample in enumerate(pos_samples):
        img_t  = sample["image"].unsqueeze(0).to(device)
        img_id = sample["image_id"]
        with torch.no_grad():
            logits, att_maps = model(img_t, return_attention=True)

        disp = (sample["image"].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
        disp = disp[:, :, 0]

        gate_names = ["gate4", "gate3", "gate2", "gate1"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        axes[0].imshow(disp, cmap="gray")
        axes[0].set_title(f"Image: {img_id}"); axes[0].axis("off")
        axes[1].imshow(sample["mask"].squeeze().numpy(), cmap="inferno")
        axes[1].set_title("Ground Truth"); axes[1].axis("off")

        for gi, gname in enumerate(gate_names):
            if gname not in att_maps or att_maps[gname] is None:
                continue
            alpha = att_maps[gname].squeeze().cpu().numpy()
            if alpha.ndim > 2:
                alpha = alpha.mean(0)
            ax = axes[2 + gi]
            im = ax.imshow(alpha, cmap="jet", vmin=0, vmax=1)
            ax.set_title(f"Gate {gname[-1]} ({alpha.shape[0]}×{alpha.shape[1]})")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f"Attention maps: {img_id}", fontsize=12)
        plt.tight_layout()
        path = os.path.join(output_dir, f"attn_{idx+1}_{img_id[:16]}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Attention maps saved to {output_dir}")


# ---------------------------------------------------------------------------
# 6. Debug overlays (mass + spiculated region in different colours)
# ---------------------------------------------------------------------------

def save_debug_overlays(config: dict, num_samples: int = 10,
                        output_dir: str = "debug_overlays"):
    _require_mpl()
    os.makedirs(output_dir, exist_ok=True)
    data_cfg  = config.get("data", {})
    dicom_dir = data_cfg.get("dicom_dir", "INbreast_complete/AllDICOMs")
    xml_dir   = data_cfg.get("xml_dir",   "INbreast_complete/AllXML")
    target    = data_cfg.get("target",    "mass")

    pos = []
    for dpath in sorted(glob.glob(os.path.join(dicom_dir, "*.dcm"))):
        img_id   = os.path.basename(dpath).split("_")[0]
        xml_path = os.path.join(xml_dir, f"{img_id}.xml")
        if not os.path.exists(xml_path):
            continue
        try:
            img, mask, _ = preprocess_sample(dpath, xml_path, config=config)
        except Exception as e:
            print(f"  skip {img_id}: {e}"); continue
        if mask.sum() > 0:
            pos.append((img_id, img, mask))
        if len(pos) >= num_samples:
            break

    for idx, (img_id, img, mask) in enumerate(pos):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img,  cmap="gray");   axes[0].set_title(f"Image {img_id}"); axes[0].axis("off")
        axes[1].imshow(mask, cmap="inferno"); axes[1].set_title(f"Mask ({target})"); axes[1].axis("off")
        overlay = np.stack([img, img * (1 - mask), img * (1 - mask)], axis=-1)
        axes[2].imshow(np.clip(overlay, 0, 1)); axes[2].set_title("Red overlay"); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{idx+1:02d}_{img_id}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Saved {len(pos)} debug overlays to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse, yaml

    parser = argparse.ArgumentParser(description="Visualization for INbreast segmentation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--curves",      action="store_true")
    parser.add_argument("--boxplots",    action="store_true")
    parser.add_argument("--froc",        action="store_true")
    parser.add_argument("--qualitative", action="store_true")
    parser.add_argument("--attention",   action="store_true")
    parser.add_argument("--overlays",    action="store_true")
    parser.add_argument("--fold",  type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = config["evaluation"].get("results_dir",        "results")
    viz_dir     = config["evaluation"].get("visualizations_dir", "visualizations")

    if args.all or args.curves:
        plot_training_curves(output_dir=os.path.join(viz_dir, "training_curves"))
    if args.all or args.boxplots:
        plot_boxplots(results_dir=results_dir, output_dir=viz_dir)
    if args.all or args.froc:
        plot_froc_curves(results_dir=results_dir, output_dir=viz_dir)
    if args.all or args.qualitative:
        plot_qualitative_grid(config, results_dir=results_dir,
                              output_dir=os.path.join(viz_dir, "qualitative"), fold=args.fold)
    if args.all or args.attention:
        plot_attention_maps(config, fold=args.fold,
                            output_dir=os.path.join(viz_dir, "attention_maps"))
    if args.overlays:
        save_debug_overlays(config, num_samples=10,
                            output_dir=config["evaluation"].get("debug_overlay_dir", "debug_overlays"))


if __name__ == "__main__":
    main()
