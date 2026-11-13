import numpy as np
import torch
from scipy.ndimage import label


def compute_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> dict:
    """
    Computes standard pixel-level segmentation metrics for binary masks:
    - Dice (F1-score)
    - IoU (Jaccard Index)
    - Sensitivity / Recall
    - Precision
    - Specificity
    """
    pred = (pred_mask > 0).astype(np.bool_)
    gt = (gt_mask > 0).astype(np.bool_)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    intersection = tp
    union = tp + fp + fn

    # Handle true negative case (both pred and gt are empty)
    if gt.sum() == 0:
        if pred.sum() == 0:
            dice = 1.0
            iou = 1.0
            sensitivity = 1.0
            precision = 1.0
        else:
            dice = 0.0
            iou = 0.0
            sensitivity = 1.0  # No positive cases to miss
            precision = 0.0
    else:
        dice = float((2.0 * intersection) / (pred.sum() + gt.sum() + eps))
        iou = float(intersection / (union + eps))
        sensitivity = float(tp / (tp + fn + eps))
        precision = float(tp / (tp + fp + eps))

    specificity = float(tn / (tn + fp + eps)) if (tn + fp) > 0 else 1.0

    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "precision": precision,
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "is_positive": bool(gt.sum() > 0)
    }


def compute_lesion_level_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, iou_threshold: float = 0.2) -> dict:
    """
    Computes lesion-level detection metrics using connected component analysis:
    - Each separate connected component in gt_mask is a distinct ground-truth lesion.
    - Each separate connected component in pred_mask is a candidate predicted lesion.
    - A predicted lesion is a True Positive (TP) hit if its IoU with any GT lesion >= iou_threshold.
    - Any predicted component that does not match a GT lesion is a False Positive (FP).
    - Any GT lesion with no matching predicted component is a False Negative (FN).
    """
    gt_binary = (gt_mask > 0).astype(np.uint8)
    pred_binary = (pred_mask > 0).astype(np.uint8)

    labeled_gt, num_gt_lesions = label(gt_binary)
    labeled_pred, num_pred_lesions = label(pred_binary)

    if num_gt_lesions == 0:
        # Negative image (no lesions)
        return {
            "gt_lesions": 0,
            "pred_lesions": num_pred_lesions,
            "tp_lesions": 0,
            "fp_lesions": num_pred_lesions,
            "fn_lesions": 0,
            "lesion_recall": 1.0 if num_pred_lesions == 0 else 0.0,
            "lesion_precision": 1.0 if num_pred_lesions == 0 else 0.0
        }

    matched_gt = set()
    matched_pred = set()

    for p_idx in range(1, num_pred_lesions + 1):
        p_comp = (labeled_pred == p_idx)
        best_iou = 0.0
        best_g_idx = None
        
        for g_idx in range(1, num_gt_lesions + 1):
            g_comp = (labeled_gt == g_idx)
            inter = np.logical_and(p_comp, g_comp).sum()
            if inter > 0:
                union = np.logical_or(p_comp, g_comp).sum()
                iou = inter / float(union)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx
        
        if best_iou >= iou_threshold:
            matched_pred.add(p_idx)
            matched_gt.add(best_g_idx)

    tp_lesions = len(matched_gt)
    fp_lesions = num_pred_lesions - len(matched_pred)
    fn_lesions = num_gt_lesions - len(matched_gt)

    lesion_recall = tp_lesions / float(num_gt_lesions) if num_gt_lesions > 0 else 1.0
    lesion_precision = tp_lesions / float(num_pred_lesions) if num_pred_lesions > 0 else (1.0 if num_gt_lesions == 0 else 0.0)

    return {
        "gt_lesions": num_gt_lesions,
        "pred_lesions": num_pred_lesions,
        "tp_lesions": tp_lesions,
        "fp_lesions": fp_lesions,
        "fn_lesions": fn_lesions,
        "lesion_recall": lesion_recall,
        "lesion_precision": lesion_precision
    }


class MetricTracker:
    """
    Accumulates per-image metrics and computes dataset-level statistics,
    separating positive-only images from all images.
    """
    def __init__(self, lesion_iou_threshold: float = 0.2):
        self.lesion_iou_threshold = lesion_iou_threshold
        self.records = []

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray, image_id: str = ""):
        pix_metrics = compute_pixel_metrics(pred_mask, gt_mask)
        les_metrics = compute_lesion_level_metrics(pred_mask, gt_mask, iou_threshold=self.lesion_iou_threshold)
        record = {"image_id": image_id, **pix_metrics, **les_metrics}
        self.records.append(record)

    def summary(self) -> dict:
        if not self.records:
            return {}

        all_records = self.records
        pos_records = [r for r in self.records if r["is_positive"]]

        def _mean(records, key):
            vals = [r[key] for r in records]
            return float(np.mean(vals)) if vals else 0.0

        total_images = len(all_records)
        total_fp_lesions = sum(r["fp_lesions"] for r in all_records)
        fp_per_image = total_fp_lesions / float(total_images) if total_images > 0 else 0.0

        total_gt_lesions = sum(r["gt_lesions"] for r in all_records)
        total_tp_lesions = sum(r["tp_lesions"] for r in all_records)
        overall_lesion_recall = total_tp_lesions / float(total_gt_lesions) if total_gt_lesions > 0 else 1.0

        summary_dict = {
            # All images
            "all_dice": _mean(all_records, "dice"),
            "all_iou": _mean(all_records, "iou"),
            "all_sensitivity": _mean(all_records, "sensitivity"),
            "all_precision": _mean(all_records, "precision"),
            "all_specificity": _mean(all_records, "specificity"),
            "fp_per_image": fp_per_image,
            "overall_lesion_recall": overall_lesion_recall,
            
            # Positive-only images (clinical lesion segmentation)
            "pos_dice": _mean(pos_records, "dice"),
            "pos_iou": _mean(pos_records, "iou"),
            "pos_sensitivity": _mean(pos_records, "sensitivity"),
            "pos_precision": _mean(pos_records, "precision"),
            "pos_lesion_recall": _mean(pos_records, "lesion_recall"),
            
            "num_total_images": total_images,
            "num_pos_images": len(pos_records),
            "num_gt_lesions": total_gt_lesions,
            "num_tp_lesions": total_tp_lesions
        }
        return summary_dict

