"""Test-set metrics. Distances are millimetres. Empty masks follow one shared rule."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, label


PER_IMAGE_COLUMNS = [
    "model",
    "fold",
    "image_id",
    "patient_id",
    "is_positive",
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "hd95_mm",
    "hd100_mm",
    "assd_mm",
    "nsd_1mm",
    "nsd_2mm",
    "penalized_hd95_mm",
    "mm_per_px_row",
    "mm_per_px_col",
    "gt_lesions",
    "pred_lesions",
    "tp_lesions_02",
    "fp_lesions_02",
    "fn_lesions_02",
    "lesion_recall_02",
    "lesion_precision_02",
    "lesion_f1_02",
    "tp_lesions_05",
    "fp_lesions_05",
    "fn_lesions_05",
    "lesion_recall_05",
    "lesion_precision_05",
    "lesion_f1_05",
    "max_prob",
    "threshold",
    "min_component_size",
    "inference_ms",
]


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> dict:
    pred = pred_mask > 0
    gt = gt_mask > 0
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    gt_sum = int(gt.sum())
    pred_sum = int(pred.sum())

    if gt_sum == 0 and pred_sum == 0:
        dice, iou, precision, recall = 1.0, 1.0, 1.0, 1.0
    elif gt_sum == 0 and pred_sum > 0:
        dice, iou, precision, recall = 0.0, 0.0, 0.0, 1.0
    elif gt_sum > 0 and pred_sum == 0:
        dice, iou, precision, recall = 0.0, 0.0, 0.0, 0.0
    else:
        dice = float((2.0 * tp) / (pred_sum + gt_sum + eps))
        iou = float(tp / (tp + fp + fn + eps))
        precision = float(tp / (tp + fp + eps))
        recall = float(tp / (tp + fn + eps))
    specificity = float(tn / (tn + fp + eps)) if (tn + fp) > 0 else 1.0
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "is_positive": gt_sum > 0,
    }


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if min_size is None or int(min_size) <= 0 or binary.sum() == 0:
        return binary
    labeled, n = label(binary)
    if n == 0:
        return binary
    keep = np.zeros(n + 1, dtype=bool)
    counts = np.bincount(labeled.ravel())
    keep[counts >= int(min_size)] = True
    keep[0] = False
    return keep[labeled].astype(np.uint8)


def _surfaces(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    if not binary.any():
        return binary
    footprint = generate_binary_structure(2, 1)
    eroded = binary_erosion(binary, structure=footprint, iterations=1, border_value=0)
    return np.logical_xor(binary, eroded)


def _directed_distances(source_surface: np.ndarray, target_surface: np.ndarray, spacing) -> np.ndarray:
    if source_surface.sum() == 0 or target_surface.sum() == 0:
        return np.array([], dtype=np.float64)
    distance = distance_transform_edt(~target_surface, sampling=spacing)
    return distance[source_surface].astype(np.float64)


def boundary_distances(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing_mm, tolerances_mm=(1.0, 2.0)) -> dict:
    """HD95, HD100, ASSD, and normalized surface Dice. spacing_mm is (row, col)."""
    spacing = (float(spacing_mm[0]), float(spacing_mm[1]))
    pred_s = _surfaces(pred_mask)
    gt_s = _surfaces(gt_mask)
    d_pred = _directed_distances(pred_s, gt_s, spacing)
    d_gt = _directed_distances(gt_s, pred_s, spacing)
    empty = {
        "hd95_mm": np.nan,
        "hd100_mm": np.nan,
        "assd_mm": np.nan,
        "nsd_1mm": np.nan,
        "nsd_2mm": np.nan,
    }
    if d_pred.size == 0 or d_gt.size == 0:
        return empty
    hd95 = float(max(np.percentile(d_pred, 95), np.percentile(d_gt, 95)))
    hd100 = float(max(d_pred.max(), d_gt.max()))
    assd = float(np.concatenate([d_pred, d_gt]).mean())
    nsd = {}
    total = float(d_pred.size + d_gt.size)
    for tau in tolerances_mm:
        key = f"nsd_{int(tau)}mm" if float(tau) == int(tau) else f"nsd_{tau}mm"
        nsd[key] = float(((d_pred <= tau).sum() + (d_gt <= tau).sum()) / total)
    # The reported columns are the 1 mm and 2 mm tolerances.
    return {
        "hd95_mm": hd95,
        "hd100_mm": hd100,
        "assd_mm": assd,
        "nsd_1mm": nsd.get("nsd_1mm", np.nan),
        "nsd_2mm": nsd.get("nsd_2mm", np.nan),
    }


def image_diagonal_mm(shape, spacing_mm) -> float:
    height, width = shape[:2]
    return float(np.hypot(height * float(spacing_mm[0]), width * float(spacing_mm[1])))


def compute_lesion_level_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, iou_threshold: float = 0.2) -> dict:
    """One-to-one component matching. A hit requires IoU >= iou_threshold."""
    gt_binary = (gt_mask > 0).astype(np.uint8)
    pred_binary = (pred_mask > 0).astype(np.uint8)
    labeled_gt, n_gt = label(gt_binary)
    labeled_pred, n_pred = label(pred_binary)

    if n_gt == 0 and n_pred == 0:
        precision, recall = 1.0, 1.0
        tp = fp = fn = 0
    elif n_gt == 0:
        precision, recall = 0.0, 1.0
        tp, fp, fn = 0, int(n_pred), 0
    elif n_pred == 0:
        precision, recall = 0.0, 0.0
        tp, fp, fn = 0, 0, int(n_gt)
    else:
        pairs = []
        gt_sets = [(labeled_gt == i) for i in range(1, n_gt + 1)]
        pred_sets = [(labeled_pred == i) for i in range(1, n_pred + 1)]
        gt_areas = [int(m.sum()) for m in gt_sets]
        pred_areas = [int(m.sum()) for m in pred_sets]
        for p_idx, p_comp in enumerate(pred_sets):
            for g_idx, g_comp in enumerate(gt_sets):
                inter = int(np.logical_and(p_comp, g_comp).sum())
                if inter <= 0:
                    continue
                union = pred_areas[p_idx] + gt_areas[g_idx] - inter
                pairs.append((inter / float(union), p_idx, g_idx))
        pairs.sort(key=lambda item: item[0], reverse=True)
        matched_p, matched_g = set(), set()
        for iou, p_idx, g_idx in pairs:
            if iou < iou_threshold:
                break
            if p_idx in matched_p or g_idx in matched_g:
                continue
            matched_p.add(p_idx)
            matched_g.add(g_idx)
        tp = len(matched_g)
        fp = int(n_pred) - len(matched_p)
        fn = int(n_gt) - len(matched_g)
        recall = tp / float(n_gt)
        precision = tp / float(n_pred) if n_pred else 0.0

    return {
        "gt_lesions": int(n_gt),
        "pred_lesions": int(n_pred),
        "tp_lesions": int(tp),
        "fp_lesions": int(fp),
        "fn_lesions": int(fn),
        "lesion_recall": float(recall),
        "lesion_precision": float(precision),
        "lesion_f1": _f1(float(precision), float(recall)),
    }


def compute_case_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing_mm, tolerances_mm=(1.0, 2.0)) -> dict:
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    pixel = compute_pixel_metrics(pred, gt)
    gt_sum = int(gt.sum())
    pred_sum = int(pred.sum())
    distances = {
        "hd95_mm": np.nan,
        "hd100_mm": np.nan,
        "assd_mm": np.nan,
        "nsd_1mm": np.nan,
        "nsd_2mm": np.nan,
    }
    if gt_sum > 0 and pred_sum > 0:
        distances = boundary_distances(pred, gt, spacing_mm, tolerances_mm=tolerances_mm)
    penalized = distances["hd95_mm"]
    if gt_sum > 0 and pred_sum == 0:
        penalized = image_diagonal_mm(gt.shape, spacing_mm)
    les02 = compute_lesion_level_metrics(pred, gt, iou_threshold=0.2)
    les05 = compute_lesion_level_metrics(pred, gt, iou_threshold=0.5)
    row = {
        "dice": pixel["dice"],
        "iou": pixel["iou"],
        "precision": pixel["precision"],
        "recall": pixel["recall"],
        "specificity": pixel["specificity"],
        "is_positive": pixel["is_positive"],
        **distances,
        "penalized_hd95_mm": penalized,
        "mm_per_px_row": float(spacing_mm[0]),
        "mm_per_px_col": float(spacing_mm[1]),
        "gt_lesions": les02["gt_lesions"],
        "pred_lesions": les02["pred_lesions"],
        "tp_lesions_02": les02["tp_lesions"],
        "fp_lesions_02": les02["fp_lesions"],
        "fn_lesions_02": les02["fn_lesions"],
        "lesion_recall_02": les02["lesion_recall"],
        "lesion_precision_02": les02["lesion_precision"],
        "lesion_f1_02": les02["lesion_f1"],
        "tp_lesions_05": les05["tp_lesions"],
        "fp_lesions_05": les05["fp_lesions"],
        "fn_lesions_05": les05["fn_lesions"],
        "lesion_recall_05": les05["lesion_recall"],
        "lesion_precision_05": les05["lesion_precision"],
        "lesion_f1_05": les05["lesion_f1"],
    }
    return row


def hausdorff95_mm(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing_mm=(1.0, 1.0)) -> float:
    """Symmetric 95th-percentile Hausdorff distance. NaN if either mask is empty."""
    if int((pred_mask > 0).sum()) == 0 or int((gt_mask > 0).sum()) == 0:
        return float("nan")
    return float(boundary_distances(pred_mask, gt_mask, spacing_mm)["hd95_mm"])


def sensitivity_at_fppi(fppi: np.ndarray, sensitivity: np.ndarray, target: float) -> float:
    """Linear interpolation. NaN when the curve never reaches `target` FP per image."""
    fppi = np.asarray(fppi, dtype=np.float64)
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    ok = np.isfinite(fppi) & np.isfinite(sensitivity)
    fppi, sensitivity = fppi[ok], sensitivity[ok]
    if fppi.size == 0 or float(np.max(fppi)) < target:
        return float("nan")
    order = np.argsort(fppi)
    x = fppi[order]
    y = sensitivity[order]
    # If several thresholds share an FPPI, keep the highest sensitivity there.
    uniq_x = []
    uniq_y = []
    for xi, yi in zip(x, y):
        if uniq_x and abs(xi - uniq_x[-1]) < 1e-12:
            uniq_y[-1] = max(uniq_y[-1], yi)
        else:
            uniq_x.append(xi)
            uniq_y.append(yi)
    return float(np.interp(target, np.asarray(uniq_x), np.asarray(uniq_y)))


def froc_operating_points(records_by_threshold: list) -> dict:
    """records_by_threshold items have tp, fp, fp_pos, n_gt, n_images, n_pos."""
    fppi, sens, fppi_pos, sens_pos = [], [], [], []
    for row in records_by_threshold:
        n_gt = row["n_gt"]
        recall = float(row["tp"] / n_gt) if n_gt else float("nan")
        fppi.append(float(row["fp"] / row["n_images"]) if row["n_images"] else float("nan"))
        sens.append(recall)
        fppi_pos.append(float(row["fp_pos"] / row["n_pos"]) if row["n_pos"] else float("nan"))
        sens_pos.append(recall)
    return {
        "sens_at_1_fppi": sensitivity_at_fppi(fppi, sens, 1.0),
        "sens_at_2_fppi": sensitivity_at_fppi(fppi, sens, 2.0),
        "sens_at_1_fppi_pos": sensitivity_at_fppi(fppi_pos, sens_pos, 1.0),
        "sens_at_2_fppi_pos": sensitivity_at_fppi(fppi_pos, sens_pos, 2.0),
    }


def image_level_auc(is_positive, max_prob) -> float:
    y = np.asarray(is_positive, dtype=np.int32)
    s = np.asarray(max_prob, dtype=np.float64)
    if y.min() == y.max():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


class MetricTracker:
    """Pixel metrics used for the training-time validation curve (threshold 0.5)."""

    def __init__(self, lesion_iou_threshold: float = 0.2):
        self.lesion_iou_threshold = lesion_iou_threshold
        self.records = []

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray, image_id: str = ""):
        pixel = compute_pixel_metrics(pred_mask, gt_mask)
        lesion = compute_lesion_level_metrics(pred_mask, gt_mask, iou_threshold=self.lesion_iou_threshold)
        self.records.append({"image_id": image_id, **pixel, **lesion})

    def summary(self) -> dict:
        if not self.records:
            return {}
        all_records = self.records
        pos_records = [r for r in all_records if r["is_positive"]]

        def _mean(records, key):
            if not records:
                return 0.0
            return float(np.mean([r[key] for r in records]))

        n = len(all_records)
        fp_lesions = sum(r["fp_lesions"] for r in all_records)
        gt_lesions = sum(r["gt_lesions"] for r in all_records)
        tp_lesions = sum(r["tp_lesions"] for r in all_records)
        return {
            "all_dice": _mean(all_records, "dice"),
            "all_iou": _mean(all_records, "iou"),
            "all_sensitivity": _mean(all_records, "recall"),
            "all_precision": _mean(all_records, "precision"),
            "all_specificity": _mean(all_records, "specificity"),
            "fp_per_image": fp_lesions / float(n) if n else 0.0,
            "overall_lesion_recall": (tp_lesions / float(gt_lesions)) if gt_lesions else 1.0,
            "pos_dice": _mean(pos_records, "dice"),
            "pos_iou": _mean(pos_records, "iou"),
            "pos_sensitivity": _mean(pos_records, "recall"),
            "pos_precision": _mean(pos_records, "precision"),
            "pos_lesion_recall": _mean(pos_records, "lesion_recall"),
            "num_total_images": n,
            "num_pos_images": len(pos_records),
            "num_gt_lesions": gt_lesions,
            "num_tp_lesions": tp_lesions,
        }
