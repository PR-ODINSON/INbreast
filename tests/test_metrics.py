"""Unit tests for src/metrics.py.

Tests:
  - Perfect prediction → Dice=1, IoU=1, HD95=0
  - Both empty → Dice=1, distances=NaN
  - Empty GT, non-empty pred → Dice=0, distances=NaN
  - Shifted mask → HD95 near shift distance
  - Pixel-to-mm scaling (HD95 proportional to spacing)
  - remove_small_components: small removed, large kept
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metrics import (
    boundary_distances,
    compute_pixel_metrics,
    hausdorff95_mm,
    remove_small_components,
)


def test_perfect_prediction():
    """Perfect match → Dice=1, IoU=1, HD95=0."""
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[20:40, 20:40] = 1
    pred = gt.copy()
    px = compute_pixel_metrics(pred, gt)
    assert abs(px["dice"] - 1.0) < 1e-6, f"dice={px['dice']}"
    assert abs(px["iou"] - 1.0) < 1e-6,  f"iou={px['iou']}"
    dist = boundary_distances(pred, gt, (1.0, 1.0))
    assert dist["hd95_mm"] < 1e-3, f"hd95={dist['hd95_mm']}"
    print("[PASS] test_perfect_prediction")


def test_empty_both():
    """Both masks empty → Dice=1, IoU=1, distances=NaN."""
    gt   = np.zeros((64, 64), dtype=np.uint8)
    pred = np.zeros((64, 64), dtype=np.uint8)
    px = compute_pixel_metrics(pred, gt)
    assert abs(px["dice"] - 1.0) < 1e-6, f"dice={px['dice']}"
    dist = boundary_distances(pred, gt, (1.0, 1.0))
    assert np.isnan(dist["hd95_mm"]), f"hd95={dist['hd95_mm']}"
    print("[PASS] test_empty_both")


def test_empty_gt_nonempty_pred():
    """Empty GT, non-empty pred → Dice=0, distances=NaN (FP image)."""
    gt   = np.zeros((64, 64), dtype=np.uint8)
    pred = np.zeros((64, 64), dtype=np.uint8)
    pred[10:20, 10:20] = 1
    px = compute_pixel_metrics(pred, gt)
    assert abs(px["dice"] - 0.0) < 1e-6, f"dice={px['dice']}"
    dist = boundary_distances(pred, gt, (1.0, 1.0))
    assert np.isnan(dist["hd95_mm"]), f"hd95={dist['hd95_mm']}"
    print("[PASS] test_empty_gt_nonempty_pred")


def test_nonempty_gt_empty_pred():
    """Non-empty GT, empty pred → Dice=0."""
    gt   = np.zeros((64, 64), dtype=np.uint8)
    gt[20:40, 20:40] = 1
    pred = np.zeros((64, 64), dtype=np.uint8)
    px = compute_pixel_metrics(pred, gt)
    assert abs(px["dice"] - 0.0) < 1e-6, f"dice={px['dice']}"
    print("[PASS] test_nonempty_gt_empty_pred")


def test_shifted_hd95():
    """Mask shifted 10 rows → HD95 ≈ 10 mm at spacing 1 mm/px."""
    gt   = np.zeros((128, 128), dtype=np.uint8)
    pred = np.zeros((128, 128), dtype=np.uint8)
    gt[50:70, 50:70]   = 1
    pred[60:80, 50:70] = 1   # shifted 10 rows
    dist = boundary_distances(pred, gt, (1.0, 1.0))
    assert 8.0 < dist["hd95_mm"] < 12.0, f"hd95={dist['hd95_mm']} (expected ≈10)"
    print(f"[PASS] test_shifted_hd95  (hd95={dist['hd95_mm']:.2f} mm)")


def test_px_to_mm_conversion():
    """HD95 scales linearly with pixel spacing."""
    gt   = np.zeros((128, 128), dtype=np.uint8)
    pred = np.zeros((128, 128), dtype=np.uint8)
    gt[50:70, 50:70]   = 1
    pred[60:80, 50:70] = 1
    d1   = boundary_distances(pred, gt, (1.0,  1.0))["hd95_mm"]
    d07  = boundary_distances(pred, gt, (0.07, 0.07))["hd95_mm"]
    ratio = d07 / d1
    assert abs(ratio - 0.07) < 0.01, f"ratio={ratio:.4f} expected ≈0.07"
    print(f"[PASS] test_px_to_mm_conversion  (ratio={ratio:.4f})")


def test_remove_small_components():
    """Small components (< min_size) removed; large ones kept."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[5:7, 5:7] = 1       # 4 pixels — small
    mask[20:40, 20:40] = 1   # 400 pixels — large
    cleaned = remove_small_components(mask, min_size=10)
    assert cleaned[5, 5] == 0,  "Small component should be removed"
    assert cleaned[30, 30] == 1, "Large component should remain"
    print("[PASS] test_remove_small_components")


def test_remove_small_components_zero():
    """min_size=0 keeps everything."""
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:7, 5:7] = 1
    cleaned = remove_small_components(mask, min_size=0)
    assert cleaned[5, 5] == 1
    print("[PASS] test_remove_small_components_zero")


if __name__ == "__main__":
    tests = [
        test_perfect_prediction,
        test_empty_both,
        test_empty_gt_nonempty_pred,
        test_nonempty_gt_empty_pred,
        test_shifted_hd95,
        test_px_to_mm_conversion,
        test_remove_small_components,
        test_remove_small_components_zero,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"[FAIL] {t.__name__}: {exc}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    if failed:
        raise SystemExit(1)

