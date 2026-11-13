import glob
import json
import logging
import os
import plistlib
import re

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:
    A = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pydicom
except ImportError:
    pydicom = None

from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Pectoral muscle is never a target, even if a caller passes that string.
ROI_NAMES = {
    "mass": {"mass"},
    "mass_spiculated": {"mass", "spiculated region", "espiculated region"},
    "calcification": {"calcification", "calcifications", "cluster"},
}


def roi_names_for_target(target: str) -> set:
    key = (target or "mass").strip().lower()
    if key not in ROI_NAMES:
        raise ValueError(f"Unknown target '{target}'. Choose mass, mass_spiculated, or calcification.")
    return set(ROI_NAMES[key])


def _require_cv2():
    if cv2 is None:
        raise ImportError("opencv-python is required for mask rasterization.")
    return cv2


def parse_dicom_name(filename: str) -> dict:
    """INbreast CSV patient ids are redacted, so the id is the second filename token."""
    stem = os.path.basename(filename).replace(".dcm", "")
    parts = stem.split("_")
    return {
        "image_id": parts[0],
        "patient_id": parts[1] if len(parts) > 1 else parts[0],
        "laterality": parts[3] if len(parts) > 3 else None,
        "view": parts[4] if len(parts) > 4 else None,
    }


def spacing_from_dataset(dcm, fallback_mm: float = 0.070, source_name: str = "") -> tuple:
    """Return (row_mm, col_mm, source). Falls back to 0.070 mm and logs a warning."""

    def _pair(value):
        if value is None:
            return None
        try:
            if len(value) < 2:
                return None
            row, col = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
        if row <= 0 or col <= 0:
            return None
        return row, col

    pixel = _pair(getattr(dcm, "PixelSpacing", None))
    if pixel is not None:
        return pixel[0], pixel[1], "PixelSpacing"
    imager = _pair(getattr(dcm, "ImagerPixelSpacing", None))
    if imager is not None:
        return imager[0], imager[1], "ImagerPixelSpacing"
    logger.warning(
        "DICOM %s has no PixelSpacing or ImagerPixelSpacing; using %.3f mm.",
        source_name or "unknown",
        fallback_mm,
    )
    return float(fallback_mm), float(fallback_mm), "fallback_0.070"


def read_spacing_mm(dicom_path: str, fallback_mm: float = 0.070) -> tuple:
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM spacing.")
    dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    return spacing_from_dataset(dcm, fallback_mm=fallback_mm, source_name=os.path.basename(dicom_path))


def mm_per_pixel(spacing_row_mm: float, spacing_col_mm: float, resize_scale: float) -> tuple:
    scale = float(resize_scale) if resize_scale else 1.0
    return float(spacing_row_mm) / scale, float(spacing_col_mm) / scale


def read_dicom_image(dicom_path: str, percentile_clip=(0.5, 99.5)) -> np.ndarray:
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files.")
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        img = img * slope + intercept
    photo = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).strip()
    if photo == "MONOCHROME1":
        img = img.max() - img
    p_low, p_high = np.percentile(img, percentile_clip)
    img = np.clip(img, p_low, p_high)
    denom = float(img.max() - img.min())
    if denom > 1e-6:
        img = (img - img.min()) / denom
    else:
        img = np.zeros_like(img)
    return img.astype(np.float32)


def _points_from_roi(roi) -> list:
    polygon = []
    for pt in roi.get("Point_px", []):
        match = re.match(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", str(pt))
        if match:
            polygon.append([float(match.group(1)), float(match.group(2))])
    return polygon


def _paint_polygon(mask: np.ndarray, polygon: list, xml_path: str, name: str) -> None:
    cv = _require_cv2()
    height, width = mask.shape
    if len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv.fillPoly(mask, [pts], 1)
        return
    if len(polygon) in (1, 2):
        logger.debug("Degenerate ROI in %s for '%s' (%d points).", xml_path, name, len(polygon))
        for x, y in polygon:
            px, py = int(round(x)), int(round(y))
            if 0 <= py < height and 0 <= px < width:
                cv.circle(mask, (px, py), radius=2, color=1, thickness=-1)


def parse_xml_rois(xml_path: str, target: str = "mass", image_shape: tuple = None, exclude_pectoral: bool = False, pectoral_xml_path: str = None) -> np.ndarray:
    """Rasterize the target ROIs. 'mass' is Mass only. Pectoral muscle is never painted."""
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if not os.path.exists(xml_path):
        return mask
    valid = roi_names_for_target(target)
    with open(xml_path, "rb") as handle:
        try:
            plist = plistlib.load(handle)
        except Exception as exc:
            logger.warning("Failed to load XML %s: %s", xml_path, exc)
            return mask
    for image in plist.get("Images", []):
        for roi in image.get("ROIs", []):
            name = str(roi.get("Name", "")).strip().lower()
            if "pectoral" in name or name not in valid:
                continue
            _paint_polygon(mask, _points_from_roi(roi), xml_path, name)
    if exclude_pectoral and pectoral_xml_path and os.path.exists(pectoral_xml_path):
        pec = np.zeros((height, width), dtype=np.uint8)
        with open(pectoral_xml_path, "rb") as handle:
            pec_plist = plistlib.load(handle)
        for image in pec_plist.get("Images", []):
            for roi in image.get("ROIs", []):
                _paint_polygon(pec, _points_from_roi(roi), pectoral_xml_path, "pectoral")
        mask[pec > 0] = 0
    return mask


def xml_has_target(xml_path: str, target: str) -> bool:
    if not os.path.exists(xml_path):
        return False
    valid = roi_names_for_target(target)
    try:
        with open(xml_path, "rb") as handle:
            plist = plistlib.load(handle)
    except Exception:
        return False
    for image in plist.get("Images", []):
        for roi in image.get("ROIs", []):
            name = str(roi.get("Name", "")).strip().lower()
            if "pectoral" in name or name not in valid:
                continue
            if len(_points_from_roi(roi)) >= 1:
                return True
    return False


def get_breast_bounding_box(image: np.ndarray, threshold: float = 0.05, margin: int = 10) -> tuple:
    cv = _require_cv2()
    binary = (image > threshold).astype(np.uint8)
    n_labels, _, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return 0, image.shape[0], 0, image.shape[1]
    largest = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    x = int(stats[largest, cv.CC_STAT_LEFT])
    y = int(stats[largest, cv.CC_STAT_TOP])
    w = int(stats[largest, cv.CC_STAT_WIDTH])
    h = int(stats[largest, cv.CC_STAT_HEIGHT])
    top = max(0, y - margin)
    bottom = min(image.shape[0], y + h + margin)
    left = max(0, x - margin)
    right = min(image.shape[1], x + w + margin)
    return top, bottom, left, right


def resize_and_pad(image: np.ndarray, mask: np.ndarray, target_size: int = 512) -> tuple:
    cv = _require_cv2()
    height, width = image.shape[:2]
    scale = min(target_size / float(height), target_size / float(width))
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    resized_img = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    resized_mask = cv.resize(mask, (new_w, new_h), interpolation=cv.INTER_NEAREST)
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    padded_img = cv.copyMakeBorder(resized_img, pad_top, pad_bottom, pad_left, pad_right, cv.BORDER_CONSTANT, value=0.0)
    padded_mask = cv.copyMakeBorder(resized_mask, pad_top, pad_bottom, pad_left, pad_right, cv.BORDER_CONSTANT, value=0)
    return padded_img, padded_mask, float(scale)


def detect_and_flip_breast(image: np.ndarray, mask: np.ndarray, laterality: str = None) -> tuple:
    if laterality == "R":
        should_flip = True
    elif laterality == "L":
        should_flip = False
    else:
        mid = image.shape[1] // 2
        should_flip = image[:, mid:].sum() > image[:, :mid].sum()
    if should_flip:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
    return image, mask, should_flip


def preprocess_sample(dicom_path: str, xml_path: str, config: dict, pectoral_xml_path: str = None, laterality: str = None) -> tuple:
    """Return image, mask, and geometry. mm/pixel is DICOM spacing divided by the resize scale."""
    data_cfg = config.get("data", {})
    clip_range = data_cfg.get("percentile_clip", (0.5, 99.5))
    target_type = data_cfg.get("target", "mass")
    image_size = int(data_cfg.get("image_size", 512))
    flip_right = data_cfg.get("flip_right_breasts", True)
    exclude_pec = data_cfg.get("exclude_pectoral_muscle", False)
    fallback = float(data_cfg.get("fallback_spacing_mm", 0.070))

    img = read_dicom_image(dicom_path, percentile_clip=clip_range)
    spacing_row, spacing_col, spacing_source = read_spacing_mm(dicom_path, fallback_mm=fallback)
    mask = parse_xml_rois(
        xml_path=xml_path,
        target=target_type,
        image_shape=img.shape,
        exclude_pectoral=exclude_pec,
        pectoral_xml_path=pectoral_xml_path,
    )
    top, bottom, left, right = get_breast_bounding_box(img)
    img_cropped = img[top:bottom, left:right]
    mask_cropped = mask[top:bottom, left:right]
    if flip_right:
        img_cropped, mask_cropped, _ = detect_and_flip_breast(img_cropped, mask_cropped, laterality=laterality)
    final_img, final_mask, scale = resize_and_pad(img_cropped, mask_cropped, target_size=image_size)
    mm_row, mm_col = mm_per_pixel(spacing_row, spacing_col, scale)
    meta = {
        "spacing_row_mm": float(spacing_row),
        "spacing_col_mm": float(spacing_col),
        "spacing_source": spacing_source,
        "crop_top": int(top),
        "crop_bottom": int(bottom),
        "crop_left": int(left),
        "crop_right": int(right),
        "resize_scale": float(scale),
        "mm_per_px_row": float(mm_row),
        "mm_per_px_col": float(mm_col),
        "target": str(target_type),
    }
    return final_img, final_mask, meta


def get_augmentations(is_train: bool = True, config: dict = None):
    if A is None or cv2 is None:
        return lambda **kwargs: kwargs
    if not is_train:
        return A.Compose([])
    aug_cfg = config.get("augmentation", {}) if config else {}
    return A.Compose(
        [
            A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_prob", 0.5)),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.08,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=aug_cfg.get("shift_scale_rotate_prob", 0.5),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=aug_cfg.get("brightness_contrast_prob", 0.5),
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=aug_cfg.get("gamma_prob", 0.3)),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=aug_cfg.get("elastic_distortion_prob", 0.2),
            ),
        ]
    )


def _cache_is_current(path: str, target: str) -> bool:
    if not os.path.exists(path):
        return False
    with np.load(path) as data:
        required = {"image", "mask", "mm_per_px_row", "mm_per_px_col", "resize_scale", "target"}
        if not required.issubset(set(data.files)):
            return False
        return str(data["target"]) == str(target)


class INbreastDataset(Dataset):
    def __init__(self, records: list, config: dict, is_train: bool = False):
        self.records = list(records)
        self.config = config
        self.is_train = is_train
        self.transforms = get_augmentations(is_train=is_train, config=config)
        self.use_cache = config.get("data", {}).get("use_cache", True)
        self.cache_dir = config.get("data", {}).get("cache_dir", "data/cache")
        self.target = config.get("data", {}).get("target", "mass")

    def __len__(self):
        return len(self.records)

    def _load_arrays(self, rec):
        cache_path = os.path.join(self.cache_dir, f"{rec['image_id']}.npz")
        if self.use_cache and _cache_is_current(cache_path, self.target):
            data = np.load(cache_path)
            meta = {
                "mm_per_px_row": float(data["mm_per_px_row"]),
                "mm_per_px_col": float(data["mm_per_px_col"]),
                "resize_scale": float(data["resize_scale"]),
                "spacing_source": str(data["spacing_source"]) if "spacing_source" in data.files else "",
            }
            return data["image"], data["mask"], meta
        img, mask, meta = preprocess_sample(
            dicom_path=rec["dicom_path"],
            xml_path=rec["xml_path"],
            config=self.config,
            pectoral_xml_path=rec.get("pectoral_xml_path"),
            laterality=rec.get("laterality"),
        )
        return img, mask, meta

    def __getitem__(self, idx):
        rec = self.records[idx]
        img, mask, meta = self._load_arrays(rec)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
        img_3ch = np.stack([img, img, img], axis=-1).astype(np.float32)
        if self.config.get("data", {}).get("normalize_imagenet", True):
            img_3ch = (img_3ch - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(img_3ch.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float(),
            "image_id": rec["image_id"],
            "patient_id": rec["patient_id"],
            "is_positive": bool(np.asarray(mask).sum() > 0),
            "mm_per_px_row": float(meta["mm_per_px_row"]),
            "mm_per_px_col": float(meta["mm_per_px_col"]),
            "resize_scale": float(meta["resize_scale"]),
        }


def collect_records(config: dict) -> list:
    data_cfg = config.get("data", {})
    dicom_dir = data_cfg.get("dicom_dir", "INbreast_complete/AllDICOMs")
    xml_dir = data_cfg.get("xml_dir", "INbreast_complete/AllXML")
    pectoral_dir = data_cfg.get("pectoral_xml_dir", "INbreast_complete/PectoralMuscle/Pectoral Muscle XML")
    target = data_cfg.get("target", "mass")
    records = []
    for dpath in sorted(glob.glob(os.path.join(dicom_dir, "*.dcm"))):
        parsed = parse_dicom_name(dpath)
        xml_path = os.path.join(xml_dir, f"{parsed['image_id']}.xml")
        pec_path = os.path.join(pectoral_dir, f"{parsed['image_id']}_muscle.xml")
        records.append(
            {
                "image_id": parsed["image_id"],
                "patient_id": parsed["patient_id"],
                "laterality": parsed["laterality"],
                "view": parsed["view"],
                "dicom_path": dpath,
                "xml_path": xml_path,
                "pectoral_xml_path": pec_path if os.path.exists(pec_path) else None,
                "is_positive": bool(xml_has_target(xml_path, target)),
            }
        )
    return records


def assign_outer_folds(patient_ids: list, labels: list, n_splits: int = 5, seed: int = 42) -> dict:
    """Outer fold k uses test=fold k, val=fold (k+1)%n, train=the other folds.

    Patients are grouped (one row per patient) and stratified by the binary label.
    """
    patient_ids = [str(p) for p in patient_ids]
    labels = np.asarray(labels)
    if len(patient_ids) != len(labels):
        raise ValueError("patient_ids and labels must have the same length.")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("patient_ids must be unique.")
    if len(patient_ids) < n_splits:
        raise ValueError(f"Need at least {n_splits} patients, got {len(patient_ids)}.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = np.asarray(patient_ids)
    fold_of = {}
    for fold_idx, (_, test_idx) in enumerate(splitter.split(np.zeros(len(patient_ids)), labels, groups)):
        for index in test_idx:
            fold_of[patient_ids[index]] = int(fold_idx)
    if len(fold_of) != len(patient_ids):
        raise RuntimeError("Fold assignment did not cover every patient.")
    nested = {}
    for k in range(n_splits):
        test_ids = [p for p, fold in fold_of.items() if fold == k]
        val_ids = [p for p, fold in fold_of.items() if fold == (k + 1) % n_splits]
        train_ids = [p for p, fold in fold_of.items() if fold not in (k, (k + 1) % n_splits)]
        nested[k] = {"train": train_ids, "val": val_ids, "test": test_ids, "fold_of": fold_of}
    return nested


def resample_training_records(records: list, negative_ratio: float, seed: int, epoch: int) -> list:
    """Subsample negatives for one training epoch. The seed does not depend on the model name.

    Val and test lists must not be passed through this function.
    """
    positives = sorted([r for r in records if r["is_positive"]], key=lambda r: r["image_id"])
    negatives = sorted([r for r in records if not r["is_positive"]], key=lambda r: r["image_id"])
    if negative_ratio is None or float(negative_ratio) < 0:
        return positives + negatives
    n_keep = int(round(len(positives) * float(negative_ratio)))
    n_keep = max(0, min(n_keep, len(negatives)))
    rng = np.random.RandomState((int(seed) + int(epoch) * 100003) % (2**31 - 1))
    if n_keep == len(negatives):
        chosen = negatives
    elif n_keep == 0:
        chosen = []
    else:
        pick = sorted(rng.choice(len(negatives), size=n_keep, replace=False).tolist())
        chosen = [negatives[i] for i in pick]
    return positives + chosen


def _samples_for(patient_ids, patient_records) -> list:
    chosen = set(patient_ids)
    samples = [rec for pid in patient_ids for rec in patient_records[pid] if rec["patient_id"] in chosen]
    return sorted(samples, key=lambda r: r["image_id"])


def build_patient_stratified_splits(config: dict, output_path: str = "splits.json") -> dict:
    """Write nested patient-grouped folds. Training negatives are not dropped here."""
    data_cfg = config.get("data", {})
    target = data_cfg.get("target", "mass")
    n_splits = int(data_cfg.get("n_splits", 5))
    seed = int(data_cfg.get("seed", 42))
    records = collect_records(config)
    if not records:
        raise RuntimeError(f"No DICOM files found under {data_cfg.get('dicom_dir')}.")

    patient_records = {}
    patient_positive = {}
    for rec in records:
        patient_records.setdefault(rec["patient_id"], []).append(rec)
        patient_positive[rec["patient_id"]] = patient_positive.get(rec["patient_id"], False) or bool(rec["is_positive"])
    patients = sorted(patient_records)
    labels = [1 if patient_positive[p] else 0 for p in patients]
    nested = assign_outer_folds(patients, labels, n_splits=n_splits, seed=seed)

    folds = {}
    for k, parts in nested.items():
        train_p, val_p, test_p = set(parts["train"]), set(parts["val"]), set(parts["test"])
        if train_p & val_p or train_p & test_p or val_p & test_p:
            raise AssertionError(f"Patient overlap in outer fold {k}.")
        folds[f"fold_{k}"] = {
            "train_patients": sorted(train_p),
            "val_patients": sorted(val_p),
            "test_patients": sorted(test_p),
            "train_samples": _samples_for(sorted(train_p), patient_records),
            "val_samples": _samples_for(sorted(val_p), patient_records),
            "test_samples": _samples_for(sorted(test_p), patient_records),
        }
    payload = {
        "scheme": "nested_patient_grouped_5fold",
        "n_splits": n_splits,
        "seed": seed,
        "target": target,
        "negative_sampling": "train_epoch_only",
        "n_patients": len(patients),
        "n_positive_patients": int(sum(labels)),
        "n_images": len(records),
        "n_positive_images": int(sum(r["is_positive"] for r in records)),
        "folds": folds,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info(
        "Wrote %s (%d patients, %d images, %d positive images, target=%s).",
        output_path,
        len(patients),
        len(records),
        payload["n_positive_images"],
        target,
    )
    return payload


def limit_records(records: list, limit: int) -> list:
    if limit is None or int(limit) <= 0:
        return list(records)
    return sorted(records, key=lambda r: r["image_id"])[: int(limit)]


def precompute_and_cache_dataset(config: dict) -> None:
    data_cfg = config.get("data", {})
    cache_dir = data_cfg.get("cache_dir", "data/cache")
    target = data_cfg.get("target", "mass")
    os.makedirs(cache_dir, exist_ok=True)
    records = collect_records(config)
    logger.info("Caching %d images for target=%s into %s", len(records), target, cache_dir)
    for rec in records:
        out_file = os.path.join(cache_dir, f"{rec['image_id']}.npz")
        if _cache_is_current(out_file, target):
            continue
        img, mask, meta = preprocess_sample(
            dicom_path=rec["dicom_path"],
            xml_path=rec["xml_path"],
            config=config,
            pectoral_xml_path=rec.get("pectoral_xml_path"),
            laterality=rec.get("laterality"),
        )
        np.savez_compressed(
            out_file,
            image=img,
            mask=mask,
            mm_per_px_row=np.float32(meta["mm_per_px_row"]),
            mm_per_px_col=np.float32(meta["mm_per_px_col"]),
            resize_scale=np.float32(meta["resize_scale"]),
            spacing_row_mm=np.float32(meta["spacing_row_mm"]),
            spacing_col_mm=np.float32(meta["spacing_col_mm"]),
            spacing_source=np.array(meta["spacing_source"]),
            crop_top=np.int32(meta["crop_top"]),
            crop_bottom=np.int32(meta["crop_bottom"]),
            crop_left=np.int32(meta["crop_left"]),
            crop_right=np.int32(meta["crop_right"]),
            target=np.array(meta["target"]),
        )
    logger.info("Cache complete: %s", cache_dir)


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Regenerate nested patient splits")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    build_patient_stratified_splits(cfg, output_path=args.out or cfg["data"].get("splits_path", "splits.json"))
