import os
import glob
import re
import json
import logging
import plistlib
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None
from sklearn.model_selection import StratifiedGroupKFold

try:
    import pydicom
except ImportError:
    pydicom = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_dicom_image(dicom_path: str, percentile_clip=(0.5, 99.5)) -> np.ndarray:
    """
    Reads DICOM file, handles MONOCHROME1/2 photometric interpretation,
    rescale slope/intercept, percentile clipping, and scales to [0, 1].
    Tissue is guaranteed to be bright.
    """
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files. Please install pydicom.")

    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)

    # Apply rescale slope and intercept if present
    slope = getattr(dcm, "RescaleSlope", 1.0)
    intercept = getattr(dcm, "RescaleIntercept", 0.0)
    if slope != 1.0 or intercept != 0.0:
        img = img * float(slope) + float(intercept)

    # Photometric Interpretation handling:
    # In MONOCHROME1, lower values are brighter, higher values are darker (bone/tissue is low).
    # We invert it so tissue is bright.
    photo = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2").strip()
    if photo == "MONOCHROME1":
        img = img.max() - img

    # Percentile clipping (0.5% - 99.5%)
    p_low, p_high = np.percentile(img, percentile_clip)
    img = np.clip(img, p_low, p_high)

    # Min-max normalization to [0, 1]
    denom = img.max() - img.min()
    if denom > 1e-6:
        img = (img - img.min()) / denom
    else:
        img = np.zeros_like(img)

    return img.astype(np.float32)


def parse_xml_rois(xml_path: str, target: str = "mass", image_shape: tuple = None, exclude_pectoral: bool = False, pectoral_xml_path: str = None) -> np.ndarray:
    """
    Parses OsiriX plist XML file, extracts contours for the given target,
    and rasterizes into a binary mask of shape image_shape (H, W).
    
    Target:
      - 'mass': 'Mass' + 'Spiculated Region' (and variants).
      - 'calcification': 'Calcification' + 'Cluster' + 'Calcifications'.
    Pectoral Muscle ROIs are never part of the lesion mask.
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    if not os.path.exists(xml_path):
        return mask

    with open(xml_path, "rb") as f:
        try:
            pl = plistlib.load(f)
        except Exception as e:
            logger.warning(f"Failed to load XML {xml_path}: {e}")
            return mask

    target = target.lower()
    if target == "mass":
        valid_roi_names = {
            "mass",
            "spiculated region",
            "spiculated region ",
            "espiculated region"
        }
    elif target == "calcification":
        valid_roi_names = {
            "calcification",
            "cluster",
            "calcifications"
        }
    else:
        valid_roi_names = {target}

    images = pl.get("Images", [])
    for img in images:
        rois = img.get("ROIs", [])
        for roi in rois:
            name = str(roi.get("Name", "")).strip().lower()
            if name in valid_roi_names:
                pts_str = roi.get("Point_px", [])
                polygon = []
                for pt in pts_str:
                    m = re.match(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", pt)
                    if m:
                        x = float(m.group(1))
                        y = float(m.group(2))
                        polygon.append([x, y])
                
                if len(polygon) >= 3:
                    pts_arr = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts_arr], 1)
                elif len(polygon) in (1, 2):
                    logger.debug(f"Degenerate ROI (<3 points) in {xml_path} for '{name}' with {len(polygon)} points. Drawing point/line.")
                    for pt in polygon:
                        px, py = int(round(pt[0])), int(round(pt[1]))
                        if 0 <= py < H and 0 <= px < W:
                            cv2.circle(mask, (px, py), radius=2, color=1, thickness=-1)

    # Optional: exclude pectoral muscle by zeroing out muscle region
    if exclude_pectoral and pectoral_xml_path and os.path.exists(pectoral_xml_path):
        pec_mask = np.zeros((H, W), dtype=np.uint8)
        with open(pectoral_xml_path, "rb") as pf:
            ppl = plistlib.load(pf)
        for pimg in ppl.get("Images", []):
            for proi in pimg.get("ROIs", []):
                ppts_str = proi.get("Point_px", [])
                ppoly = []
                for pt in ppts_str:
                    m = re.match(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", pt)
                    if m:
                        ppoly.append([float(m.group(1)), float(m.group(2))])
                if len(ppoly) >= 3:
                    cv2.fillPoly(pec_mask, [np.array(ppoly, dtype=np.int32).reshape((-1, 1, 2))], 1)
        mask[pec_mask > 0] = 0

    return mask


def get_breast_bounding_box(image: np.ndarray, threshold: float = 0.05, margin: int = 10) -> tuple:
    """
    Finds the breast bounding box by thresholding and finding the largest connected component.
    Returns (top, bottom, left, right).
    """
    binary = (image > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0, image.shape[0], 0, image.shape[1]

    # Find largest non-background component
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x = stats[largest_idx, cv2.CC_STAT_LEFT]
    y = stats[largest_idx, cv2.CC_STAT_TOP]
    w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    h = stats[largest_idx, cv2.CC_STAT_HEIGHT]

    top = max(0, y - margin)
    bottom = min(image.shape[0], y + h + margin)
    left = max(0, x - margin)
    right = min(image.shape[1], x + w + margin)

    return top, bottom, left, right


def resize_and_pad(image: np.ndarray, mask: np.ndarray, target_size: int = 512) -> tuple:
    """
    Aspect-ratio preserving resize followed by symmetric zero-padding to (target_size, target_size).
    Bilinear interpolation for image, nearest-neighbor for mask.
    """
    h, w = image.shape
    scale = min(target_size / h, target_size / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    padded_img = cv2.copyMakeBorder(resized_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0.0)
    padded_mask = cv2.copyMakeBorder(resized_mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    return padded_img, padded_mask


def detect_and_flip_breast(image: np.ndarray, mask: np.ndarray, laterality: str = None) -> tuple:
    """
    Ensures consistent breast orientation. If breast tissue is on the right,
    flips horizontally so breast tissue is consistently on the left.
    """
    if laterality == "R":
        should_flip = True
    elif laterality == "L":
        should_flip = False
    else:
        # Fallback heuristic: compare mass of left half vs right half
        mid = image.shape[1] // 2
        left_mass = image[:, :mid].sum()
        right_mass = image[:, mid:].sum()
        should_flip = (right_mass > left_mass)

    if should_flip:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

    return image, mask, should_flip


def preprocess_sample(dicom_path: str, xml_path: str, config: dict, pectoral_xml_path: str = None, laterality: str = None) -> tuple:
    """
    Complete preprocessing pipeline for a single sample:
    1. Read DICOM with photometric & 14-bit normalization.
    2. Parse XML to binary mask.
    3. Crop to breast bounding box.
    4. Optional horizontal flip for orientation consistency.
    5. Resize & pad to target square size (default 512).
    """
    data_cfg = config.get("data", {})
    clip_range = data_cfg.get("percentile_clip", (0.5, 99.5))
    target_type = data_cfg.get("target", "mass")
    image_size = data_cfg.get("image_size", 512)
    flip_right = data_cfg.get("flip_right_breasts", True)
    exclude_pec = data_cfg.get("exclude_pectoral_muscle", False)

    # 1. Read DICOM
    img = read_dicom_image(dicom_path, percentile_clip=clip_range)

    # 2. Parse Mask
    mask = parse_xml_rois(
        xml_path=xml_path,
        target=target_type,
        image_shape=img.shape,
        exclude_pectoral=exclude_pec,
        pectoral_xml_path=pectoral_xml_path
    )

    # 3. Crop to breast bounding box
    top, bottom, left, right = get_breast_bounding_box(img)
    img_cropped = img[top:bottom, left:right]
    mask_cropped = mask[top:bottom, left:right]

    # 4. Consistent flipping
    if flip_right:
        img_cropped, mask_cropped, _ = detect_and_flip_breast(img_cropped, mask_cropped, laterality=laterality)

    # 5. Resize and pad
    final_img, final_mask = resize_and_pad(img_cropped, mask_cropped, target_size=image_size)

    return final_img, final_mask


def get_augmentations(is_train: bool = True, config: dict = None):
    """
    Albumentations pipeline:
    - Train: horizontal flip, small rotation/scale/shift, brightness/contrast, gamma, light elastic distortion.
    - Val/Test: No augmentations, just ImageNet normalization and PyTorch tensor conversion.
    """
    if A is None:
        return lambda **kwargs: kwargs

    if is_train:
        aug_cfg = config.get("augmentation", {}) if config else {}
        return A.Compose([
            A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_prob", 0.5)),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.08,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=aug_cfg.get("shift_scale_rotate_prob", 0.5)
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=aug_cfg.get("brightness_contrast_prob", 0.5)
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=aug_cfg.get("gamma_prob", 0.3)),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=aug_cfg.get("elastic_distortion_prob", 0.2)
            ),
        ])
    else:
        return A.Compose([])


class INbreastDataset(Dataset):
    """
    PyTorch Dataset for INbreast mammograms.
    Supports on-the-fly preprocessing as well as fast loading from pre-cached files.
    Replicates single-channel mammograms to 3 channels with ImageNet normalization.
    """
    def __init__(self, records: list, config: dict, is_train: bool = False):
        self.records = records
        self.config = config
        self.is_train = is_train
        self.transforms = get_augmentations(is_train=is_train, config=config)
        self.use_cache = config.get("data", {}).get("use_cache", True)
        self.cache_dir = config.get("data", {}).get("cache_dir", "data/cache")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image_id = rec["image_id"]

        # Attempt to load from cache
        cache_path = os.path.join(self.cache_dir, f"{image_id}.npz")
        if self.use_cache and os.path.exists(cache_path):
            data = np.load(cache_path)
            img = data["image"]
            mask = data["mask"]
        else:
            img, mask = preprocess_sample(
                dicom_path=rec["dicom_path"],
                xml_path=rec["xml_path"],
                config=self.config,
                pectoral_xml_path=rec.get("pectoral_xml_path"),
                laterality=rec.get("laterality")
            )

        # Apply Albumentations
        augmented = self.transforms(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]

        # Replicate 1 channel to 3 channels for ImageNet pretrained encoder
        img_3ch = np.stack([img, img, img], axis=-1).astype(np.float32)

        # Apply ImageNet normalization: (img - mean) / std
        if self.config.get("data", {}).get("normalize_imagenet", True):
            img_3ch = (img_3ch - IMAGENET_MEAN) / IMAGENET_STD

        # Convert to PyTorch tensors: Image (3, H, W), Mask (1, H, W)
        tensor_img = torch.from_numpy(img_3ch.transpose(2, 0, 1)).float()
        tensor_mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": tensor_img,
            "mask": tensor_mask,
            "image_id": image_id,
            "patient_id": rec["patient_id"],
            "is_positive": bool(mask.sum() > 0)
        }


def build_patient_stratified_splits(config: dict, output_path: str = "splits.json") -> dict:
    """
    Builds 5-fold cross-validation splits grouped strictly by patient_id
    and stratified by whether the patient has any target lesions.
    Guarantees ZERO patient leakage across folds.
    """
    data_cfg = config.get("data", {})
    dicom_dir = data_cfg.get("dicom_dir", "INbreast_complete/AllDICOMs")
    xml_dir = data_cfg.get("xml_dir", "INbreast_complete/AllXML")
    csv_path = data_cfg.get("csv_path", "INbreast_complete/INbreast.csv")
    pectoral_dir = data_cfg.get("pectoral_xml_dir", "INbreast_complete/PectoralMuscle/Pectoral Muscle XML")
    target = data_cfg.get("target", "mass").lower()
    negative_ratio = data_cfg.get("negative_ratio", 1.0)
    n_splits = data_cfg.get("n_splits", 5)
    seed = data_cfg.get("seed", 42)

    # 1. Collect all DICOM files
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    records = []
    patient_positives = {}
    patient_records = {}

    for dpath in dicom_files:
        fname = os.path.basename(dpath)
        parts = fname.replace(".dcm", "").split("_")
        img_id = parts[0]
        patient_id = parts[1] if len(parts) > 1 else "unknown"
        lat = parts[3] if len(parts) > 3 else "unknown"
        view = parts[4] if len(parts) > 4 else "unknown"
        xml_path = os.path.join(xml_dir, f"{img_id}.xml")
        pec_path = os.path.join(pectoral_dir, f"{img_id}_muscle.xml")

        # Check if target lesion exists in XML
        has_target = False
        if os.path.exists(xml_path):
            with open(xml_path, "rb") as xf:
                try:
                    pl = plistlib.load(xf)
                    for img in pl.get("Images", []):
                        for roi in img.get("ROIs", []):
                            name = str(roi.get("Name", "")).strip().lower()
                            if target == "mass" and name in ["mass", "spiculated region", "spiculated region ", "espiculated region"]:
                                has_target = True
                            elif target == "calcification" and name in ["calcification", "cluster", "calcifications"]:
                                has_target = True
                except Exception:
                    pass

        rec = {
            "image_id": img_id,
            "patient_id": patient_id,
            "laterality": lat,
            "view": view,
            "dicom_path": dpath,
            "xml_path": xml_path,
            "pectoral_xml_path": pec_path if os.path.exists(pec_path) else None,
            "is_positive": has_target
        }
        records.append(rec)
        patient_records.setdefault(patient_id, []).append(rec)
        patient_positives[patient_id] = patient_positives.get(patient_id, False) or has_target

    # Stratified Patient-level split
    unique_patients = list(patient_records.keys())
    patient_labels = [1 if patient_positives[p] else 0 for p in unique_patients]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Split patients
    folds = {}
    patient_array = np.array(unique_patients)
    labels_array = np.array(patient_labels)

    for fold_idx, (train_pat_idx, val_pat_idx) in enumerate(sgkf.split(patient_array, labels_array, groups=patient_array)):
        train_pats = set(patient_array[train_pat_idx])
        val_pats = set(patient_array[val_pat_idx])

        # Assert zero leakage
        assert len(train_pats.intersection(val_pats)) == 0, f"Patient leakage detected in fold {fold_idx}!"

        # Gather sample records
        train_records = [r for p in train_pats for r in patient_records[p]]
        val_records = [r for p in val_pats for r in patient_records[p]]

        # Optional negative subsampling for train fold if negative_ratio > 0
        if negative_ratio > 0:
            pos_train = [r for r in train_records if r["is_positive"]]
            neg_train = [r for r in train_records if not r["is_positive"]]
            n_neg_keep = int(round(len(pos_train) * negative_ratio))
            np.random.seed(seed + fold_idx)
            if len(neg_train) > n_neg_keep:
                neg_train = list(np.random.choice(neg_train, size=n_neg_keep, replace=False))
            train_records = pos_train + neg_train

        folds[f"fold_{fold_idx}"] = {
            "train_patients": list(train_pats),
            "val_patients": list(val_pats),
            "train_samples": train_records,
            "val_samples": val_records
        }

    splits_data = {
        "n_splits": n_splits,
        "seed": seed,
        "target": target,
        "folds": folds
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits_data, f, indent=2)

    logger.info(f"Generated 5-fold patient-stratified splits saved to {output_path}")
    return splits_data


def precompute_and_cache_dataset(config: dict) -> None:
    """
    Precomputes cropped, normalized, padded 512x512 images and masks,
    saving them as .npz files into cache_dir.
    This provides a 20-50x speedup during training!
    """
    data_cfg = config.get("data", {})
    cache_dir = data_cfg.get("cache_dir", "data/cache")
    dicom_dir = data_cfg.get("dicom_dir", "INbreast_complete/AllDICOMs")
    xml_dir = data_cfg.get("xml_dir", "INbreast_complete/AllXML")
    pectoral_dir = data_cfg.get("pectoral_xml_dir", "INbreast_complete/PectoralMuscle/Pectoral Muscle XML")

    os.makedirs(cache_dir, exist_ok=True)
    dicoms = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    logger.info(f"Precomputing dataset cache for {len(dicoms)} images into {cache_dir}...")

    cached_count = 0
    for dpath in dicoms:
        fname = os.path.basename(dpath)
        img_id = fname.replace(".dcm", "").split("_")[0]
        out_file = os.path.join(cache_dir, f"{img_id}.npz")

        if os.path.exists(out_file):
            cached_count += 1
            continue

        xml_path = os.path.join(xml_dir, f"{img_id}.xml")
        pec_path = os.path.join(pectoral_dir, f"{img_id}_muscle.xml")
        lat = fname.split("_")[3] if len(fname.split("_")) > 3 else None

        img, mask = preprocess_sample(
            dicom_path=dpath,
            xml_path=xml_path,
            config=config,
            pectoral_xml_path=pec_path if os.path.exists(pec_path) else None,
            laterality=lat
        )
        np.savez_compressed(out_file, image=img, mask=mask)
        cached_count += 1

    logger.info(f"Caching complete! {cached_count} samples cached in {cache_dir}.")
