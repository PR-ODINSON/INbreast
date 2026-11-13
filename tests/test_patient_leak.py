import os
import sys
import json
import yaml
try:
    import pytest
except ImportError:
    pytest = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import build_patient_stratified_splits


def test_patient_leak_in_splits():
    """
    Hard Rule Test:
    Asserts zero patient leakage between train and validation sets across all 5 folds.
    """
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    splits_path = config["data"].get("splits_path", "splits.json")
    if not os.path.exists(splits_path):
        splits_data = build_patient_stratified_splits(config, output_path=splits_path)
    else:
        with open(splits_path, "r") as f:
            splits_data = json.load(f)

    folds = splits_data["folds"]
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    all_val_patients = []
    for fold_name, fold_info in folds.items():
        train_pats = set(fold_info["train_patients"])
        val_pats = set(fold_info["val_patients"])

        # Core assertion: Train and Val sets must be completely disjoint
        intersection = train_pats.intersection(val_pats)
        assert len(intersection) == 0, (
            f"CRITICAL LEAKAGE in {fold_name}: {len(intersection)} patients appear in both "
            f"train and val sets! Patients: {intersection}"
        )

        all_val_patients.extend(fold_info["val_patients"])

        # Assert no sample in val has a patient in train
        val_sample_patients = {s["patient_id"] for s in fold_info["val_samples"]}
        train_sample_patients = {s["patient_id"] for s in fold_info["train_samples"]}
        sample_intersection = train_sample_patients.intersection(val_sample_patients)
        assert len(sample_intersection) == 0, (
            f"CRITICAL SAMPLE LEAKAGE in {fold_name}: {sample_intersection}"
        )

    # Assert every patient is evaluated in exactly one validation fold
    assert len(all_val_patients) == len(set(all_val_patients)), "Some patients appear in multiple validation folds!"
    print(f"\n[PASS] Verified zero patient leakage across all 5 folds. Total unique patients: {len(set(all_val_patients))}")


if __name__ == "__main__":
    test_patient_leak_in_splits()
