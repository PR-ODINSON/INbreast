"""3-way patient-level leak test for nested 5-fold splits.

Checks:
  - train ∩ val = ∅
  - train ∩ test = ∅
  - val  ∩ test = ∅
  - Every patient appears in TEST exactly once across all 5 folds.
"""

import json
import os
import sys

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import build_patient_stratified_splits


def test_patient_leak_in_splits():
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    splits_path = config["data"].get("splits_path", "splits.json")
    if not os.path.exists(splits_path):
        splits_data = build_patient_stratified_splits(config, output_path=splits_path)
    else:
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

    scheme = splits_data.get("scheme", "")
    assert "nested" in scheme, (
        f"splits.json scheme='{scheme}' is not nested. Regenerate with src/dataset.py."
    )

    folds = splits_data["folds"]
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    all_test_patients = []

    for fold_name, fold_info in folds.items():
        for key in ("train_patients", "val_patients", "test_patients",
                    "train_samples", "val_samples", "test_samples"):
            assert key in fold_info, f"{fold_name} missing '{key}'"

        train_pats = set(fold_info["train_patients"])
        val_pats   = set(fold_info["val_patients"])
        test_pats  = set(fold_info["test_patients"])

        # 3-way partition at patient level
        assert train_pats & val_pats  == set(), f"{fold_name}: train∩val={train_pats & val_pats}"
        assert train_pats & test_pats == set(), f"{fold_name}: train∩test={train_pats & test_pats}"
        assert val_pats  & test_pats  == set(), f"{fold_name}: val∩test={val_pats & test_pats}"

        # Sample-level (each sample's patient_id must be in the right partition)
        def _check_samples(samples, allowed, label):
            bad = {s["patient_id"] for s in samples if s["patient_id"] not in allowed}
            assert not bad, f"{fold_name} {label} samples contain alien patients: {bad}"

        _check_samples(fold_info["train_samples"], train_pats, "train")
        _check_samples(fold_info["val_samples"],   val_pats,   "val")
        _check_samples(fold_info["test_samples"],  test_pats,  "test")

        # Cross-partition sample leakage
        train_sp = {s["patient_id"] for s in fold_info["train_samples"]}
        val_sp   = {s["patient_id"] for s in fold_info["val_samples"]}
        test_sp  = {s["patient_id"] for s in fold_info["test_samples"]}
        assert train_sp & val_sp  == set(), f"{fold_name}: sample train∩val"
        assert train_sp & test_sp == set(), f"{fold_name}: sample train∩test"
        assert val_sp  & test_sp  == set(), f"{fold_name}: sample val∩test"

        all_test_patients.extend(fold_info["test_patients"])

    # Every patient appears in TEST exactly once
    counts = {}
    for pid in all_test_patients:
        counts[pid] = counts.get(pid, 0) + 1
    multi = {p: c for p, c in counts.items() if c > 1}
    assert not multi, f"Patients in test more than once: {multi}"

    total = splits_data["n_patients"]
    assert len(all_test_patients) == total, (
        f"Not all patients covered in test: {len(all_test_patients)} entries for {total} patients"
    )

    print(
        f"\n[PASS] 3-way partition verified across all 5 folds. "
        f"{total} unique patients, each in TEST exactly once."
    )


if __name__ == "__main__":
    test_patient_leak_in_splits()
