import os
import sys
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import parse_xml_rois, resize_and_pad, get_breast_bounding_box


def test_xml_rasterization_mass():
    """
    Tests XML ROI parsing and polygon rasterization on a known positive INbreast image.
    Image 20586908 is documented to have 2 mass contours.
    """
    xml_path = "INbreast_complete/AllXML/20586908.xml"
    if not os.path.exists(xml_path):
        pytest.skip(f"Dataset file {xml_path} not found.")

    shape = (4084, 3328)
    mask = parse_xml_rois(xml_path=xml_path, target="mass", image_shape=shape)

    assert mask.shape == shape, f"Expected shape {shape}, got {mask.shape}"
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1}), "Mask values must be strictly binary {0, 1}"
    
    nonzero_pixels = np.count_nonzero(mask)
    # Moreira et al. and our earlier check confirmed ~36,900 pixels
    assert 35000 < nonzero_pixels < 40000, f"Unexpected nonzero pixel count: {nonzero_pixels}"
    print(f"\n[PASS] Verified mass rasterization for 20586908: {nonzero_pixels:,} positive pixels.")


def test_empty_xml_rasterization():
    """
    Tests that a missing or non-existent XML produces an all-zero mask without crashing.
    """
    shape = (512, 512)
    mask = parse_xml_rois(xml_path="non_existent.xml", target="mass", image_shape=shape)
    assert mask.shape == shape
    assert np.count_nonzero(mask) == 0
    print("\n[PASS] Verified empty mask handling for negative sample.")


def test_resize_and_pad():
    """
    Tests aspect-preserving padding and resizing to square dimensions.
    """
    img = np.random.rand(100, 200).astype(np.float32)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[40:60, 80:120] = 1

    target_size = 512
    p_img, p_mask, _scale = resize_and_pad(img, mask, target_size=target_size)

    assert p_img.shape == (target_size, target_size)
    assert p_mask.shape == (target_size, target_size)
    assert set(np.unique(p_mask)).issubset({0, 1})
    assert np.count_nonzero(p_mask) > 0
    print(f"\n[PASS] Verified aspect-preserving resize & pad to {target_size}x{target_size}.")


def test_spiculated_excluded_from_mass_target():
    """Spiculated Region ROIs must NOT appear when target='mass'.

    When an XML contains both Mass and Spiculated Region contours,
    the mass-only mask must have fewer (or equal) pixels than the
    mass_spiculated mask — and the masks must differ.
    """
    import glob
    import plistlib

    xml_dir = "INbreast_complete/AllXML"
    if not os.path.exists(xml_dir):
        print("[SKIP] test_spiculated_excluded_from_mass_target: dataset not found")
        return

    # Find an XML that contains a Spiculated Region ROI
    spiculated_xml = None
    for xf in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
        try:
            with open(xf, "rb") as fh:
                plist = plistlib.load(fh)
        except Exception:
            continue
        for img in plist.get("Images", []):
            for roi in img.get("ROIs", []):
                name = str(roi.get("Name", "")).strip().lower()
                if "spiculated" in name:
                    spiculated_xml = xf
                    break
            if spiculated_xml:
                break
        if spiculated_xml:
            break

    if spiculated_xml is None:
        print("[SKIP] test_spiculated_excluded_from_mass_target: no spiculated XML found")
        return

    # Use DICOM full resolution so the polygon points land on the canvas.
    # INbreast images are either 4084x3328 or 3328x2560.
    H, W = 4084, 3328
    mask_mass = parse_xml_rois(spiculated_xml, target="mass",             image_shape=(H, W))
    mask_spic = parse_xml_rois(spiculated_xml, target="mass_spiculated",  image_shape=(H, W))

    # mass_spiculated adds the ellipses, so it must have >= pixels
    assert mask_spic.sum() >= mask_mass.sum(), (
        f"mass_spiculated ({mask_spic.sum()}) should have >= pixels than mass ({mask_mass.sum()})"
    )
    # They should differ (spiculated region is non-empty)
    assert not np.array_equal(mask_mass, mask_spic), (
        "Masks are identical — spiculated region was not added for mass_spiculated target"
    )
    print(
        f"[PASS] test_spiculated_excluded_from_mass_target: "
        f"mass={mask_mass.sum()} px, mass_spiculated={mask_spic.sum()} px  "
        f"(file: {os.path.basename(spiculated_xml)})"
    )


def test_resize_and_pad_returns_scale():
    """resize_and_pad returns a 3-tuple (img, mask, scale)."""
    img  = np.random.rand(100, 200).astype(np.float32)
    mask = np.zeros((100, 200), dtype=np.uint8)
    result = resize_and_pad(img, mask, target_size=512)
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
    p_img, p_mask, scale = result
    assert p_img.shape == (512, 512)
    assert isinstance(scale, float)
    print(f"[PASS] test_resize_and_pad_returns_scale  scale={scale:.4f}")


if __name__ == "__main__":
    test_xml_rasterization_mass()
    test_empty_xml_rasterization()
    test_resize_and_pad()
    test_spiculated_excluded_from_mass_target()
    test_resize_and_pad_returns_scale()
