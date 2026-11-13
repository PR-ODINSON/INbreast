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
    p_img, p_mask = resize_and_pad(img, mask, target_size=target_size)

    assert p_img.shape == (target_size, target_size)
    assert p_mask.shape == (target_size, target_size)
    assert set(np.unique(p_mask)).issubset({0, 1})
    assert np.count_nonzero(p_mask) > 0
    print(f"\n[PASS] Verified aspect-preserving resize & pad to {target_size}x{target_size}.")


if __name__ == "__main__":
    test_xml_rasterization_mass()
    test_empty_xml_rasterization()
    test_resize_and_pad()
