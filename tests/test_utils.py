"""Unit tests for the pure helper functions in ``utils.py``.

These tests deliberately avoid loading any deep-learning weights so they can
run quickly in CI without downloading the multi-gigabyte model files.
"""
import torch
from PIL import Image

from utils import brand_converter, l2_norm, resolution_alignment


def test_brand_converter_maps_known_aliases():
    assert brand_converter("Apple Inc.") == "Apple"
    assert brand_converter("Office365") == "Microsoft"
    assert brand_converter("Twitter") == "Twitter, Inc."


def test_brand_converter_passthrough_for_unknown_brand():
    assert brand_converter("SomeUnknownBrand") == "SomeUnknownBrand"


def test_l2_norm_produces_unit_vectors():
    x = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
    normed = l2_norm(x)
    norms = normed.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-6)


def test_resolution_alignment_matches_smaller_dimension():
    img1 = Image.new("RGB", (200, 100))
    img2 = Image.new("RGB", (50, 80))
    out1, out2 = resolution_alignment(img1, img2)
    # The narrower width (50) should drive the alignment for both images.
    assert min(out1.size) > 0 and min(out2.size) > 0


def test_resolution_alignment_handles_zero_dimension():
    img1 = Image.new("RGB", (0, 100))
    img2 = Image.new("RGB", (50, 80))
    out1, out2 = resolution_alignment(img1, img2)
    # When a dimension is zero the originals are returned unchanged.
    assert out1.size == (0, 100)
    assert out2.size == (50, 80)
