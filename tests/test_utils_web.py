"""Unit tests for the web-tool helper functions in ``WEBtool/utils_web.py``."""
import pickle

import numpy as np

from utils_web import allowed_file, convert_to_base64, domain_map_add, domain_map_delete


def test_allowed_file_accepts_images():
    assert allowed_file("logo.png")
    assert allowed_file("photo.JPG")
    assert allowed_file("scan.jpeg")


def test_allowed_file_rejects_non_images():
    assert not allowed_file("malware.exe")
    assert not allowed_file("no_extension")


def test_convert_to_base64_returns_none_for_none():
    assert convert_to_base64(None) is None


def test_convert_to_base64_encodes_image():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    encoded = convert_to_base64(image)
    assert isinstance(encoded, str) and len(encoded) > 0


def test_domain_map_add_and_delete(tmp_path):
    domain_map_path = tmp_path / "domain_map.pkl"
    with open(domain_map_path, "wb") as f:
        pickle.dump({}, f)

    domain_map_add("Example", "example.com, example.org", str(domain_map_path))
    with open(domain_map_path, "rb") as f:
        domain_map = pickle.load(f)
    assert domain_map["Example"] == ["example.com", "example.org"]

    # Adding again should not introduce duplicates.
    domain_map_add("Example", "example.com, example.net", str(domain_map_path))
    with open(domain_map_path, "rb") as f:
        domain_map = pickle.load(f)
    assert domain_map["Example"] == ["example.com", "example.org", "example.net"]

    domain_map_delete("Example", str(domain_map_path))
    with open(domain_map_path, "rb") as f:
        domain_map = pickle.load(f)
    assert "Example" not in domain_map
