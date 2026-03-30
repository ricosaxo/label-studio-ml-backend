"""
Tests for the base64 data-URL support added to set_image() in model.py.

These tests verify the exact logic change without requiring sam2 or a GPU.
The logic under test is the if/else branch in set_image():

    if isinstance(image_url, str) and image_url.startswith('data:image'):
        _, b64_data = image_url.split(',', 1)
        image = Image.open(BytesIO(base64.b64decode(b64_data)))
    else:
        image_path = self.get_local_path(image_url, task_id=task_id)
        image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
"""
import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image


def _make_data_url(width: int, height: int, color: tuple, fmt: str = "PNG") -> str:
    """Helper: create a data:image/...;base64 URL from a solid-color image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{b64}"


def _decode_data_url(data_url: str) -> np.ndarray:
    """Exact copy of the new set_image() branch for isolated testing."""
    _, b64_data = data_url.split(",", 1)
    image = Image.open(BytesIO(base64.b64decode(b64_data)))
    return np.array(image.convert("RGB"))


class TestBase64DataURL:
    def test_png_shape(self):
        url = _make_data_url(20, 10, (128, 64, 32))
        arr = _decode_data_url(url)
        assert arr.shape == (10, 20, 3), f"Expected (10, 20, 3), got {arr.shape}"

    def test_png_pixel_values(self):
        url = _make_data_url(5, 5, (255, 0, 0))  # pure red
        arr = _decode_data_url(url)
        assert arr[0, 0, 0] == 255, "Red channel should be 255"
        assert arr[0, 0, 1] == 0,   "Green channel should be 0"
        assert arr[0, 0, 2] == 0,   "Blue channel should be 0"

    def test_jpeg_roundtrip(self):
        """JPEG is lossy, so only check shape and approximate values."""
        url = _make_data_url(8, 8, (0, 255, 0), fmt="JPEG")
        arr = _decode_data_url(url)
        assert arr.shape == (8, 8, 3)
        # Green-dominant (lossy compression: allow 30-unit tolerance)
        assert arr[4, 4, 1] > arr[4, 4, 0] + 30
        assert arr[4, 4, 1] > arr[4, 4, 2] + 30

    def test_dtype_is_uint8(self):
        url = _make_data_url(4, 4, (10, 20, 30))
        arr = _decode_data_url(url)
        assert arr.dtype == np.uint8

    def test_data_url_detection(self):
        """The branch condition: only trigger on data:image prefix."""
        url = _make_data_url(2, 2, (0, 0, 0))
        assert isinstance(url, str) and url.startswith("data:image"), \
            "Helper must produce a data:image URL"

    def test_non_data_url_not_matched(self):
        """Verify that normal URLs fall through to the else-branch (no exception here)."""
        normal_urls = [
            "http://example.com/image.png",
            "/local/path/image.jpg",
            "s3://bucket/key.png",
        ]
        for url in normal_urls:
            assert not (isinstance(url, str) and url.startswith("data:image")), \
                f"URL should NOT trigger base64 branch: {url}"
