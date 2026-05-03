"""
Unit tests for GenAI_Image_Search_application_YOLOv5.py

These tests use mocked ML modules (via conftest.py) so they run in seconds
without requiring GPU, model weights, or heavy ML framework imports.
"""

import os
import sys

import pytest
from PIL import Image

# Ensure the app directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))


# ---------------------------------------------------------------------------
# Import app module (heavy deps are already mocked by conftest.py)
# ---------------------------------------------------------------------------

def _import_app():
    """Import (or reimport) the YOLOv5 application module."""
    mod_name = "GenAI_Image_Search_application_YOLOv5"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    import GenAI_Image_Search_application_YOLOv5 as app
    return app


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_image(tmp_path, name="test.png", mode="RGB", size=(64, 64), color="red"):
    path = str(tmp_path / name)
    Image.new(mode, size, color=color).save(path)
    return path


# ===========================================================================
# Tests - preprocess_image
# ===========================================================================


class TestPreprocessImageYOLO:
    def test_rgb_passthrough(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        img = Image.new("RGB", (4, 4), color="blue")
        result = app.preprocess_image(img)
        assert result.mode == "RGB"

    def test_rgba_converted_to_rgb(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        img = Image.new("RGBA", (4, 4), color=(255, 0, 0, 128))
        result = app.preprocess_image(img)
        assert result.mode == "RGB"

    def test_grayscale_converted_to_rgb(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        img = Image.new("L", (4, 4), color=128)
        result = app.preprocess_image(img)
        assert result.mode == "RGB"

    def test_palette_converted_to_rgb(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        img = Image.new("P", (4, 4))
        result = app.preprocess_image(img)
        assert result.mode == "RGB"

    def test_cmyk_converted_to_rgb(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        img = Image.new("CMYK", (4, 4), color=(0, 0, 0, 0))
        result = app.preprocess_image(img)
        assert result.mode == "RGB"

    def test_preserves_dimensions(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        img = Image.new("RGBA", (100, 200), color=(255, 0, 0, 128))
        result = app.preprocess_image(img)
        assert result.size == (100, 200)


# ===========================================================================
# Tests - detect_objects
# ===========================================================================


class TestDetectObjects:
    def test_calls_yolo_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        img_path = _make_image(tmp_path, "detect.png", color="blue")
        # detect_objects opens the image and passes to yolo_model
        result = app.detect_objects(img_path)
        # Should have called the mocked yolo_model
        assert result is not None


# ===========================================================================
# Tests - search_images (empty input)
# ===========================================================================


class TestSearchImagesYOLO:
    def test_empty_file_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        images, probs = app.search_images("test", [], 25)
        assert images == []
        assert probs == []
