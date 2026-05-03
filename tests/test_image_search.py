"""
Unit tests for GenAI_Image_Search_application.py

These tests use mocked ML modules (via conftest.py) so they run in seconds
without requiring GPU, model weights, or heavy ML framework imports.
"""

import os
import sys
import csv
import io
import base64

import pytest
from PIL import Image

# Ensure the app directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))


# ---------------------------------------------------------------------------
# Import app module (heavy deps are already mocked by conftest.py)
# ---------------------------------------------------------------------------

def _import_app():
    """Import (or reimport) the application module."""
    mod_name = "GenAI_Image_Search_application"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    import GenAI_Image_Search_application as app
    return app


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_image(tmp_path, name="test.png", mode="RGB", size=(4, 4), color="red"):
    path = str(tmp_path / name)
    Image.new(mode, size, color=color).save(path)
    return path


# ===========================================================================
# Tests - preprocess_image
# ===========================================================================


class TestPreprocessImage:
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
# Tests - image_to_base64
# ===========================================================================


class TestImageToBase64:
    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        img_path = _make_image(tmp_path, "img.png", color="green")
        b64 = app.image_to_base64(img_path)
        decoded = base64.b64decode(b64)
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (4, 4)

    def test_returns_string(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        img_path = _make_image(tmp_path)
        result = app.image_to_base64(img_path)
        assert isinstance(result, str)

    def test_valid_base64_encoding(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        img_path = _make_image(tmp_path, size=(10, 10))
        b64 = app.image_to_base64(img_path)
        raw = base64.b64decode(b64)
        assert len(raw) > 0


# ===========================================================================
# Tests - export_results_as_csv
# ===========================================================================


class TestExportResultsAsCsv:
    def test_csv_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        images = ["img1.png", "img2.png"]
        probs = [85.5, 42.3]
        filepath, csv_content = app.export_results_as_csv("test prompt", images, probs)

        reader = csv.reader(io.StringIO(csv_content), delimiter=";")
        rows = list(reader)

        assert rows[0] == ["Index", "Image File Name", "Probability"]
        assert rows[1][1] == "img1.png"
        assert rows[2][1] == "img2.png"
        assert len(rows) == 3

    def test_filename_contains_prompt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        filepath, _ = app.export_results_as_csv("hello world", [], [])
        basename = os.path.basename(filepath)
        assert "hello world" in basename
        assert basename.endswith(".csv")

    def test_filename_contains_timestamp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        filepath, _ = app.export_results_as_csv("test", [], [])
        basename = os.path.basename(filepath)
        assert basename[:8].isdigit()

    def test_empty_results(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        filepath, csv_content = app.export_results_as_csv("empty", [], [])
        reader = csv.reader(io.StringIO(csv_content), delimiter=";")
        rows = list(reader)
        assert len(rows) == 1  # only header

    def test_filepath_in_temp_csv_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        filepath, _ = app.export_results_as_csv("test", [], [])
        assert "temp_csv" in filepath

    def test_probabilities_in_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        images = ["a.png", "b.png"]
        probs = [99.1, 50.0]
        _, csv_content = app.export_results_as_csv("p", images, probs)
        reader = csv.reader(io.StringIO(csv_content), delimiter=";")
        rows = list(reader)
        assert float(rows[1][2]) == 99.1
        assert float(rows[2][2]) == 50.0


# ===========================================================================
# Tests - display_uploaded_images
# ===========================================================================


class TestDisplayUploadedImages:
    def test_returns_empty_on_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        assert app.display_uploaded_images(None) == []

    def test_returns_empty_on_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        assert app.display_uploaded_images([]) == []

    def test_creates_thumbnails(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        img_path = _make_image(tmp_path, "photo.png", size=(800, 800), color="blue")
        result = app.display_uploaded_images([img_path])
        assert len(result) == 1
        thumb = Image.open(result[0])
        assert max(thumb.size) <= 300

    def test_multiple_images(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        paths = [_make_image(tmp_path, f"img{i}.png", size=(600, 400)) for i in range(3)]
        result = app.display_uploaded_images(paths)
        assert len(result) == 3

    def test_skips_invalid_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        bad_path = str(tmp_path / "not_an_image.txt")
        with open(bad_path, "w") as f:
            f.write("not an image")
        result = app.display_uploaded_images([bad_path])
        assert result == []


# ===========================================================================
# Tests - search_images (empty input)
# ===========================================================================


class TestSearchImages:
    def test_empty_file_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        images, probs = app.search_images("test", [], 25)
        assert images == []
        assert probs == []


# ===========================================================================
# Tests - export_csv
# ===========================================================================


class TestExportCsv:
    def test_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = _import_app()
        app.filtered_images_list = ["img1.png", "img2.png"]
        app.probabilities_list = [80.0, 60.0]

        filepath = app.export_csv("test prompt")
        assert os.path.exists(filepath)
        with open(filepath, "r") as f:
            content = f.read()
        assert "img1.png" in content
        assert "img2.png" in content
