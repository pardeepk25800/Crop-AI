# tests/test_preprocessing.py — Preprocessing Tests for CropAI
# Run: python -m pytest tests/test_preprocessing.py -v

import pytest
import sys
import os
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImageUtils:
    """Tests for image utility functions."""

    def test_validate_valid_image(self, sample_image_bytes):
        """Valid JPEG should pass validation."""
        from utils import validate_image
        is_valid, msg = validate_image(sample_image_bytes)
        assert is_valid is True
        assert "Valid" in msg

    def test_validate_oversized_image(self):
        """Oversized image should fail validation."""
        from utils import validate_image
        # Create a fake oversized payload
        is_valid, msg = validate_image(b"x" * (11 * 1024 * 1024), max_size_mb=10)
        assert is_valid is False
        assert "too large" in msg

    def test_validate_corrupt_image(self):
        """Corrupt bytes should fail validation."""
        from utils import validate_image
        is_valid, msg = validate_image(b"not an image at all")
        assert is_valid is False
        assert "Invalid" in msg

    def test_load_image_from_pil(self, sample_image_pil):
        """load_image should accept PIL Image."""
        from utils import load_image
        img = load_image(sample_image_pil, target_size=128)
        assert img.size == (128, 128)
        assert img.mode == "RGB"

    def test_load_image_from_bytes(self, sample_image_bytes):
        """load_image should accept bytes."""
        from utils import load_image
        img = load_image(sample_image_bytes, target_size=224)
        assert img.size == (224, 224)

    def test_load_image_invalid_type(self):
        """load_image should raise TypeError for invalid input."""
        from utils import load_image
        with pytest.raises(TypeError):
            load_image(12345)

    def test_image_to_bytes(self, sample_image_pil):
        """image_to_bytes should produce valid bytes."""
        from utils import image_to_bytes
        b = image_to_bytes(sample_image_pil)
        assert isinstance(b, bytes)
        assert len(b) > 0
        # Verify the bytes can be opened as image
        img = Image.open(BytesIO(b))
        assert img.mode == "RGB"


class TestYieldDataCleaning:
    """Tests for yield data preprocessing."""

    def test_clean_yield_data_removes_negatives(self):
        """Cleaning should remove rows with negative numeric values."""
        from preprocessing import clean_yield_data

        df = pd.DataFrame({
            "Crop": ["Rice", "Rice", "Rice"],
            "Season": ["Kharif", "Kharif", "Kharif"],
            "State": ["Punjab", "Punjab", "Punjab"],
            "Area": [5.0, -1.0, 3.0],  # One negative
            "Annual_Rainfall": [1200, 800, 1500],
            "Fertilizer": [100, 120, 90],
            "Pesticide": [1.5, 2.0, 1.0],
            "Yield": [3500, 3200, 3800],
        })
        cleaned = clean_yield_data(df)
        assert len(cleaned) == 2

    def test_engineer_yield_features(self):
        """Feature engineering should add new columns."""
        from preprocessing import engineer_yield_features

        df = pd.DataFrame({
            "Area": [5.0],
            "Annual_Rainfall": [1200.0],
            "Fertilizer": [120.0],
            "Pesticide": [1.5],
            "Temperature": [28.0],
        })
        result = engineer_yield_features(df)
        assert "Rainfall_per_Area" in result.columns
        assert "Fertilizer_per_Area" in result.columns
        assert "Log_Area" in result.columns
        assert "Rain_Temp_Interact" in result.columns


class TestFormattingUtils:
    """Tests for formatting helper functions."""

    def test_format_bytes(self):
        """format_bytes should produce readable strings."""
        from utils import format_bytes
        assert "KB" in format_bytes(2048)
        assert "MB" in format_bytes(1024 * 1024)
        assert "B" in format_bytes(100)

    def test_format_duration(self):
        """format_duration should handle various time ranges."""
        from utils import format_duration
        assert "s" in format_duration(30)
        assert "m" in format_duration(120)
        assert "h" in format_duration(7200)

    def test_timer_context_manager(self):
        """Timer context manager should measure elapsed time."""
        import time as time_module
        from utils import Timer

        with Timer() as t:
            time_module.sleep(0.1)
        assert t.elapsed >= 0.05  # At least ~50ms

    def test_set_seed_reproducibility(self):
        """set_seed should make random operations reproducible."""
        from utils import set_seed

        set_seed(42)
        a1 = np.random.rand(5)
        set_seed(42)
        a2 = np.random.rand(5)
        np.testing.assert_array_equal(a1, a2)

    def test_get_device(self):
        """get_device should return a valid device string/object."""
        from utils import get_device
        device = get_device()
        assert device is not None


class TestDataSummary:
    """Tests for data summary functions."""

    def test_get_data_summary(self):
        """get_data_summary should return correct shape info."""
        from preprocessing import get_data_summary

        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4.0, 5.0, 6.0],
            "C": ["x", "y", "z"],
        })
        summary = get_data_summary(df)
        assert summary["shape"] == (3, 3)
        assert summary["missing_total"] == 0
        assert "numeric_stats" in summary
