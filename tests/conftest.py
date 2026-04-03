# tests/conftest.py — Shared Pytest Fixtures for CropAI

import sys
import os
import pytest
import numpy as np
from PIL import Image
from io import BytesIO

# Add project root to path so tests can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image_bytes():
    """Generate a sample leaf-like image as bytes for API testing."""
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))  # Green leaf
    # Add some variation
    pixels = np.array(img)
    noise = np.random.randint(-10, 10, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_image_pil():
    """Generate a sample PIL Image for model testing."""
    return Image.new("RGB", (224, 224), color=(34, 139, 34))


@pytest.fixture
def sample_yield_input():
    """Sample yield prediction input data."""
    return {
        "crop":        "Rice",
        "season":      "Kharif",
        "state":       "Punjab",
        "area":        5.0,
        "rainfall":    1200.0,
        "temperature": 28.0,
        "fertilizer":  120.0,
        "pesticide":   1.5,
    }


@pytest.fixture
def invalid_yield_input():
    """Invalid yield prediction input (bad crop name)."""
    return {
        "crop":        "InvalidCrop",
        "season":      "Kharif",
        "state":       "Punjab",
        "area":        5.0,
        "rainfall":    1200.0,
        "temperature": 28.0,
        "fertilizer":  120.0,
        "pesticide":   1.5,
    }


@pytest.fixture
def api_client():
    """FastAPI TestClient for API endpoint testing."""
    from fastapi.testclient import TestClient
    from api import app
    return TestClient(app)
