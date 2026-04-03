# utils.py — Common Utility Functions for CropAI
# Shared helpers for image handling, benchmarking, reproducibility, and formatting.

import os
import io
import time
import random
import functools
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1.  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """
    Set random seed for full reproducibility across all libraries.
    Call at the top of any training / evaluation script.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 2.  IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_IMAGE_SIZE_MB = 10


def validate_image(file_bytes: bytes, max_size_mb: float = MAX_IMAGE_SIZE_MB) -> Tuple[bool, str]:
    """
    Validate an uploaded image: check size, format, and corruption.

    Returns:
        (is_valid, message)
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"Image too large: {size_mb:.1f} MB (max {max_size_mb} MB)"

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # Checks for corruption
        return True, f"Valid {img.format} image ({img.size[0]}×{img.size[1]})"
    except Exception as e:
        return False, f"Invalid image file: {e}"


def load_image(source: Union[str, bytes, Image.Image], target_size: Optional[int] = None) -> Image.Image:
    """
    Unified image loader — accepts file path, raw bytes, or PIL Image.

    Args:
        source:      File path (str), raw bytes, or PIL.Image
        target_size: Optional resize to (target_size × target_size)

    Returns:
        RGB PIL Image
    """
    if isinstance(source, str):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Image not found: {source}")
        img = Image.open(source)
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, Image.Image):
        img = source
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")

    img = img.convert("RGB")

    if target_size:
        img = img.resize((target_size, target_size), Image.LANCZOS)

    return img


def image_to_bytes(img: Image.Image, format: str = "JPEG", quality: int = 85) -> bytes:
    """Convert PIL Image to bytes buffer."""
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TIMING / BENCHMARKING
# ─────────────────────────────────────────────────────────────────────────────

def timer(func):
    """
    Decorator to time function execution.

    Usage:
        @timer
        def train_epoch(...):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"⏱️  {func.__name__}() → {elapsed:.2f}s")
        return result
    return wrapper


class Timer:
    """
    Context-manager timer for profiling code blocks.

    Usage:
        with Timer("Model inference"):
            prediction = model.predict(img)
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            print(f"⏱️  {self.label}: {self.elapsed:.3f}s")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def format_bytes(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def print_banner(title: str, width: int = 60):
    """Print a styled section banner."""
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model) -> dict:
    """
    Count trainable and total parameters of a PyTorch model.

    Returns:
        dict with 'total', 'trainable', 'frozen' counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
        "total_M":   f"{total / 1e6:.2f}M",
    }


def get_device():
    """Return the best available PyTorch device."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return "cpu"
