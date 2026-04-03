# Contributing to CropAI

Thank you for considering contributing to CropAI! This document outlines the guidelines for contributing to this project.

---

## 🚀 Getting Started

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Install dependencies**: `pip install -r requirements.txt`
5. **Make your changes**
6. **Run tests**: `python -m pytest tests/ -v`
7. **Commit**: `git commit -m "feat: add your feature description"`
8. **Push**: `git push origin feature/your-feature-name`
9. **Create a Pull Request**

---

## 📂 Project Structure

```
crop_ai/
├── config.py            # Central configuration — add new constants here
├── utils.py             # Shared utility functions
├── preprocessing.py     # Data preprocessing pipelines
├── data_generator.py    # Synthetic dataset creation
├── disease_model.py     # CNN disease detection model
├── yield_model.py       # Yield prediction ensemble
├── evaluate.py          # Model evaluation & reporting
├── visualization.py     # Plotting & EDA
├── database.py          # Prediction history (SQLite)
├── logger_config.py     # Centralized logging
├── api.py               # FastAPI REST backend
├── streamlit_app.py     # Streamlit web UI
├── train.py             # One-click training pipeline
├── tests/               # Unit tests (pytest)
│   ├── conftest.py      # Shared fixtures
│   ├── test_api.py      # API endpoint tests
│   ├── test_models.py   # Model architecture tests
│   └── test_preprocessing.py  # Data processing tests
└── requirements.txt     # Pin all dependencies
```

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

---

## 📐 Code Style

- **Python 3.10+** — Use modern Python features (type hints, f-strings, walrus operator)
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Docstrings**: Use Google-style docstrings for all public functions
- **Type hints**: Required for function parameters and return types
- **Max line length**: 100 characters
- **Imports**: Standard library → Third-party → Local modules

Example:
```python
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from config import IMAGE_SIZE
from utils import validate_image


def process_leaf_image(image_path: str, target_size: int = IMAGE_SIZE) -> Tuple[bool, Optional[Image.Image]]:
    """
    Process a leaf image for disease detection.

    Args:
        image_path:  Path to the leaf image file
        target_size: Resize dimension (default from config)

    Returns:
        Tuple of (success, processed_image or None)
    """
    ...
```

---

## 🔀 Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `test:` | Adding/updating tests |
| `refactor:` | Code refactoring (no new features, no bug fixes) |
| `style:` | Formatting, whitespace, etc. |
| `perf:` | Performance improvement |

---

## 🐛 Reporting Bugs

When reporting bugs, include:
1. **Steps to reproduce** the issue
2. **Expected behavior** vs **Actual behavior**
3. **Python version** and **OS**
4. **Full error traceback** if applicable
5. **Sample input** that triggers the bug (e.g., image or request payload)

---

## 🔒 Security

If you discover a security vulnerability, please report it privately instead of creating a public issue.

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
