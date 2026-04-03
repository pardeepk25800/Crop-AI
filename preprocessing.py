# preprocessing.py — Data Preprocessing Pipelines for CropAI
# Image augmentations, yield data cleaning, and feature engineering utilities.

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List, Optional

from config import (IMAGE_SIZE, MEAN, STD, CATEGORICAL_COLS, NUMERICAL_COLS,
                    YIELD_FEATURES, YIELD_TARGET)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def get_train_augmentations():
    """
    Get Albumentations-based training augmentation pipeline.
    Stronger augmentation than torchvision for better generalization.

    Returns:
        albumentations.Compose object
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),

        # Color augmentations (critical for leaf disease detection)
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.8),

        # Noise & blur (simulates camera quality variation)
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.3),

        # Simulate partial occlusion (leaves overlapping)
        A.CoarseDropout(
            max_holes=8, max_height=IMAGE_SIZE // 10,
            max_width=IMAGE_SIZE // 10, fill_value=0, p=0.2
        ),

        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_augmentations():
    """
    Validation/test augmentation — only resize and normalize.

    Returns:
        albumentations.Compose object
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_tta_augmentations(n_augments: int = 5):
    """
    Test-Time Augmentation (TTA) pipeline.
    Generate multiple augmented versions of a single image for ensemble prediction.

    Args:
        n_augments: Number of augmented copies to generate

    Returns:
        List of albumentations.Compose objects
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    base_normalize = [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]

    augments = [
        A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE)] + base_normalize),                    # Original
        A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.HorizontalFlip(p=1.0)] + base_normalize),
        A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.VerticalFlip(p=1.0)] + base_normalize),
        A.Compose([A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=(0.8, 1.0))] + base_normalize),
        A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Rotate(limit=15, p=1.0)] + base_normalize),
    ]

    return augments[:n_augments]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  YIELD DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def clean_yield_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive cleaning pipeline for crop yield data.

    Steps:
        1. Drop rows with missing target or features
        2. Remove negative values in numeric columns
        3. Remove extreme outliers (> 3 IQR)
        4. Standardize string columns (title case)
        5. Remove duplicate rows

    Args:
        df: Raw yield DataFrame

    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)

    # Drop missing values in critical columns
    df = df.dropna(subset=[YIELD_TARGET])
    df = df.dropna(subset=YIELD_FEATURES)

    # Remove negative values in numeric columns
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df = df[df[col] >= 0]

    # Remove extreme outliers (> 3× IQR)
    Q1, Q3 = df[YIELD_TARGET].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    df = df[(df[YIELD_TARGET] >= lower_bound) & (df[YIELD_TARGET] <= upper_bound)]

    # Standardize string columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Remove duplicates
    df = df.drop_duplicates()

    removed = initial_rows - len(df)
    print(f"[Preprocessing] Cleaned: {initial_rows} → {len(df)} rows "
          f"(removed {removed}, {removed / initial_rows * 100:.1f}%)")

    return df.reset_index(drop=True)


def engineer_yield_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for yield prediction.

    New features:
        - Rainfall_per_Area: Rainfall intensity per hectare
        - Fertilizer_per_Area: Fertilizer usage per hectare
        - Pesticide_per_Area: Pesticide usage per hectare
        - Fertilizer_Pesticide: Interaction term
        - Rain_Temp_Interact: Rainfall × Temperature interaction
        - Log_Area: Log-transformed area (reduces skewness)
        - Yield_Potential_Index: Composite agronomic index

    Args:
        df: Cleaned yield DataFrame

    Returns:
        DataFrame with additional engineered features
    """
    eps = 1e-6  # Avoid division by zero

    df = df.copy()

    # Basic ratios
    df["Rainfall_per_Area"]   = df["Annual_Rainfall"] / (df["Area"] + eps)
    df["Fertilizer_per_Area"] = df["Fertilizer"]      / (df["Area"] + eps)
    df["Pesticide_per_Area"]  = df["Pesticide"]       / (df["Area"] + eps)

    # Interaction terms
    df["Fertilizer_Pesticide"] = df["Fertilizer"] * df["Pesticide"]

    if "Temperature" in df.columns:
        df["Rain_Temp_Interact"] = df["Annual_Rainfall"] * df["Temperature"]

    # Log-transform for skewed features
    df["Log_Area"] = np.log1p(df["Area"])

    # Composite agronomic index
    if all(c in df.columns for c in ["Annual_Rainfall", "Fertilizer", "Temperature"]):
        df["Yield_Potential_Index"] = (
            df["Annual_Rainfall"] / 1000 *
            df["Fertilizer"] / 200 *
            np.clip(df["Temperature"] / 25, 0.5, 1.5)
        )

    print(f"[Preprocessing] Engineered {len(df.columns)} total features")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary report of the dataset for logging/display.

    Returns:
        dict with shape, column info, stats, and missing values
    """
    summary = {
        "shape":          df.shape,
        "columns":        list(df.columns),
        "dtypes":         df.dtypes.value_counts().to_dict(),
        "missing_total":  int(df.isnull().sum().sum()),
        "missing_pct":    round(df.isnull().mean().mean() * 100, 2),
        "duplicates":     int(df.duplicated().sum()),
    }

    # Stats for numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        summary["numeric_stats"] = {
            col: {
                "mean": round(df[col].mean(), 2),
                "std":  round(df[col].std(), 2),
                "min":  round(df[col].min(), 2),
                "max":  round(df[col].max(), 2),
            }
            for col in num_cols[:10]  # Limit to first 10
        }

    return summary
