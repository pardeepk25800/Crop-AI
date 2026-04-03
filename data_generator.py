# data_generator.py — Generate synthetic datasets for training & testing
# In production: replace with real PlantVillage images + FAO crop yield CSV

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import random
import json
from config import (DATA_DIR, DISEASE_CLASSES, DISEASE_DATA_DIR,
                    YIELD_DATA_PATH, IMAGE_SIZE)

random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC LEAF IMAGE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette per disease category
DISEASE_PALETTES = {
    "healthy":       [(34, 139, 34), (0, 128, 0), (50, 205, 50)],
    "blight":        [(139, 69, 19), (160, 82, 45), (101, 67, 33)],
    "rust":          [(184, 115, 51), (210, 105, 30), (165, 42, 42)],
    "scab":          [(85, 107, 47), (107, 142, 35), (154, 205, 50)],
    "rot":           [(75, 0, 130), (128, 0, 128), (148, 0, 211)],
    "spot":          [(255, 165, 0), (255, 140, 0), (255, 127, 80)],
    "mildew":        [(200, 200, 200), (220, 220, 220), (180, 180, 180)],
    "virus":         [(255, 255, 0), (255, 215, 0), (240, 230, 140)],
    "mite":          [(205, 133, 63), (139, 90, 43), (160, 120, 90)],
    "bacterial":     [(139, 0, 0), (178, 34, 34), (220, 20, 60)],
}

def _palette_for_class(class_name: str):
    name_lower = class_name.lower()
    for key, palette in DISEASE_PALETTES.items():
        if key in name_lower:
            return palette
    return DISEASE_PALETTES["healthy"]


def generate_leaf_image(class_name: str, size: int = IMAGE_SIZE) -> Image.Image:
    """Create a synthetic leaf-like RGB image for a given disease class."""
    palette = _palette_for_class(class_name)
    base_color = random.choice(palette)

    # Green background (leaf base)
    bg = np.full((size, size, 3), (34, 120, 34), dtype=np.uint8)

    # Add leaf shape (ellipse mask)
    y_idx, x_idx = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    mask = ((x_idx - cx) ** 2 / (cx * 0.7) ** 2 +
            (y_idx - cy) ** 2 / (cy * 0.85) ** 2) <= 1
    new_color = [max(0, min(255, c + random.randint(-20, 20))) for c in base_color]
    bg[mask] = np.array(new_color, dtype=np.uint8)

    # Add noise / texture
    noise = np.random.randint(-15, 15, (size, size, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add disease spots (random circles) for non-healthy classes
    if "healthy" not in class_name.lower():
        img_pil = Image.fromarray(bg)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img_pil)
        spot_color = tuple(random.choice([
            (139, 69, 19), (101, 67, 33), (165, 42, 42), (85, 107, 47)
        ]))
        for _ in range(random.randint(3, 12)):
            x = random.randint(cx - cx//2, cx + cx//2)
            y = random.randint(cy - cy//2, cy + cy//2)
            r = random.randint(5, 25)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=spot_color)
        bg = np.array(img_pil)

    img = Image.fromarray(bg).filter(ImageFilter.GaussianBlur(radius=0.8))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.9, 1.3))
    return img


def generate_disease_dataset(samples_per_class: int = 50):
    """
    Generate synthetic leaf images for all 38 disease classes.
    Real usage: download PlantVillage from Kaggle and point DISEASE_DATA_DIR there.
    """
    print(f"[DataGen] Creating synthetic PlantVillage dataset — "
          f"{samples_per_class} images × {len(DISEASE_CLASSES)} classes")

    class_counts = {}
    for cls in DISEASE_CLASSES:
        cls_dir = os.path.join(DISEASE_DATA_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(samples_per_class):
            img = generate_leaf_image(cls)
            img.save(os.path.join(cls_dir, f"img_{i:04d}.jpg"), quality=85)
        class_counts[cls] = samples_per_class

    total = sum(class_counts.values())
    print(f"[DataGen] [OK] {total} images saved -> {DISEASE_DATA_DIR}")

    # Save class index mapping
    mapping = {i: cls for i, cls in enumerate(DISEASE_CLASSES)}
    with open(os.path.join(DATA_DIR, "class_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)
    print("[DataGen] [OK] class_mapping.json saved")
    return class_counts


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SYNTHETIC CROP YIELD CSV GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

CROPS = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane",
         "Soybean", "Potato", "Tomato", "Groundnut", "Bajra"]
SEASONS = ["Kharif", "Rabi", "Zaid", "Whole Year"]
STATES = [
    "Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Andhra Pradesh",
    "Karnataka", "Tamil Nadu", "West Bengal", "Madhya Pradesh", "Rajasthan",
    "Bihar", "Gujarat", "Odisha", "Assam", "Jharkhand",
]

# Base yield (kg/ha) per crop
BASE_YIELDS = {
    "Rice": 3500, "Wheat": 3200, "Maize": 3000, "Cotton": 1800,
    "Sugarcane": 60000, "Soybean": 1400, "Potato": 18000,
    "Tomato": 20000, "Groundnut": 1600, "Bajra": 1200,
}

# Optimal rainfall (mm) per crop
OPT_RAIN = {
    "Rice": 1500, "Wheat": 400, "Maize": 700, "Cotton": 700,
    "Sugarcane": 1800, "Soybean": 700, "Potato": 600,
    "Tomato": 500, "Groundnut": 600, "Bajra": 400,
}

# Optimal temperature (°C) per crop
OPT_TEMP = {
    "Rice": 28, "Wheat": 20, "Maize": 26, "Cotton": 28,
    "Sugarcane": 30, "Soybean": 24, "Potato": 18,
    "Tomato": 25, "Groundnut": 28, "Bajra": 30,
}


def _simulate_yield(crop, rainfall, temperature, fertilizer, pesticide):
    base = BASE_YIELDS[crop]

    # Rainfall factor (penalise deviation from optimum)
    r_diff = abs(rainfall - OPT_RAIN[crop]) / 1000
    r_factor = max(0.5, 1.0 - r_diff * 0.35)

    # Temperature factor
    t_diff = abs(temperature - OPT_TEMP[crop]) / 10
    t_factor = max(0.6, 1.0 - t_diff * 0.25)

    # Fertilizer factor (diminishing returns)
    f_factor = min(1.25, 0.75 + (fertilizer / 500) * 0.5)

    # Pesticide factor
    p_factor = min(1.10, 0.90 + (pesticide / 50) * 0.2)

    noise = np.random.normal(1.0, 0.08)
    yield_val = base * r_factor * t_factor * f_factor * p_factor * noise
    return max(100, round(yield_val, 2))


def generate_yield_dataset(n_rows: int = 5000):
    """Generate a realistic synthetic crop yield CSV."""
    print(f"[DataGen] Generating crop yield dataset — {n_rows} rows")

    records = []
    for _ in range(n_rows):
        crop      = random.choice(CROPS)
        season    = random.choice(SEASONS)
        state     = random.choice(STATES)
        area      = round(random.uniform(0.5, 200.0), 2)
        rainfall  = round(random.uniform(200, 3000), 1)
        temp      = round(random.uniform(10, 45), 1)
        fertilizer = round(random.uniform(10, 500), 1)
        pesticide  = round(random.uniform(0.1, 50.0), 2)
        yield_val  = _simulate_yield(crop, rainfall, temp, fertilizer, pesticide)

        records.append({
            "Crop":            crop,
            "Season":          season,
            "State":           state,
            "Area":            area,
            "Annual_Rainfall": rainfall,
            "Temperature":     temp,
            "Fertilizer":      fertilizer,
            "Pesticide":       pesticide,
            "Yield":           yield_val,
        })

    df = pd.DataFrame(records)

    # Add a few intentional outliers to stress-test the model
    outlier_idx = df.sample(frac=0.01).index
    df.loc[outlier_idx, "Yield"] *= random.uniform(1.5, 2.0)

    os.makedirs(os.path.dirname(YIELD_DATA_PATH), exist_ok=True)
    df.to_csv(YIELD_DATA_PATH, index=False)
    print(f"[DataGen] [OK] {len(df)} rows saved -> {YIELD_DATA_PATH}")
    print(df.describe().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CropAI — Data Generator")
    print("=" * 60)
    generate_disease_dataset(samples_per_class=30)
    generate_yield_dataset(n_rows=5000)
    print("\n[DataGen] ✅ All datasets generated successfully!")
