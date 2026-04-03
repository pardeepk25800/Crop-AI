# config.py — Central configuration for CropAI project

import os

# Load environment variables from .env file (if it exists)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — use defaults

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_DIR       = os.path.join(BASE_DIR, "models", "saved")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

DISEASE_DATA_DIR = os.path.join(DATA_DIR, "plantvillage")   # image dataset
YIELD_DATA_PATH  = os.path.join(DATA_DIR, "crop_yield.csv") # tabular dataset

DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.pth")
YIELD_MODEL_PATH   = os.path.join(MODEL_DIR, "yield_model.joblib")
SCALER_PATH        = os.path.join(MODEL_DIR, "yield_scaler.joblib")
ENCODER_PATH       = os.path.join(MODEL_DIR, "yield_encoder.joblib")

# Create directories
for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── DISEASE DETECTION ────────────────────────────────────────────────────────
IMAGE_SIZE   = 224          # EfficientNet input size
BATCH_SIZE   = 32
EPOCHS       = 5
LR           = 1e-4
LR_PATIENCE  = 5            # ReduceLROnPlateau patience
DROPOUT      = 0.3
TRAIN_SPLIT  = 0.8
VAL_SPLIT    = 0.1          # of total; remaining 0.1 → test

BACKBONE     = "efficientnet_b3"   # or "resnet50", "mobilenet_v3"
NUM_WORKERS  = 0
PIN_MEMORY   = True

# Normalisation (ImageNet stats)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# 38 PlantVillage disease classes
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
NUM_CLASSES = len(DISEASE_CLASSES)

# ─── YIELD PREDICTION ─────────────────────────────────────────────────────────
YIELD_FEATURES    = ["Crop", "Season", "State", "Area", "Annual_Rainfall",
                     "Fertilizer", "Pesticide"]
YIELD_TARGET      = "Yield"
CATEGORICAL_COLS  = ["Crop", "Season", "State"]
NUMERICAL_COLS    = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]

XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth":    None,
    "random_state": 42,
    "n_jobs":       -1,
}

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
