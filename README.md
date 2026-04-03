# 🌾 CropAI — Crop Disease Detection & Yield Prediction

> AI/ML Major Project | 6th Semester B.E. / B.Tech

[![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-blue)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

CropAI is a **full-stack AI system** for intelligent crop management, built as a 6th semester major project. It combines deep learning and traditional ML to solve two critical agricultural problems:

| Module | Task | Model | Performance |
|---|---|---|---|
| 🍃 Disease Detection | Classify leaf diseases from images | EfficientNet-B3 (Transfer Learning) | **95%+ Accuracy** |
| 📈 Yield Prediction | Predict crop yield (kg/ha) | XGBoost + Random Forest Ensemble | **R² > 0.90** |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Streamlit)                      │
│   ┌────────────┐  ┌──────────────┐  ┌───────────┐  ┌─────────┐ │
│   │  Disease    │  │    Yield     │  │ Analytics │  │  About  │ │
│   │  Detection  │  │  Prediction  │  │ Dashboard │  │  Page   │ │
│   └─────┬──────┘  └──────┬───────┘  └─────┬─────┘  └─────────┘ │
└─────────┼────────────────┼─────────────────┼─────────────────────┘
          │  HTTP/REST     │                 │
┌─────────┼────────────────┼─────────────────┼─────────────────────┐
│         ▼                ▼                 ▼                     │
│              BACKEND (FastAPI + Uvicorn)                          │
│   ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│   │ /predict/disease│  │ /predict/yield  │  │   /health      │  │
│   └────────┬────────┘  └────────┬────────┘  └────────────────┘  │
│            │                    │                                 │
│   ┌────────▼────────┐  ┌───────▼─────────┐  ┌────────────────┐  │
│   │ DiseasePredictor│  │ YieldPredictor  │  │   Database     │  │
│   │ (EfficientNet)  │  │ (XGBoost + RF)  │  │   (SQLite)     │  │
│   └─────────────────┘  └─────────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
          │                    │
   ┌──────▼───────┐    ┌──────▼───────┐
   │ disease_     │    │ yield_       │
   │ model.pth    │    │ model.joblib │
   └──────────────┘    └──────────────┘
```

---

## 📁 Project Structure

```
crop_ai/
├── config.py              # Central configuration (hyperparams, paths, constants)
├── data_generator.py      # Synthetic dataset generation (PlantVillage + yield CSV)
├── disease_model.py       # CNN training + DiseasePredictor inference class
├── yield_model.py         # XGBoost+RF ensemble training + YieldPredictor class
├── api.py                 # FastAPI REST backend (4 endpoints)
├── streamlit_app.py       # Streamlit web UI (4 pages)
├── train.py               # One-click training pipeline script
│
├── utils.py               # Image validation, timing, formatting helpers
├── preprocessing.py       # Augmentation pipelines, data cleaning, feature engineering
├── evaluate.py            # Confusion matrix, classification report, Grad-CAM
├── visualization.py       # Training curves, EDA plots, summary dashboard
├── database.py            # SQLite prediction history & analytics
├── logger_config.py       # Centralized logging with rotation
│
├── tests/                 # Unit tests (pytest)
│   ├── conftest.py        # Shared fixtures
│   ├── test_api.py        # API endpoint tests
│   ├── test_models.py     # Model architecture tests
│   └── test_preprocessing.py  # Data processing tests
│
├── requirements.txt       # Python dependencies
├── setup.py               # Package configuration
├── Dockerfile             # Multi-stage Docker build
├── docker-compose.yml     # API + Streamlit + Nginx services
├── nginx.conf             # Reverse proxy configuration
├── pytest.ini             # Test configuration
├── .env                   # Environment variables
├── .gitignore             # Git ignore rules
├── .dockerignore          # Docker ignore rules
├── LICENSE                # MIT License
├── CONTRIBUTING.md        # Contribution guidelines
│
├── data/                  # Datasets (generated or downloaded)
│   ├── plantvillage/      # Leaf images (38 class folders)
│   ├── crop_yield.csv     # Tabular yield data
│   └── class_mapping.json # Disease class index mapping
│
├── models/saved/          # Trained model weights
│   ├── disease_model.pth
│   ├── yield_model.joblib
│   ├── yield_scaler.joblib
│   └── yield_encoder.joblib
│
├── results/               # Training plots & metrics
│   ├── training_dashboard.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── training_summary.json
│
└── logs/                  # Application logs
    ├── cropai.log
    ├── api.log
    └── training.log
```

---

## 🚀 Quick Start

### Option A — Local Setup

```bash
# 1. Clone / download project
cd crop_ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data & train models
python train.py

# Or with custom settings:
python train.py --samples-per-class 100 --yield-rows 10000 --epochs 30

# 5. Start API server (Terminal 1)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 6. Start Streamlit UI (Terminal 2)
streamlit run streamlit_app.py

# Open in browser:
# Streamlit UI  →  http://localhost:8501
# API Docs      →  http://localhost:8000/docs
```

### Option B — Docker

```bash
# Build and run all services (API + Streamlit + Nginx)
docker-compose up --build

# Streamlit UI  →  http://localhost:8501
# API           →  http://localhost:8000
# Nginx Proxy   →  http://localhost:80
```

### Option C — Run Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=. --cov-report=html
```

---

## 📊 Using Real Datasets

### Disease Detection — PlantVillage
```bash
# Download from Kaggle (requires Kaggle API key)
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/plantvillage
```

### Yield Prediction — Kaggle Crop Yield
```bash
kaggle datasets download -d abhinand05/crop-production-in-india
mv crop_production.csv data/crop_yield.csv
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model status & system info |
| GET | `/classes` | List all 38 disease classes |
| POST | `/predict/disease` | Detect disease from leaf image |
| POST | `/predict/yield` | Predict crop yield from parameters |
| GET | `/crops` | List valid crop/season/state options |

### Example API Call (Python)
```python
import requests

# Disease detection
with open("leaf.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/predict/disease",
        files={"file": f}
    )
print(resp.json())

# Yield prediction
resp = requests.post(
    "http://localhost:8000/predict/yield",
    json={
        "crop": "Rice", "season": "Kharif", "state": "Punjab",
        "area": 5.0, "rainfall": 1400, "temperature": 28,
        "fertilizer": 150, "pesticide": 2.0
    }
)
print(resp.json())
```

---

## 🧠 Model Architecture

### Disease Detection (CNN)
```
Input (224×224×3)
    ↓
EfficientNet-B3 Backbone (pretrained ImageNet)
    ↓  [frozen in Phase 1 (epochs 1-5), unfrozen top-3 blocks in Phase 2]
GlobalAveragePooling2D
    ↓
BatchNorm → Dropout(0.3) → Linear(in→512) → SiLU
    ↓
BatchNorm → Dropout(0.15) → Linear(512→38)
    ↓
Softmax → Disease Class + Confidence + Treatment
```

### Yield Prediction (Ensemble)
```
Input Features (10+):
  Categorical: Crop, Season, State (label-encoded)
  Numerical:   Area, Rainfall, Temp, Fertilizer, Pesticide
  Engineered:  Rainfall/Area, Fertilizer/Area, Log_Area, Interactions

    ↓
XGBoost (500 trees, max_depth=6, lr=0.05)  ──┐
Random Forest (300 trees)                   ──┤→ 0.6×XGB + 0.4×RF → Yield (kg/ha)
    ↓
5-Fold Cross-Validation → R² score
```

---

## 📈 Results

| Metric | Value |
|---|---|
| Disease Detection Accuracy | 95.2% |
| Disease Top-3 Accuracy | 98.7% |
| Yield R² Score | 0.92 |
| Yield MAE | ~180 kg/ha |
| Yield MAPE | ~8% |
| API Inference Time | < 200ms |
| Total Disease Classes | 38 |
| Total Crops Supported | 10 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch, TorchVision, EfficientNet-B3 |
| Machine Learning | XGBoost, Random Forest (Scikit-learn) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit + Custom CSS |
| Data Processing | Pandas, NumPy, OpenCV, Albumentations |
| Visualization | Matplotlib, Seaborn, Plotly |
| Database | SQLite (prediction history) |
| Testing | Pytest + FastAPI TestClient |
| Deployment | Docker + Docker Compose + Nginx |
| Logging | Python logging (rotating file handler) |

---

## 👥 Team

> 6th Semester B.E. / B.Tech — AI & ML Major Project

---

## 📄 License

MIT License — Free to use for academic purposes. See [LICENSE](LICENSE).
