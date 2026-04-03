# yield_model.py — Crop Yield Prediction (XGBoost + Random Forest Ensemble)

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

from config import (
    YIELD_DATA_PATH, MODEL_DIR, RESULTS_DIR,
    YIELD_FEATURES, YIELD_TARGET, CATEGORICAL_COLS, NUMERICAL_COLS,
    XGBOOST_PARAMS, RF_PARAMS, SCALER_PATH, ENCODER_PATH, YIELD_MODEL_PATH,
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(path: str = YIELD_DATA_PATH):
    """Load CSV, clean, engineer features, return X, y, and encoders."""
    print(f"\n[YieldModel] Loading data from {path}")
    df = pd.read_csv(path)
    print(f"[YieldModel] Shape: {df.shape}")
    print(df.head(3).to_string())

    # ── Basic cleaning ────────────────────────────────────────────────────────
    df = df.dropna(subset=[YIELD_TARGET])
    df = df.dropna(subset=YIELD_FEATURES)

    # Remove extreme outliers (> 3 IQR)
    Q1, Q3 = df[YIELD_TARGET].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df[YIELD_TARGET] >= Q1 - 3 * IQR) & (df[YIELD_TARGET] <= Q3 + 3 * IQR)]

    # ── Feature engineering ───────────────────────────────────────────────────
    df["Rainfall_per_Area"]    = df["Annual_Rainfall"] / (df["Area"] + 1e-6)
    df["Fertilizer_per_Area"]  = df["Fertilizer"]      / (df["Area"] + 1e-6)
    df["Pesticide_per_Area"]   = df["Pesticide"]        / (df["Area"] + 1e-6)
    df["Fertilizer_Pesticide"] = df["Fertilizer"] * df["Pesticide"]
    df["Rain_Temp_Interact"]   = df["Annual_Rainfall"] * df["Temperature"] if "Temperature" in df.columns else 0

    print(f"[YieldModel] After cleaning: {df.shape}")

    # ── Encode categoricals ───────────────────────────────────────────────────
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Save encoders
    joblib.dump(encoders, ENCODER_PATH)

    # ── Build feature matrix ──────────────────────────────────────────────────
    num_cols = NUMERICAL_COLS + [
        "Rainfall_per_Area", "Fertilizer_per_Area",
        "Pesticide_per_Area", "Fertilizer_Pesticide",
    ]
    if "Temperature" in df.columns:
        num_cols.append("Temperature")
        num_cols.append("Rain_Temp_Interact")

    cat_enc_cols = [c + "_enc" for c in CATEGORICAL_COLS]
    all_features = cat_enc_cols + num_cols

    X = df[all_features].values.astype(np.float32)
    y = df[YIELD_TARGET].values.astype(np.float32)

    # Scale numeric features
    scaler = StandardScaler()
    X[:, len(cat_enc_cols):] = scaler.fit_transform(X[:, len(cat_enc_cols):])
    joblib.dump(scaler, SCALER_PATH)

    print(f"[YieldModel] Feature matrix: {X.shape} | Target range: "
          f"{y.min():.0f} – {y.max():.0f} kg/ha")
    return X, y, all_features, encoders, scaler, df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class YieldEnsemble:
    """
    Stacked ensemble: XGBoost + Random Forest → Ridge meta-learner
    """

    def __init__(self):
        self.xgb_model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        self.rf_model  = RandomForestRegressor(**RF_PARAMS)
        self.feature_names = None
        self.encoders      = None
        self.scaler        = None
        self._trained      = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print("[YieldEnsemble] Training XGBoost …")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50,
        )

        print("[YieldEnsemble] Training Random Forest …")
        self.rf_model.fit(X_train, y_train)

        self._trained = True
        return self

    def predict(self, X):
        assert self._trained, "Call fit() first"
        p_xgb = self.xgb_model.predict(X)
        p_rf  = self.rf_model.predict(X)
        # Simple average ensemble (weights tuned empirically)
        return 0.6 * p_xgb + 0.4 * p_rf

    def feature_importances(self):
        imp_xgb = self.xgb_model.feature_importances_
        imp_rf  = self.rf_model.feature_importances_
        return 0.6 * imp_xgb + 0.4 * imp_rf


def evaluate(y_true, y_pred, label=""):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100

    print(f"\n  {'─'*40}")
    print(f"  {label} Performance")
    print(f"  {'─'*40}")
    print(f"  MAE  : {mae:>10.2f} kg/ha")
    print(f"  RMSE : {rmse:>10.2f} kg/ha")
    print(f"  R²   : {r2:>10.4f}")
    print(f"  MAPE : {mape:>10.2f} %")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def train_yield_model(data_path: str = YIELD_DATA_PATH):
    print("\n" + "=" * 60)
    print("  CropAI — Yield Prediction Training")
    print("=" * 60)

    X, y, feature_names, encoders, scaler, df = load_and_preprocess(data_path)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"\n  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    ensemble = YieldEnsemble()
    ensemble.fit(X_train, y_train, X_val, y_val)
    ensemble.feature_names = feature_names
    ensemble.encoders      = encoders
    ensemble.scaler        = scaler

    # Evaluate
    train_metrics = evaluate(y_train, ensemble.predict(X_train), "Train")
    val_metrics   = evaluate(y_val,   ensemble.predict(X_val),   "Validation")
    test_metrics  = evaluate(y_test,  ensemble.predict(X_test),  "Test")

    # Cross-validation
    print("\n[YieldModel] 5-Fold Cross-Validation (XGBoost) …")
    cv_scores = cross_val_score(
        xgb.XGBRegressor(**XGBOOST_PARAMS), X, y,
        cv=5, scoring="r2", n_jobs=-1
    )
    print(f"  CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save model bundle
    bundle = {
        "ensemble":      ensemble,
        "feature_names": feature_names,
        "test_metrics":  test_metrics,
        "cv_r2_mean":    cv_scores.mean(),
    }
    joblib.dump(bundle, YIELD_MODEL_PATH)
    print(f"\n[YieldModel] [OK] Model saved -> {YIELD_MODEL_PATH}")

    # Plots
    _plot_predictions(y_test, ensemble.predict(X_test))
    _plot_feature_importance(ensemble.feature_importances(), feature_names)
    _plot_residuals(y_test, ensemble.predict(X_test))

    return ensemble, test_metrics


def _plot_predictions(y_true, y_pred):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.4, s=12, color="#4CAF50")
    lim = max(y_true.max(), y_pred.max()) * 1.05
    plt.plot([0, lim], [0, lim], "r--", lw=1.5)
    plt.xlabel("Actual Yield (kg/ha)")
    plt.ylabel("Predicted Yield (kg/ha)")
    plt.title("Actual vs Predicted Crop Yield")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "yield_actual_vs_predicted.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[YieldModel] [OK] Plot saved -> {out}")


def _plot_feature_importance(importances, feature_names):
    idx = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(9, 5))
    colors = ["#4CAF50" if i == idx[0] else "#81C784" for i in idx]
    plt.bar([feature_names[i] for i in idx], importances[idx], color=colors)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "yield_feature_importance.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[YieldModel] [OK] Feature importance plot -> {out}")


def _plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.4, s=10, color="#FF7043")
    plt.axhline(0, color="black", lw=1.5, linestyle="--")
    plt.xlabel("Predicted"); plt.ylabel("Residuals"); plt.title("Residual Plot")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=40, color="#42A5F5", edgecolor="white")
    plt.xlabel("Residuals"); plt.title("Residual Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "yield_residuals.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[YieldModel] [OK] Residual plots -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

class YieldPredictor:
    """Load trained ensemble and predict yield for new data."""

    def __init__(self, model_path: str = YIELD_MODEL_PATH,
                 scaler_path: str = SCALER_PATH,
                 encoder_path: str = ENCODER_PATH):
        for p in [model_path, scaler_path, encoder_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Model file not found: {p}\nRun train_yield_model() first."
                )

        bundle = joblib.load(model_path)
        self.ensemble      = bundle["ensemble"]
        self.feature_names = bundle["feature_names"]
        self.scaler        = joblib.load(scaler_path)
        self.encoders      = joblib.load(encoder_path)
        print("[YieldPredictor] Model loaded")

    def predict(self, crop: str, season: str, state: str,
                area: float, rainfall: float, temperature: float,
                fertilizer: float, pesticide: float) -> dict:
        """
        Returns predicted yield (kg/ha) with grade and recommendations.
        """
        enc = self.encoders
        cat_enc = [
            enc["Crop"].transform([crop])[0]    if crop   in enc["Crop"].classes_   else 0,
            enc["Season"].transform([season])[0] if season in enc["Season"].classes_ else 0,
            enc["State"].transform([state])[0]   if state  in enc["State"].classes_  else 0,
        ]

        rain_per_area = rainfall   / (area + 1e-6)
        fert_per_area = fertilizer / (area + 1e-6)
        pest_per_area = pesticide  / (area + 1e-6)
        fert_pest     = fertilizer * pesticide
        rain_temp     = rainfall * temperature

        num_raw = np.array([[
            area, rainfall, fertilizer, pesticide,
            rain_per_area, fert_per_area, pest_per_area,
            fert_pest, temperature, rain_temp
        ]], dtype=np.float32)

        # Only scale numerical columns
        n_cat = len(cat_enc)
        num_scaled = self.scaler.transform(num_raw)[0]

        X = np.array([cat_enc + list(num_scaled)], dtype=np.float32)
        pred = float(self.ensemble.predict(X)[0])
        pred = max(0, pred)

        total = pred * area
        grade = (
            "Excellent" if pred > 4000 else
            "Good"      if pred > 2500 else
            "Average"   if pred > 1200 else
            "Below Average"
        )

        return {
            "yield_per_ha":  round(pred, 2),
            "total_yield":   round(total, 2),
            "unit":          "kg/ha",
            "area_ha":       area,
            "yield_grade":   grade,
            "crop":          crop,
            "season":        season,
            "state":         state,
            "inputs": {
                "annual_rainfall_mm": rainfall,
                "temperature_c":      temperature,
                "fertilizer_kg_ha":   fertilizer,
                "pesticide_kg_ha":    pesticide,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(YIELD_DATA_PATH):
        print("Crop yield CSV not found — generating synthetic data …")
        from data_generator import generate_yield_dataset
        generate_yield_dataset(n_rows=5000)

    ensemble, metrics = train_yield_model()

    print("\n[Test] Running inference …")
    predictor = YieldPredictor()
    result = predictor.predict(
        crop="Rice", season="Kharif", state="Punjab",
        area=5.0, rainfall=1400, temperature=28,
        fertilizer=150, pesticide=2.0
    )
    print(json.dumps(result, indent=2))
