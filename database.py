# database.py — SQLite Prediction History for CropAI
# Logs disease and yield predictions for history tracking and analytics.

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict

from config import BASE_DIR

DB_PATH = os.path.join(BASE_DIR, "data", "predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATABASE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Get SQLite connection with row factory for dict-like results."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """
    Initialize the SQLite database with prediction tables.
    Safe to call multiple times (uses CREATE IF NOT EXISTS).
    """
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS disease_predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            filename    TEXT,
            file_size   INTEGER,
            predicted_class TEXT NOT NULL,
            confidence  REAL NOT NULL,
            is_healthy  INTEGER NOT NULL,
            crop        TEXT,
            disease     TEXT,
            severity    TEXT,
            spread_risk TEXT,
            treatment   TEXT,
            inference_ms REAL,
            top_predictions TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS yield_predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL,
            crop         TEXT NOT NULL,
            season       TEXT NOT NULL,
            state        TEXT NOT NULL,
            area_ha      REAL NOT NULL,
            rainfall_mm  REAL NOT NULL,
            temperature_c REAL NOT NULL,
            fertilizer_kg REAL NOT NULL,
            pesticide_kg  REAL NOT NULL,
            yield_per_ha  REAL NOT NULL,
            total_yield   REAL NOT NULL,
            yield_grade   TEXT,
            inference_ms  REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            model_type  TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metadata    TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"[Database] [OK] Initialized -> {DB_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOGGING PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def log_disease_prediction(result: dict, filename: str = None,
                            file_size: int = None):
    """
    Log a disease prediction to the database.

    Args:
        result:    Prediction result dict from DiseasePredictor
        filename:  Original filename of uploaded image
        file_size: File size in bytes
    """
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO disease_predictions
            (timestamp, filename, file_size, predicted_class, confidence,
             is_healthy, crop, disease, severity, spread_risk, treatment,
             inference_ms, top_predictions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        filename,
        file_size,
        result.get("predicted_class", ""),
        result.get("confidence", 0.0),
        1 if result.get("is_healthy", False) else 0,
        result.get("crop", ""),
        result.get("disease", ""),
        result.get("severity", ""),
        result.get("spread_risk", ""),
        result.get("treatment", ""),
        result.get("inference_time_ms", 0.0),
        json.dumps(result.get("top_predictions", [])),
    ))

    conn.commit()
    conn.close()


def log_yield_prediction(request: dict, result: dict):
    """
    Log a yield prediction to the database.

    Args:
        request: Input request dict (crop, season, state, etc.)
        result:  Prediction result dict from YieldPredictor
    """
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO yield_predictions
            (timestamp, crop, season, state, area_ha, rainfall_mm,
             temperature_c, fertilizer_kg, pesticide_kg,
             yield_per_ha, total_yield, yield_grade, inference_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        request.get("crop", ""),
        request.get("season", ""),
        request.get("state", ""),
        request.get("area", 0),
        request.get("rainfall", 0),
        request.get("temperature", 0),
        request.get("fertilizer", 0),
        request.get("pesticide", 0),
        result.get("yield_per_ha", 0),
        result.get("total_yield", 0),
        result.get("yield_grade", ""),
        result.get("inference_time_ms", 0),
    ))

    conn.commit()
    conn.close()


def log_model_metrics(model_type: str, metrics: dict, metadata: dict = None):
    """
    Log training metrics to the database.

    Args:
        model_type: 'disease' or 'yield'
        metrics:    Dict of metric_name → metric_value
        metadata:   Optional extra info (epochs, dataset size, etc.)
    """
    conn = _get_connection()
    cursor = conn.cursor()
    ts = datetime.now().isoformat()
    meta_str = json.dumps(metadata) if metadata else None

    for name, value in metrics.items():
        cursor.execute("""
            INSERT INTO model_metrics
                (timestamp, model_type, metric_name, metric_value, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (ts, model_type, name, float(value), meta_str))

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RETRIEVING HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_disease_history(limit: int = 50) -> List[Dict]:
    """Get recent disease predictions from the database."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM disease_predictions
        ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_yield_history(limit: int = 50) -> List[Dict]:
    """Get recent yield predictions from the database."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM yield_predictions
        ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_prediction_stats() -> Dict:
    """
    Get aggregate statistics of all predictions.

    Returns:
        dict with total counts, averages, and breakdowns
    """
    conn = _get_connection()
    cursor = conn.cursor()

    stats = {}

    # Disease stats
    cursor.execute("SELECT COUNT(*) as total FROM disease_predictions")
    stats["total_disease_predictions"] = cursor.fetchone()["total"]

    cursor.execute("""
        SELECT predicted_class, COUNT(*) as count
        FROM disease_predictions
        GROUP BY predicted_class
        ORDER BY count DESC LIMIT 10
    """)
    stats["top_diseases"] = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT AVG(confidence) as avg_conf FROM disease_predictions")
    row = cursor.fetchone()
    stats["avg_disease_confidence"] = round(row["avg_conf"], 2) if row["avg_conf"] else 0

    # Yield stats
    cursor.execute("SELECT COUNT(*) as total FROM yield_predictions")
    stats["total_yield_predictions"] = cursor.fetchone()["total"]

    cursor.execute("""
        SELECT crop, AVG(yield_per_ha) as avg_yield, COUNT(*) as count
        FROM yield_predictions
        GROUP BY crop
        ORDER BY count DESC
    """)
    stats["yield_by_crop"] = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return stats


def clear_history(table: str = "all"):
    """
    Clear prediction history.

    Args:
        table: 'disease', 'yield', 'metrics', or 'all'
    """
    conn = _get_connection()
    cursor = conn.cursor()

    if table in ("disease", "all"):
        cursor.execute("DELETE FROM disease_predictions")
    if table in ("yield", "all"):
        cursor.execute("DELETE FROM yield_predictions")
    if table in ("metrics", "all"):
        cursor.execute("DELETE FROM model_metrics")

    conn.commit()
    conn.close()
    print(f"[Database] [OK] Cleared history: {table}")


# ─────────────────────────────────────────────────────────────────────────────
# Initialize database on import
init_database()
