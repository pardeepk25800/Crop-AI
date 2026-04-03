# api.py — FastAPI Backend for CropAI
# Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload

import os
import io
import json
import time
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import database
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CropAI-API")

app = FastAPI(
    title       = "CropAI — Disease & Yield Intelligence API",
    description = "AI-powered crop disease detection and yield prediction",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─── Lazy-load models ────────────────────────────────────────────────────────
_disease_predictor = None
_yield_predictor   = None

def get_disease_predictor():
    global _disease_predictor
    if _disease_predictor is None:
        from disease_model import DiseasePredictor
        _disease_predictor = DiseasePredictor()
    return _disease_predictor

def get_yield_predictor():
    global _yield_predictor
    if _yield_predictor is None:
        from yield_model import YieldPredictor
        _yield_predictor = YieldPredictor()
    return _yield_predictor


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class YieldRequest(BaseModel):
    crop:        str   = Field(..., example="Rice")
    season:      str   = Field(..., example="Kharif")
    state:       str   = Field(..., example="Punjab")
    area:        float = Field(..., gt=0,  le=10000, example=5.0)
    rainfall:    float = Field(..., gt=0,  le=5000,  example=1200.0,
                               description="Annual rainfall in mm")
    temperature: float = Field(..., gt=-10, lt=60,  example=28.0,
                               description="Average temperature in °C")
    fertilizer:  float = Field(..., ge=0,  le=1000,  example=120.0,
                               description="Fertilizer usage kg/ha")
    pesticide:   float = Field(..., ge=0,  le=100,   example=1.5,
                               description="Pesticide usage kg/ha")

    @validator("crop")
    def crop_must_be_valid(cls, v):
        valid = ["Rice","Wheat","Maize","Cotton","Sugarcane",
                 "Soybean","Potato","Tomato","Groundnut","Bajra"]
        if v not in valid:
            raise ValueError(f"Crop must be one of {valid}")
        return v


class DiseaseResponse(BaseModel):
    predicted_class:  str
    confidence:       float
    is_healthy:       bool
    crop:             str
    disease:          str
    top_predictions:  list
    treatment:        str
    severity:         str
    spread_risk:      str
    inference_time_ms: float


class YieldResponse(BaseModel):
    yield_per_ha:  float
    total_yield:   float
    unit:          str
    area_ha:       float
    yield_grade:   str
    crop:          str
    season:        str
    state:         str
    recommendations: list
    inference_time_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# DISEASE KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_INFO = {
    "blight": {
        "treatment":   "Apply copper-based fungicide (Bordeaux mixture). Remove infected leaves immediately. Avoid overhead irrigation. Use resistant varieties.",
        "severity":    "High",
        "spread_risk": "Very High",
    },
    "rust": {
        "treatment":   "Apply triazole-based fungicide (Propiconazole). Scout fields regularly. Ensure proper plant spacing for air circulation.",
        "severity":    "Moderate",
        "spread_risk": "High",
    },
    "scab": {
        "treatment":   "Apply captan or sulfur-based fungicide during wet periods. Rake and destroy fallen leaves. Plant scab-resistant varieties.",
        "severity":    "Moderate",
        "spread_risk": "Medium",
    },
    "rot": {
        "treatment":   "Apply myclobutanil or captan fungicide. Remove infected plant material. Improve drainage and air circulation.",
        "severity":    "High",
        "spread_risk": "High",
    },
    "spot": {
        "treatment":   "Apply chlorothalonil or mancozeb every 7–10 days. Maintain proper plant nutrition. Avoid excessive nitrogen.",
        "severity":    "Moderate",
        "spread_risk": "Medium",
    },
    "mildew": {
        "treatment":   "Apply sulfur-based fungicide or potassium bicarbonate. Improve air circulation. Avoid overhead watering.",
        "severity":    "Low",
        "spread_risk": "Medium",
    },
    "virus": {
        "treatment":   "No chemical cure. Remove and destroy infected plants. Control insect vectors (aphids, whiteflies). Use resistant varieties.",
        "severity":    "Very High",
        "spread_risk": "High",
    },
    "healthy": {
        "treatment":   "No treatment needed. Continue standard crop management practices. Monitor regularly for early signs of infection.",
        "severity":    "None",
        "spread_risk": "None",
    },
    "default": {
        "treatment":   "Consult your local agricultural extension officer for precise diagnosis and treatment.",
        "severity":    "Moderate",
        "spread_risk": "Medium",
    },
}

YIELD_RECOMMENDATIONS = {
    "Rice":      ["Transplant at 25-day-old seedlings.", "Maintain 2–5 cm water depth during tillering.", "Apply nitrogen in 3 splits for best uptake."],
    "Wheat":     ["Sow Nov 1–15 for Rabi season.", "Irrigate at crown root initiation and jointing.", "Use zero-till sowing to conserve moisture."],
    "Maize":     ["Maintain 60×20 cm plant spacing.", "Irrigate at tasseling and silking stages.", "Use hybrid varieties for 20–30% higher yield."],
    "Cotton":    ["Use drip irrigation for 40% water saving.", "Monitor bollworm with pheromone traps weekly.", "Topping at 90 days promotes boll set."],
    "Sugarcane": ["Use trench planting for better root development.", "Apply press mud compost for soil health.", "Intercrop with legumes in first 3 months."],
    "Soybean":   ["Inoculate seeds with Rhizobium culture.", "Avoid excess nitrogen — soybean is self-fixing.", "Apply boron + molybdenum foliar spray at flowering."],
    "Potato":    ["Use certified disease-free seed tubers.", "Earth up at 30 and 60 days after planting.", "Apply potassium at 120 kg/ha for tuber quality."],
    "Tomato":    ["Stake plants to prevent soil contact.", "Apply calcium nitrate to prevent blossom end rot.", "Use mulching to maintain soil moisture."],
    "Groundnut": ["Inoculate with Bradyrhizobium for nitrogen fixation.", "Apply gypsum at 250 kg/ha at pegging stage.", "Avoid waterlogging — especially at pod development."],
    "Bajra":     ["Thin to one plant per hill at 2-leaf stage.", "Apply phosphorus fertilizer at sowing.", "Harvest when 85% of grains are hard and mature."],
}


def _get_disease_info(class_name: str) -> dict:
    name_lower = class_name.lower()
    for key, info in DISEASE_INFO.items():
        if key in name_lower:
            return info
    return DISEASE_INFO["default"]


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "CropAI API is running 🌾"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":     "healthy",
        "timestamp":  time.time(),
        "models": {
            "disease": "loaded" if _disease_predictor else "not loaded",
            "yield":   "loaded" if _yield_predictor   else "not loaded",
        }
    }


@app.get("/classes", tags=["Disease"])
def list_disease_classes():
    """Return all disease classes the model can predict."""
    from config import DISEASE_CLASSES
    return {"classes": DISEASE_CLASSES, "total": len(DISEASE_CLASSES)}


@app.post("/predict/disease", response_model=DiseaseResponse, tags=["Disease"])
async def predict_disease(
    file: UploadFile = File(..., description="Leaf image (JPG/PNG/WEBP)"),
    top_k: int = Query(3, ge=1, le=10),
):
    """
    Upload a leaf image and get disease prediction with confidence scores.

    - **file**: Leaf image file (JPG, PNG, WEBP) max 10MB
    - **top_k**: Number of top predictions to return (1–10)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be < 10MB")

    try:
        predictor = get_disease_predictor()
        t0     = time.perf_counter()
        result = predictor.predict(contents, top_k=top_k)
        elapsed = (time.perf_counter() - t0) * 1000

        info = _get_disease_info(result["predicted_class"])
        logger.info(f"Disease prediction: {result['predicted_class']} "
                    f"({result['confidence']:.1f}%) in {elapsed:.0f}ms")

        result_dict = {
            **result,
            "treatment":         info["treatment"],
            "severity":          info["severity"],
            "spread_risk":       info["spread_risk"],
            "inference_time_ms": round(elapsed, 2),
        }
        
        # Log to database
        try:
            database.log_disease_prediction(
                result=result_dict, 
                filename=file.filename, 
                file_size=len(contents)
            )
        except Exception as db_err:
            logger.error(f"Failed to log disease prediction: {db_err}")

        return result_dict

    except Exception as e:
        logger.error(f"Disease prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/yield", response_model=YieldResponse, tags=["Yield"])
def predict_yield(request: YieldRequest):
    """
    Predict crop yield based on environmental and field parameters.

    Returns predicted yield in kg/ha, total harvest, grade, and recommendations.
    """
    try:
        predictor = get_yield_predictor()
        t0     = time.perf_counter()
        result = predictor.predict(
            crop        = request.crop,
            season      = request.season,
            state       = request.state,
            area        = request.area,
            rainfall    = request.rainfall,
            temperature = request.temperature,
            fertilizer  = request.fertilizer,
            pesticide   = request.pesticide,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        recs = YIELD_RECOMMENDATIONS.get(request.crop, ["Follow standard agronomic practices."])
        logger.info(f"Yield prediction: {result['yield_per_ha']:.0f} kg/ha "
                    f"for {request.crop} in {elapsed:.0f}ms")

        result_dict = {
            **result,
            "recommendations":  recs,
            "inference_time_ms": round(elapsed, 2),
        }
        
        # Log to database
        try:
            database.log_yield_prediction(
                request=request.model_dump() if hasattr(request, 'model_dump') else request.dict(),
                result=result_dict
            )
        except Exception as db_err:
            logger.error(f"Failed to log yield prediction: {db_err}")

        return result_dict

    except Exception as e:
        logger.error(f"Yield prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/crops", tags=["Yield"])
def list_crops():
    return {
        "crops":   list(YIELD_RECOMMENDATIONS.keys()),
        "seasons": ["Kharif", "Rabi", "Zaid", "Whole Year"],
        "states":  [
            "Punjab", "Haryana", "Uttar Pradesh", "Maharashtra",
            "Andhra Pradesh", "Karnataka", "Tamil Nadu", "West Bengal",
            "Madhya Pradesh", "Rajasthan", "Bihar", "Gujarat",
            "Odisha", "Assam", "Jharkhand",
        ],
    }


@app.get("/stats", tags=["Analytics"])
def get_stats():
    """Get aggregated prediction statistics."""
    try:
        return database.get_prediction_stats()
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", tags=["Analytics"])
def get_history(limit: int = Query(50, ge=1, le=500)):
    """Get combined recent prediction history."""
    try:
        disease = database.get_disease_history(limit)
        for d in disease:
            d["analysis_type"] = "Disease"
            
        yield_hist = database.get_yield_history(limit)
        for y in yield_hist:
            y["analysis_type"] = "Yield"
            
        combined = disease + yield_hist
        # Sort combined list by timestamp descending
        combined.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"history": combined[:limit]}
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    print(f"\n🌾 CropAI API starting at http://{API_HOST}:{API_PORT}")
    print(f"   Docs: http://{API_HOST}:{API_PORT}/docs")
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
