# tests/test_api.py — API Endpoint Tests for CropAI
# Run: python -m pytest tests/test_api.py -v

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHealthEndpoints:
    """Tests for health check and info endpoints."""

    def test_root_endpoint(self, api_client):
        """GET / should return status ok."""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data

    def test_health_endpoint(self, api_client):
        """GET /health should return system health info."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "models" in data

    def test_classes_endpoint(self, api_client):
        """GET /classes should return all 38 disease classes."""
        response = api_client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert data["total"] == 38
        assert len(data["classes"]) == 38

    def test_crops_endpoint(self, api_client):
        """GET /crops should return valid crop, season, and state lists."""
        response = api_client.get("/crops")
        assert response.status_code == 200
        data = response.json()
        assert "crops" in data
        assert "seasons" in data
        assert "states" in data
        assert len(data["crops"]) >= 10
        assert "Kharif" in data["seasons"]
        assert "Punjab" in data["states"]


class TestDiseaseEndpoint:
    """Tests for /predict/disease endpoint."""

    def test_disease_prediction_invalid_file(self, api_client):
        """POST /predict/disease with non-image should return 400."""
        response = api_client.post(
            "/predict/disease",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_disease_prediction_no_file(self, api_client):
        """POST /predict/disease without file should return 422."""
        response = api_client.post("/predict/disease")
        assert response.status_code == 422


class TestYieldEndpoint:
    """Tests for /predict/yield endpoint."""

    def test_yield_prediction_invalid_crop(self, api_client, invalid_yield_input):
        """POST /predict/yield with invalid crop should return 422."""
        response = api_client.post("/predict/yield", json=invalid_yield_input)
        assert response.status_code == 422

    def test_yield_prediction_missing_fields(self, api_client):
        """POST /predict/yield with missing fields should return 422."""
        response = api_client.post("/predict/yield", json={"crop": "Rice"})
        assert response.status_code == 422

    def test_yield_prediction_negative_area(self, api_client, sample_yield_input):
        """POST /predict/yield with negative area should return 422."""
        payload = sample_yield_input.copy()
        payload["area"] = -1.0
        response = api_client.post("/predict/yield", json=payload)
        assert response.status_code == 422

    def test_yield_prediction_extreme_temperature(self, api_client, sample_yield_input):
        """POST /predict/yield with extreme temperature (>60) should return 422."""
        payload = sample_yield_input.copy()
        payload["temperature"] = 100.0
        response = api_client.post("/predict/yield", json=payload)
        assert response.status_code == 422


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_docs_endpoint(self, api_client):
        """GET /docs should return Swagger UI."""
        response = api_client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint(self, api_client):
        """GET /redoc should return ReDoc UI."""
        response = api_client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self, api_client):
        """GET /openapi.json should return valid OpenAPI schema."""
        response = api_client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "/predict/disease" in data["paths"]
        assert "/predict/yield" in data["paths"]
