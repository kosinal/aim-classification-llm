"""Tests for main FastAPI application."""

from fastapi.testclient import TestClient

from aim.main import app


client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Flagging Classifier API", "version": "0.1.0"}


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "healthy"
    assert "models_loaded" in response_data
    assert "model_count" in response_data
    assert "project_ids" in response_data
    assert isinstance(response_data["models_loaded"], bool)
    assert isinstance(response_data["model_count"], int)
    assert isinstance(response_data["project_ids"], list)
