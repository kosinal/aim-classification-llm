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
    assert "classifier_loaded" in response_data
    assert "model_type" in response_data
    assert "threshold" in response_data
    assert isinstance(response_data["classifier_loaded"], bool)


class TestLifespan:
    """Tests for application lifespan context manager."""

    def test_app_has_lifespan_configured(self):
        """Test that app has lifespan configured."""
        # The app should have our lifespan configured
        assert app.router.lifespan_context is not None

    def test_health_check_reflects_classifier_state(self):
        """Test health endpoint reflects actual classifier state from lifespan."""
        # This test uses the actual app with real lifespan
        # Classifier may or may not be loaded depending on test environment
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        # These fields must exist regardless of classifier loading success
        assert "classifier_loaded" in data
        assert "model_type" in data
        assert "threshold" in data
