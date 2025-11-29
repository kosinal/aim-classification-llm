"""Tests for API routes."""

from unittest.mock import MagicMock

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aim.models import FlagClassifier
from aim.routes import router
from aim.schemas import AssessRequest, AssessResponse


@pytest.fixture
def app_with_models():
    """Create FastAPI app with mocked models for testing."""
    app = FastAPI()
    app.include_router(router)

    # Create mock models
    mock_model_1 = MagicMock(spec=FlagClassifier)
    mock_model_2 = MagicMock(spec=FlagClassifier)

    # Setup app state with models
    app.state.models = {
        "1": mock_model_1,
        "2": mock_model_2,
    }

    return app, {"1": mock_model_1, "2": mock_model_2}


@pytest.fixture
def app_without_models():
    """Create FastAPI app without models for testing."""
    app = FastAPI()
    app.include_router(router)
    app.state.models = None
    return app


class TestAssessContentEndpoint:
    """Tests for /api/project/{project_id}/assess endpoint."""

    def test_assess_content_success_positive_prediction(self, app_with_models):
        """Test successful content assessment with positive prediction."""
        app, models = app_with_models
        client = TestClient(app)

        # Mock prediction response
        mock_prediction = MagicMock()
        mock_prediction.reasoning = "Content is highly relevant to project goals"
        mock_prediction.prediction_score = "0.95"
        mock_prediction.prediction = "positive"
        models["1"].return_value = mock_prediction

        # Make request
        response = client.post(
            "/api/project/1/assess",
            json={
                "summary": "Important project update about budget allocation",
                "author": "John Smith",
                "title": "Budget Allocation Update",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["recommend"] is True
        assert data["recommendation_score"] == 0.95
        assert data["reasoning"] == "Content is highly relevant to project goals"
        assert data["project_id"] == "1"

    def test_assess_content_success_negative_prediction(self, app_with_models):
        """Test successful content assessment with negative prediction."""
        app, models = app_with_models
        client = TestClient(app)

        # Mock prediction response
        mock_prediction = MagicMock()
        mock_prediction.reasoning = "Content not relevant to project"
        mock_prediction.prediction_score = "0.12"
        mock_prediction.prediction = "negative"
        models["2"].return_value = mock_prediction

        # Make request
        response = client.post(
            "/api/project/2/assess",
            json={
                "summary": "Irrelevant spam content",
                "author": "Spam Bot",
                "title": "Spam Article",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["recommend"] is False
        assert data["recommendation_score"] == 0.12
        assert data["reasoning"] == "Content not relevant to project"
        assert data["project_id"] == "2"

    def test_assess_content_with_string_score_parsing(self, app_with_models):
        """Test that string scores are properly converted to float."""
        app, models = app_with_models
        client = TestClient(app)

        # Mock prediction with string score
        mock_prediction = MagicMock()
        mock_prediction.reasoning = "Analysis complete"
        mock_prediction.prediction_score = "0.75"  # String instead of float
        mock_prediction.prediction = "positive"
        models["1"].return_value = mock_prediction

        response = client.post(
            "/api/project/1/assess",
            json={"summary": "Test content", "author": "Test Author", "title": "Test Title"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["recommendation_score"] == 0.75  # Should be converted to float

    def test_assess_content_with_invalid_score_fallback(self, app_with_models):
        """Test fallback logic when score parsing fails."""
        app, models = app_with_models
        client = TestClient(app)

        # Mock prediction with unparseable score
        mock_prediction = MagicMock()
        mock_prediction.reasoning = "Analysis complete"
        mock_prediction.prediction_score = "invalid"  # Can't convert to float
        mock_prediction.prediction = "positive"
        models["1"].return_value = mock_prediction

        response = client.post(
            "/api/project/1/assess",
            json={"summary": "Test content", "author": "Test Author", "title": "Test Title"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["recommendation_score"] == 1.0  # Fallback for positive

    def test_assess_content_with_invalid_score_negative_fallback(self, app_with_models):
        """Test fallback logic for negative prediction when score parsing fails."""
        app, models = app_with_models
        client = TestClient(app)

        # Mock prediction with unparseable score
        mock_prediction = MagicMock()
        mock_prediction.reasoning = "Analysis complete"
        mock_prediction.prediction_score = "invalid"
        mock_prediction.prediction = "negative"
        models["1"].return_value = mock_prediction

        response = client.post(
            "/api/project/1/assess",
            json={"summary": "Test content", "author": "Test Author", "title": "Test Title"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["recommendation_score"] == 0.0  # Fallback for negative

    def test_assess_content_models_not_loaded(self, app_without_models):
        """Test error response when models are not loaded."""
        client = TestClient(app_without_models)

        response = client.post(
            "/api/project/1/assess",
            json={"summary": "Test content", "author": "Test Author", "title": "Test Title"},
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Models not loaded yet"

    def test_assess_content_project_not_found(self, app_with_models):
        """Test error response when project model is not found."""
        app, _ = app_with_models
        client = TestClient(app)

        response = client.post(
            "/api/project/999/assess",
            json={"summary": "Test content", "author": "Test Author", "title": "Test Title"},
        )

        assert response.status_code == 404
        assert "No model found for project 999" in response.json()["detail"]
        assert "Available projects" in response.json()["detail"]

    def test_assess_content_model_processing_error(self, app_with_models):
        """Test error response when model processing fails."""
        app, models = app_with_models
        client = TestClient(app)

        # Mock model to raise exception
        models["1"].side_effect = Exception("Model processing error")

        response = client.post(
            "/api/project/1/assess",
            json={"summary": "Test content", "author": "Test Author", "title": "Test Title"},
        )

        assert response.status_code == 500
        assert "Model processing error" in response.json()["detail"]

    def test_assess_content_validates_request_schema(self, app_with_models):
        """Test that request schema validation works."""
        app, _ = app_with_models
        client = TestClient(app)

        # Empty summary should fail validation
        response = client.post(
            "/api/project/1/assess",
            json={"summary": "", "author": "Test Author", "title": "Test Title"},
        )

        assert response.status_code == 422  # Validation error

    def test_assess_content_missing_summary_field(self, app_with_models):
        """Test error when summary field is missing."""
        app, _ = app_with_models
        client = TestClient(app)

        response = client.post(
            "/api/project/1/assess", json={"author": "Test Author", "title": "Test Title"}
        )

        assert response.status_code == 422  # Validation error

    def test_assess_content_missing_author_field(self, app_with_models):
        """Test error when author field is missing."""
        app, _ = app_with_models
        client = TestClient(app)

        response = client.post(
            "/api/project/1/assess", json={"summary": "Test summary", "title": "Test Title"}
        )

        assert response.status_code == 422  # Validation error

    def test_assess_content_missing_title_field(self, app_with_models):
        """Test error when title field is missing."""
        app, _ = app_with_models
        client = TestClient(app)

        response = client.post(
            "/api/project/1/assess", json={"summary": "Test summary", "author": "Test Author"}
        )

        assert response.status_code == 422  # Validation error

    def test_assess_content_missing_all_fields(self, app_with_models):
        """Test error when all required fields are missing."""
        app, _ = app_with_models
        client = TestClient(app)

        response = client.post("/api/project/1/assess", json={})

        assert response.status_code == 422  # Validation error

    def test_assess_content_with_different_project_ids(self, app_with_models):
        """Test that correct model is used for different project IDs."""
        app, models = app_with_models
        client = TestClient(app)

        # Setup different responses for different models
        mock_pred_1 = MagicMock()
        mock_pred_1.reasoning = "Project 1 analysis"
        mock_pred_1.prediction_score = "0.8"
        mock_pred_1.prediction = "positive"
        models["1"].return_value = mock_pred_1

        mock_pred_2 = MagicMock()
        mock_pred_2.reasoning = "Project 2 analysis"
        mock_pred_2.prediction_score = "0.3"
        mock_pred_2.prediction = "negative"
        models["2"].return_value = mock_pred_2

        # Test project 1
        response1 = client.post(
            "/api/project/1/assess",
            json={"summary": "Test 1", "author": "Author 1", "title": "Title 1"},
        )
        assert response1.status_code == 200
        assert response1.json()["project_id"] == "1"
        assert response1.json()["reasoning"] == "Project 1 analysis"

        # Test project 2
        response2 = client.post(
            "/api/project/2/assess",
            json={"summary": "Test 2", "author": "Author 2", "title": "Title 2"},
        )
        assert response2.status_code == 200
        assert response2.json()["project_id"] == "2"
        assert response2.json()["reasoning"] == "Project 2 analysis"

        # Verify correct models were called
        assert models["1"].called
        assert models["2"].called

    def test_assess_content_case_insensitive_prediction(self, app_with_models):
        """Test that prediction comparison is case-insensitive."""
        app, models = app_with_models
        client = TestClient(app)

        # Test with uppercase POSITIVE
        mock_prediction = MagicMock()
        mock_prediction.reasoning = "Test"
        mock_prediction.prediction_score = "0.9"
        mock_prediction.prediction = "POSITIVE"
        models["1"].return_value = mock_prediction

        response = client.post(
            "/api/project/1/assess",
            json={"summary": "Test", "author": "Test Author", "title": "Test Title"},
        )
        assert response.json()["recommend"] is True

        # Test with mixed case Negative
        mock_prediction.prediction = "Negative"
        response = client.post(
            "/api/project/1/assess",
            json={"summary": "Test", "author": "Test Author", "title": "Test Title"},
        )
        assert response.json()["recommend"] is False

    def test_assess_content_formats_input_correctly(self, app_with_models):
        """Test that author, title, and summary are formatted correctly for the model."""
        app, models = app_with_models
        client = TestClient(app)

        mock_prediction = MagicMock()
        mock_prediction.reasoning = "Test"
        mock_prediction.prediction_score = "0.8"
        mock_prediction.prediction = "positive"
        models["1"].return_value = mock_prediction

        client.post(
            "/api/project/1/assess",
            json={
                "summary": "Test summary content",
                "author": "Jane Doe",
                "title": "Test Article Title",
            },
        )

        # Verify the model was called with correctly formatted string
        models["1"].assert_called_once()
        call_args = models["1"].call_args
        assert call_args.kwargs["project_id"] == "1"
        assert (
            call_args.kwargs["summary"]
            == "Author:Jane Doe\nTitle:Test Article Title\nSummary:Test summary content"
        )
