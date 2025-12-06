"""Tests for API routes."""

from unittest.mock import MagicMock

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aim.predictor import EmbeddingClassifier, UnsupportedProjectError
from aim.routes import router
from aim.schemas import AssessRequest, AssessResponse


@pytest.fixture
def app_with_classifier():
    """Create FastAPI app with mocked classifier for testing."""
    app = FastAPI()
    app.include_router(router)

    # Create mock classifier
    mock_classifier = MagicMock(spec=EmbeddingClassifier)

    # Setup app state with classifier
    app.state.classifier = mock_classifier

    return app, mock_classifier


@pytest.fixture
def app_without_classifier():
    """Create FastAPI app without classifier for testing."""
    app = FastAPI()
    app.include_router(router)
    app.state.classifier = None
    return app


class TestAssessContentEndpoint:
    """Tests for /api/project/{project_id}/assess endpoint."""

    def test_assess_content_success_positive_prediction(self, app_with_classifier):
        """Test successful content assessment with positive prediction."""
        app, mock_classifier = app_with_classifier
        client = TestClient(app)

        # Mock prediction response
        mock_classifier.predict.return_value = {
            "recommend": True,
            "recommendation_score": 0.95,
        }

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
        assert data["project_id"] == "project_1"

    def test_assess_content_success_negative_prediction(self, app_with_classifier):
        """Test successful content assessment with negative prediction."""
        app, mock_classifier = app_with_classifier
        client = TestClient(app)

        # Mock prediction response
        mock_classifier.predict.return_value = {
            "recommend": False,
            "recommendation_score": 0.12,
        }

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
        assert data["project_id"] == "project_2"

    def test_assess_content_classifier_not_loaded(self, app_without_classifier):
        """Test error response when classifier is not loaded."""
        client = TestClient(app_without_classifier)

        response = client.post(
            "/api/project/1/assess",
            json={"summary": "Test content", "author": "Test Author", "title": "Test Title"},
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Classifier not loaded yet"

    def test_assess_content_model_processing_error(self, app_with_classifier):
        """Test error response when model processing fails."""
        app, mock_classifier = app_with_classifier
        client = TestClient(app)

        # Mock classifier to raise exception
        mock_classifier.predict.side_effect = Exception("Embedding generation failed")

        response = client.post(
            "/api/project/1/assess",
            json={"summary": "Test content", "author": "Test Author", "title": "Test Title"},
        )

        assert response.status_code == 500
        assert "Embedding generation failed" in response.json()["detail"]

    def test_assess_content_validates_request_schema(self, app_with_classifier):
        """Test that request schema validation works."""
        app, _ = app_with_classifier
        client = TestClient(app)

        # Empty summary should fail validation
        response = client.post(
            "/api/project/1/assess",
            json={"summary": "", "author": "Test Author", "title": "Test Title"},
        )

        assert response.status_code == 422  # Validation error

    def test_assess_content_missing_summary_field(self, app_with_classifier):
        """Test error when summary field is missing."""
        app, _ = app_with_classifier
        client = TestClient(app)

        response = client.post(
            "/api/project/1/assess", json={"author": "Test Author", "title": "Test Title"}
        )

        assert response.status_code == 422  # Validation error

    def test_assess_content_missing_author_field(self, app_with_classifier):
        """Test error when author field is missing."""
        app, _ = app_with_classifier
        client = TestClient(app)

        response = client.post(
            "/api/project/1/assess", json={"summary": "Test summary", "title": "Test Title"}
        )

        assert response.status_code == 422  # Validation error

    def test_assess_content_missing_title_field(self, app_with_classifier):
        """Test error when title field is missing."""
        app, _ = app_with_classifier
        client = TestClient(app)

        response = client.post(
            "/api/project/1/assess", json={"summary": "Test summary", "author": "Test Author"}
        )

        assert response.status_code == 422  # Validation error

    def test_assess_content_missing_all_fields(self, app_with_classifier):
        """Test error when all required fields are missing."""
        app, _ = app_with_classifier
        client = TestClient(app)

        response = client.post("/api/project/1/assess", json={})

        assert response.status_code == 422  # Validation error

    def test_assess_content_with_different_project_ids(self, app_with_classifier):
        """Test that correct project_id format is used for different project IDs."""
        app, mock_classifier = app_with_classifier
        client = TestClient(app)

        # Setup response
        mock_classifier.predict.return_value = {
            "recommend": True,
            "recommendation_score": 0.8,
            "reasoning": "Test reasoning",
        }

        # Test project 1
        response1 = client.post(
            "/api/project/1/assess",
            json={"summary": "Test 1", "author": "Author 1", "title": "Title 1"},
        )
        assert response1.status_code == 200
        assert response1.json()["project_id"] == "project_1"

        # Test project 3
        response2 = client.post(
            "/api/project/3/assess",
            json={"summary": "Test 3", "author": "Author 3", "title": "Title 3"},
        )
        assert response2.status_code == 200
        assert response2.json()["project_id"] == "project_3"

        # Verify classifier was called with correct project_id format
        calls = mock_classifier.predict.call_args_list
        assert calls[0].kwargs["project_id"] == "project_1"
        assert calls[1].kwargs["project_id"] == "project_3"

    def test_assess_content_calls_classifier_with_correct_args(self, app_with_classifier):
        """Test that classifier is called with correctly formatted arguments."""
        app, mock_classifier = app_with_classifier
        client = TestClient(app)

        mock_classifier.predict.return_value = {
            "recommend": True,
            "recommendation_score": 0.8,
            "reasoning": "Test",
        }

        client.post(
            "/api/project/1/assess",
            json={
                "summary": "Test summary content",
                "author": "Jane Doe",
                "title": "Test Article Title",
            },
        )

        # Verify the classifier was called with correct arguments
        mock_classifier.predict.assert_called_once_with(
            project_id="project_1",
            author="Jane Doe",
            title="Test Article Title",
            summary="Test summary content",
        )

    def test_assess_content_with_low_threshold_score(self, app_with_classifier):
        """Test prediction with low score near threshold."""
        app, mock_classifier = app_with_classifier
        client = TestClient(app)

        # Low score but above 0.10 threshold
        mock_classifier.predict.return_value = {
            "recommend": True,
            "recommendation_score": 0.12,
            "reasoning": "Marginally recommended",
        }

        response = client.post(
            "/api/project/1/assess",
            json={"summary": "Test", "author": "Test", "title": "Test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["recommend"] is True
        assert data["recommendation_score"] == 0.12

    def test_assess_content_unsupported_project_returns_404(self, app_with_classifier):
        """Test that unsupported project_id returns 404 status code."""
        app, mock_classifier = app_with_classifier
        client = TestClient(app)

        # Mock classifier to raise UnsupportedProjectError
        mock_classifier.predict.side_effect = UnsupportedProjectError(
            project_id="project_99",
            supported_projects=["project_1", "project_2", "project_3"],
        )

        response = client.post(
            "/api/project/99/assess",
            json={"summary": "Test content", "author": "Test Author", "title": "Test Title"},
        )

        assert response.status_code == 404
        detail = response.json()["detail"]
        assert "project_99" in detail
        assert "not supported" in detail
        assert "project_1, project_2, project_3" in detail
