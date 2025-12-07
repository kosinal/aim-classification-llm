"""Tests for XGBoost embedding classifier."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xgboost as xgb

from aim.predictor import EmbeddingClassifier, UnsupportedProjectError


@pytest.fixture
def mock_classifier_components(tmp_path):
    """Create mock classifier components for testing."""
    # Create mock files and directories
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    embedding_path = tmp_path / "embeddings"
    embedding_path.mkdir()
    thresholds_path = tmp_path / "thresholds.pkl"

    # Create dummy model files for multiple projects
    for project_id in ["project_1", "project_2", "project_3", "project_4"]:
        model_file = models_dir / f"xgb_embedding_classifier_project_{project_id}.json"
        model_file.write_text("{}")

    return {
        "models_dir": models_dir,
        "embedding_path": embedding_path,
        "thresholds_path": thresholds_path,
    }


class TestEmbeddingClassifier:
    """Tests for EmbeddingClassifier."""

    @patch("aim.predictor.SentenceTransformer")
    @patch("aim.predictor.xgb.Booster")
    def test_initialization(
        self, mock_booster_class, mock_sentence_transformer, mock_classifier_components
    ):
        """Test EmbeddingClassifier initialization."""
        components = mock_classifier_components

        # Setup mock booster
        mock_booster = MagicMock()
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            models_dir=components["models_dir"],
            embedding_model_path=components["embedding_path"],
            default_threshold=0.5,
        )

        assert classifier.default_threshold == 0.5
        assert len(classifier.models) == 4
        assert set(classifier.supported_projects) == {"project_1", "project_2", "project_3", "project_4"}
        assert mock_booster_class.called
        assert mock_sentence_transformer.called

    @patch("aim.predictor.SentenceTransformer")
    @patch("aim.predictor.xgb.Booster")
    def test_predict_positive_recommendation(
        self, mock_booster_class, mock_sentence_transformer_class, mock_classifier_components
    ):
        """Test prediction with positive recommendation."""
        components = mock_classifier_components

        # Setup mocks
        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer_class.return_value = mock_embedding_model

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.85])
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            models_dir=components["models_dir"],
            embedding_model_path=components["embedding_path"],
            default_threshold=0.5,
        )

        result = classifier.predict(
            project_id="project_1",
            author="John Doe",
            title="Test Article",
            summary="This is a test summary",
        )

        assert result["recommend"] is True
        assert result["recommendation_score"] == 0.85

    @patch("aim.predictor.SentenceTransformer")
    @patch("aim.predictor.xgb.Booster")
    def test_predict_negative_recommendation(
        self, mock_booster_class, mock_sentence_transformer_class, mock_classifier_components
    ):
        """Test prediction with negative recommendation."""
        components = mock_classifier_components

        # Setup mocks
        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer_class.return_value = mock_embedding_model

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.15])
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            models_dir=components["models_dir"],
            embedding_model_path=components["embedding_path"],
            default_threshold=0.5,
        )

        result = classifier.predict(
            project_id="project_2",
            author="Jane Smith",
            title="Another Article",
            summary="Different summary",
        )

        assert result["recommend"] is False
        assert result["recommendation_score"] == 0.15

    @patch("aim.predictor.SentenceTransformer")
    @patch("aim.predictor.xgb.Booster")
    def test_predict_with_low_threshold(
        self, mock_booster_class, mock_sentence_transformer_class, mock_classifier_components
    ):
        """Test prediction with per-project threshold loaded from file."""
        import pickle
        components = mock_classifier_components

        # Setup thresholds data and write to file
        thresholds_data = {
            "project_1": {"threshold": 0.10, "recall": 0.95, "precision": 0.70, "f1": 0.81},
            "project_2": {"threshold": 0.15, "recall": 0.92, "precision": 0.75, "f1": 0.83},
        }
        with open(components["thresholds_path"], "wb") as f:
            pickle.dump(thresholds_data, f)

        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer_class.return_value = mock_embedding_model

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.12])
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            models_dir=components["models_dir"],
            embedding_model_path=components["embedding_path"],
            thresholds_path=components["thresholds_path"],
            default_threshold=0.5,
        )

        result = classifier.predict(
            project_id="project_1",
            author="Test",
            title="Test",
            summary="Test",
        )

        assert result["recommend"] is True  # Should be True with low threshold (0.12 >= 0.10)
        assert result["recommendation_score"] == 0.12
        assert classifier.get_threshold("project_1") == 0.10

    @patch("aim.predictor.SentenceTransformer")
    @patch("aim.predictor.xgb.Booster")
    def test_predict_formats_text_correctly(
        self, mock_booster_class, mock_sentence_transformer_class, mock_classifier_components
    ):
        """Test that input text is formatted correctly."""
        components = mock_classifier_components

        # Setup mocks
        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer_class.return_value = mock_embedding_model

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.75])
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            models_dir=components["models_dir"],
            embedding_model_path=components["embedding_path"],
            default_threshold=0.5,
        )

        classifier.predict(
            project_id="project_1",
            author="Test Author",
            title="Test Title",
            summary="Test Summary",
        )

        # Verify embedding model was called with correctly formatted text
        mock_embedding_model.encode.assert_called_once()
        call_args = mock_embedding_model.encode.call_args[0][0]
        assert len(call_args) == 1
        text = call_args[0]
        assert "Author: Test Author" in text
        assert "Title: Test Title" in text
        assert "Summary: Test Summary" in text

    @patch("aim.predictor.SentenceTransformer")
    @patch("aim.predictor.xgb.Booster")
    def test_predict_handles_none_values(
        self, mock_booster_class, mock_sentence_transformer_class, mock_classifier_components
    ):
        """Test that None values are handled correctly."""
        components = mock_classifier_components

        # Setup mocks
        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer_class.return_value = mock_embedding_model

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.5])
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            models_dir=components["models_dir"],
            embedding_model_path=components["embedding_path"],
            default_threshold=0.5,
        )

        classifier.predict(
            project_id="project_1",
            author=None,
            title=None,
            summary=None,
        )

        # Verify embedding was called with Unknown/empty defaults
        call_args = mock_embedding_model.encode.call_args[0][0]
        text = call_args[0]
        assert "Author: Unknown" in text
        assert "Title: " in text
        assert "Summary: " in text

    @patch("aim.predictor.SentenceTransformer")
    @patch("aim.predictor.xgb.Booster")
    def test_predict_raises_error_for_unsupported_project(
        self, mock_booster_class, mock_sentence_transformer_class, mock_classifier_components
    ):
        """Test that UnsupportedProjectError is raised for unsupported project_id."""
        components = mock_classifier_components

        # Setup mocks
        mock_embedding_model = MagicMock()
        mock_sentence_transformer_class.return_value = mock_embedding_model

        mock_booster = MagicMock()
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            models_dir=components["models_dir"],
            embedding_model_path=components["embedding_path"],
            default_threshold=0.5,
        )

        # Test with unsupported project_id
        with pytest.raises(UnsupportedProjectError) as exc_info:
            classifier.predict(
                project_id="project_99",
                author="Test",
                title="Test",
                summary="Test",
            )

        assert exc_info.value.project_id == "project_99"
        assert set(exc_info.value.supported_projects) == {"project_1", "project_2", "project_3", "project_4"}
        assert "project_99" in str(exc_info.value)
        assert "not supported" in str(exc_info.value)
