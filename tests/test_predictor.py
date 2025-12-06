"""Tests for XGBoost embedding classifier."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xgboost as xgb

from aim.predictor import EmbeddingClassifier


@pytest.fixture
def mock_classifier_components(tmp_path):
    """Create mock classifier components for testing."""
    # Create mock files
    model_path = tmp_path / "model.json"
    encoder_path = tmp_path / "encoder.pkl"
    embedding_path = tmp_path / "embeddings"

    # Create dummy files
    model_path.write_text("{}")
    embedding_path.mkdir()

    # Mock encoder
    mock_encoder = MagicMock()
    mock_encoder.transform.return_value = np.array([[1, 0, 0, 0]])

    return {
        "model_path": model_path,
        "encoder_path": encoder_path,
        "embedding_path": embedding_path,
        "mock_encoder": mock_encoder,
    }


class TestEmbeddingClassifier:
    """Tests for EmbeddingClassifier."""

    @patch("aim.predictor.SentenceTransformer")
    @patch("aim.predictor.xgb.Booster")
    @patch("builtins.open")
    @patch("aim.predictor.pickle.load")
    def test_initialization(
        self, mock_pickle_load, mock_open, mock_booster, mock_sentence_transformer, mock_classifier_components
    ):
        """Test EmbeddingClassifier initialization."""
        components = mock_classifier_components
        mock_pickle_load.return_value = components["mock_encoder"]

        classifier = EmbeddingClassifier(
            model_path=components["model_path"],
            encoder_path=components["encoder_path"],
            embedding_model_path=components["embedding_path"],
            threshold=0.5,
        )

        assert classifier.threshold == 0.5
        assert mock_booster.called
        assert mock_sentence_transformer.called

    @patch("aim.predictor.SentenceTransformer")
    @patch("aim.predictor.xgb.Booster")
    @patch("builtins.open")
    @patch("aim.predictor.pickle.load")
    def test_predict_positive_recommendation(
        self, mock_pickle_load, mock_open, mock_booster_class, mock_sentence_transformer_class, mock_classifier_components
    ):
        """Test prediction with positive recommendation."""
        components = mock_classifier_components

        # Setup mocks
        mock_encoder = components["mock_encoder"]
        mock_pickle_load.return_value = mock_encoder

        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer_class.return_value = mock_embedding_model

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.85])
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            model_path=components["model_path"],
            encoder_path=components["encoder_path"],
            embedding_model_path=components["embedding_path"],
            threshold=0.5,
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
    @patch("builtins.open")
    @patch("aim.predictor.pickle.load")
    def test_predict_negative_recommendation(
        self, mock_pickle_load, mock_open, mock_booster_class, mock_sentence_transformer_class, mock_classifier_components
    ):
        """Test prediction with negative recommendation."""
        components = mock_classifier_components

        # Setup mocks
        mock_encoder = components["mock_encoder"]
        mock_pickle_load.return_value = mock_encoder

        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer_class.return_value = mock_embedding_model

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.15])
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            model_path=components["model_path"],
            encoder_path=components["encoder_path"],
            embedding_model_path=components["embedding_path"],
            threshold=0.5,
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
    @patch("builtins.open")
    @patch("aim.predictor.pickle.load")
    def test_predict_with_low_threshold(
        self, mock_pickle_load, mock_open, mock_booster_class, mock_sentence_transformer_class, mock_classifier_components
    ):
        """Test prediction with low threshold (0.10 from notebook)."""
        components = mock_classifier_components

        # Setup mocks
        mock_encoder = components["mock_encoder"]
        mock_pickle_load.return_value = mock_encoder

        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer_class.return_value = mock_embedding_model

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.12])
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            model_path=components["model_path"],
            encoder_path=components["encoder_path"],
            embedding_model_path=components["embedding_path"],
            threshold=0.10,  # Low threshold from notebook
        )

        result = classifier.predict(
            project_id="project_1",
            author="Test",
            title="Test",
            summary="Test",
        )

        assert result["recommend"] is True  # Should be True with low threshold
        assert result["recommendation_score"] == 0.12

    @patch("aim.predictor.SentenceTransformer")
    @patch("aim.predictor.xgb.Booster")
    @patch("builtins.open")
    @patch("aim.predictor.pickle.load")
    def test_predict_formats_text_correctly(
        self, mock_pickle_load, mock_open, mock_booster_class, mock_sentence_transformer_class, mock_classifier_components
    ):
        """Test that input text is formatted correctly."""
        components = mock_classifier_components

        # Setup mocks
        mock_encoder = components["mock_encoder"]
        mock_pickle_load.return_value = mock_encoder

        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer_class.return_value = mock_embedding_model

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.75])
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            model_path=components["model_path"],
            encoder_path=components["encoder_path"],
            embedding_model_path=components["embedding_path"],
            threshold=0.5,
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
    @patch("builtins.open")
    @patch("aim.predictor.pickle.load")
    def test_predict_handles_none_values(
        self, mock_pickle_load, mock_open, mock_booster_class, mock_sentence_transformer_class, mock_classifier_components
    ):
        """Test that None values are handled correctly."""
        components = mock_classifier_components

        # Setup mocks
        mock_encoder = components["mock_encoder"]
        mock_pickle_load.return_value = mock_encoder

        mock_embedding_model = MagicMock()
        mock_embedding_model.encode.return_value = np.random.rand(1, 384)
        mock_sentence_transformer_class.return_value = mock_embedding_model

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.5])
        mock_booster_class.return_value = mock_booster

        classifier = EmbeddingClassifier(
            model_path=components["model_path"],
            encoder_path=components["encoder_path"],
            embedding_model_path=components["embedding_path"],
            threshold=0.5,
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
