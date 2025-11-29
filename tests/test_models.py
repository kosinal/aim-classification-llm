"""Tests for DSPy model definitions."""

from unittest.mock import MagicMock, patch

import dspy
import pytest

from aim.models import FlagAssessor, FlagClassifier


class TestFlagAssessor:
    """Tests for FlagAssessor signature."""

    def test_flag_assessor_is_dspy_signature(self):
        """Test that FlagAssessor is a DSPy signature."""
        assert issubclass(FlagAssessor, dspy.Signature)

    def test_flag_assessor_signature_fields_exist(self):
        """Test that FlagAssessor signature defines required fields."""
        # DSPy signatures define fields differently than regular classes
        # We verify the signature is properly structured
        assert FlagAssessor.__doc__ is not None
        assert "project_id" in str(FlagAssessor.__dict__) or hasattr(FlagAssessor, "project_id")
        assert "summary" in str(FlagAssessor.__dict__) or hasattr(FlagAssessor, "summary")


class TestFlagClassifier:
    """Tests for FlagClassifier module."""

    def test_flag_classifier_initialization(self):
        """Test FlagClassifier initializes correctly."""
        classifier = FlagClassifier()

        assert isinstance(classifier, dspy.Module)
        assert hasattr(classifier, "prog")
        assert isinstance(classifier.prog, dspy.ChainOfThought)

    @patch("dspy.ChainOfThought.__call__")
    def test_forward_calls_chain_of_thought(self, mock_cot_call):
        """Test that forward method calls ChainOfThought with correct arguments."""
        # Mock the prediction response
        mock_prediction = MagicMock(spec=dspy.Prediction)
        mock_prediction.reasoning = "Test reasoning"
        mock_prediction.prediction_score = "0.85"
        mock_prediction.prediction = "positive"
        mock_cot_call.return_value = mock_prediction

        classifier = FlagClassifier()
        result = classifier.forward(project_id="1", summary="Test summary")

        # Verify ChainOfThought was called with correct arguments
        mock_cot_call.assert_called_once_with(project_id="1", summary="Test summary")
        assert result == mock_prediction

    @patch("dspy.ChainOfThought.__call__")
    def test_forward_returns_prediction_object(self, mock_cot_call):
        """Test that forward returns a dspy.Prediction object."""
        mock_prediction = MagicMock(spec=dspy.Prediction)
        mock_prediction.reasoning = "Content is relevant to project goals"
        mock_prediction.prediction_score = "0.92"
        mock_prediction.prediction = "positive"
        mock_cot_call.return_value = mock_prediction

        classifier = FlagClassifier()
        result = classifier(project_id="2", summary="Important project update")

        assert result == mock_prediction
        assert result.reasoning == "Content is relevant to project goals"
        assert result.prediction_score == "0.92"
        assert result.prediction == "positive"

    @patch("dspy.ChainOfThought.__call__")
    def test_forward_with_negative_prediction(self, mock_cot_call):
        """Test forward with negative prediction."""
        mock_prediction = MagicMock(spec=dspy.Prediction)
        mock_prediction.reasoning = "Content not relevant"
        mock_prediction.prediction_score = "0.15"
        mock_prediction.prediction = "negative"
        mock_cot_call.return_value = mock_prediction

        classifier = FlagClassifier()
        result = classifier(project_id="3", summary="Spam content")

        assert result.prediction == "negative"
        assert result.prediction_score == "0.15"

    def test_classifier_can_be_saved_and_loaded(self, tmp_path):
        """Test that classifier can be saved and loaded."""
        classifier = FlagClassifier()
        save_path = tmp_path / "test_classifier.json"

        # Save the classifier
        classifier.save(str(save_path))

        # Load into a new classifier
        new_classifier = FlagClassifier()
        new_classifier.load(str(save_path))

        # Both should have ChainOfThought programs
        assert hasattr(new_classifier, "prog")
        assert isinstance(new_classifier.prog, dspy.ChainOfThought)
