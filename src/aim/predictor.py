"""XGBoost-based embedding classifier for content recommendation."""

import pickle

from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder


class EmbeddingClassifier:
    """XGBoost classifier using sentence embeddings and project encoding."""

    def __init__(
        self,
        model_path: str | Path,
        encoder_path: str | Path,
        embedding_model_path: str | Path,
        threshold: float = 0.5,
    ) -> None:
        """
        Initialize the embedding classifier.

        Args:
            model_path: Path to the XGBoost model JSON file
            encoder_path: Path to the project encoder pickle file
            embedding_model_path: Path to the sentence transformer model directory
            threshold: Classification threshold (default: 0.5)
        """
        self.threshold = threshold

        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        # Load project encoder
        with open(encoder_path, "rb") as f:
            self.encoder: OneHotEncoder = pickle.load(f)

        # Load embedding model
        self.embedding_model = SentenceTransformer(str(embedding_model_path))

    def predict(
        self, project_id: str, author: str, title: str, summary: str
    ) -> dict[str, Any]:
        """
        Predict recommendation for given content.

        Args:
            project_id: The project identifier (e.g., "project_1")
            author: Author of the content
            title: Title of the content
            summary: Summary of the content

        Returns:
            Dictionary with:
                - recommend: bool (whether to recommend)
                - recommendation_score: float (probability 0-1)
                - reasoning: str (explanation of prediction)
        """
        # Prepare combined text (matching training format)
        combined_text = (
            f"Author: {author if author else 'Unknown'}\n"
            f"Title: {title if title else ''}\n"
            f"Summary: {summary if summary else ''}"
        )

        # Generate embedding
        embedding = self.embedding_model.encode([combined_text])
        embedding_array = np.array(embedding)

        # One-hot encode project_id
        project_encoded = self.encoder.transform([[project_id]])

        # Combine features
        features = np.hstack([embedding_array, project_encoded])

        # Create DMatrix and predict
        prediction_proba = self.model.predict(xgb.DMatrix(features))[0]

        # Apply threshold
        recommend = bool(prediction_proba >= self.threshold)

        return {
            "recommend": recommend,
            "recommendation_score": float(prediction_proba),
        }
