"""XGBoost-based embedding classifier for content recommendation."""

import pickle
import re

from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from sentence_transformers import SentenceTransformer


class UnsupportedProjectError(Exception):
    """Raised when a project_id is not supported by the classifier."""

    def __init__(self, project_id: str, supported_projects: list[str]) -> None:
        """
        Initialize the exception.

        Args:
            project_id: The unsupported project identifier
            supported_projects: List of supported project identifiers
        """
        self.project_id = project_id
        self.supported_projects = supported_projects
        message = (
            f"Project '{project_id}' is not supported. "
            f"Supported projects: {', '.join(supported_projects)}"
        )
        super().__init__(message)


class EmbeddingClassifier:
    """XGBoost classifier using sentence embeddings with separate models per project."""

    def __init__(
        self,
        models_dir: str | Path,
        embedding_model_path: str | Path,
        thresholds_path: str | Path | None = None,
        default_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the embedding classifier with per-project models.

        Args:
            models_dir: Directory containing XGBoost model JSON files
            embedding_model_path: Path to the sentence transformer model directory
            thresholds_path: Optional path to pickle file with optimal thresholds per project
            default_threshold: Default classification threshold when project-specific not available
        """
        self.default_threshold = default_threshold
        self.models: dict[str, xgb.Booster] = {}
        self.thresholds: dict[str, float] = {}

        models_dir = Path(models_dir)

        # Load all project-specific models
        model_pattern = "xgb_embedding_classifier_project_*.json"
        model_files = list(models_dir.glob(model_pattern))

        if not model_files:
            raise FileNotFoundError(
                f"No models found matching pattern '{model_pattern}' in {models_dir}"
            )

        # Extract project_id from filename and load model
        pattern = re.compile(r"xgb_embedding_classifier_project_(.+)\.json")
        for model_file in model_files:
            match = pattern.match(model_file.name)
            if match:
                project_id = match.group(1)
                booster = xgb.Booster()
                booster.load_model(str(model_file))
                self.models[project_id] = booster

        if not self.models:
            raise ValueError("No valid project models loaded")

        # Load optimal thresholds if provided
        if thresholds_path is not None:
            thresholds_path = Path(thresholds_path)
            if thresholds_path.exists():
                with open(thresholds_path, "rb") as f:
                    thresholds_data = pickle.load(f)
                    # Extract threshold values from the metadata dict
                    self.thresholds = {
                        project_id: data["threshold"]
                        for project_id, data in thresholds_data.items()
                        if "threshold" in data
                    }

        # Load embedding model
        self.embedding_model = SentenceTransformer(str(embedding_model_path))

    @property
    def supported_projects(self) -> list[str]:
        """Get list of supported project IDs."""
        return sorted(self.models.keys())

    def get_threshold(self, project_id: str) -> float:
        """Get threshold for a specific project."""
        return self.thresholds.get(project_id, self.default_threshold)

    def predict(
        self, project_id: str, author: str, title: str, summary: str
    ) -> dict[str, Any]:
        """
        Predict recommendation for given content using project-specific model.

        Args:
            project_id: The project identifier (e.g., "project_1")
            author: Author of the content
            title: Title of the content
            summary: Summary of the content

        Returns:
            Dictionary with:
                - recommend: bool (whether to recommend)
                - recommendation_score: float (probability 0-1)

        Raises:
            UnsupportedProjectError: If project_id is not supported
        """
        # Validate project_id is supported
        if project_id not in self.models:
            raise UnsupportedProjectError(project_id, self.supported_projects)

        # Get project-specific model and threshold
        model = self.models[project_id]
        threshold = self.get_threshold(project_id)

        # Prepare combined text (matching training format)
        combined_text = (
            f"Author: {author if author else 'Unknown'}\n"
            f"Title: {title if title else ''}\n"
            f"Summary: {summary if summary else ''}"
        )

        # Generate embedding
        embedding = self.embedding_model.encode([combined_text])
        embedding_array = np.array(embedding)

        # Create DMatrix and predict using project-specific model
        prediction_proba = model.predict(xgb.DMatrix(embedding_array))[0]

        # Apply project-specific threshold
        recommend = bool(prediction_proba >= threshold)

        return {
            "recommend": recommend,
            "recommendation_score": float(prediction_proba),
        }
