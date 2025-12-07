"""Main FastAPI application."""

import logging
import os

from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI

from aim import config
from aim.predictor import EmbeddingClassifier
from aim.routes import router


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:  # pragma: no cover
    """
    Lifespan context manager for FastAPI application.

    Handles XGBoost per-project model loading on startup and cleanup on shutdown.
    """
    logger.info("Loading XGBoost per-project embedding classifiers...")

    # Get paths to model artifacts
    models_dir = Path(config.settings.model_base_path)
    embedding_model_path = models_dir / "_emb"
    thresholds_path = models_dir / "xgb_project_optimal_thresholds.pkl"

    # Verify required paths exist
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found at {models_dir}")
    if not embedding_model_path.exists():
        raise FileNotFoundError(f"Embedding model not found at {embedding_model_path}")
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Thresholds file not found at {thresholds_path}")

    # Load the classifier with per-project models
    classifier = EmbeddingClassifier(
        models_dir=models_dir,
        embedding_model_path=embedding_model_path,
        thresholds_path=thresholds_path,
        default_threshold=0.1,
    )

    application.state.classifier = classifier
    logger.info(
        f"XGBoost embedding classifiers loaded successfully for projects: "
        f"{', '.join(classifier.supported_projects)}"
    )

    yield

    logger.info("Shutting down application...")
    application.state.classifier = None


app = FastAPI(
    title="Aim Home Assignment - Recommendation Classifier API",
    description="Content recommendation classifier using DSPy",
    version="0.1.0",
    lifespan=lifespan,
)

# Include API router
app.include_router(router)


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {"message": "Flagging Classifier API", "version": "0.1.0"}


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    classifier_loaded = hasattr(app.state, "classifier") and app.state.classifier is not None

    if classifier_loaded:
        classifier = app.state.classifier
        project_thresholds = {
            project_id: classifier.get_threshold(project_id)
            for project_id in classifier.supported_projects
        }
        return {
            "status": "healthy",
            "classifier_loaded": True,
            "model_type": "xgboost_embedding_per_project",
            "loaded_projects": classifier.supported_projects,
            "project_thresholds": project_thresholds,
            "default_threshold": classifier.default_threshold,
        }
    else:
        return {
            "status": "healthy",
            "classifier_loaded": False,
            "model_type": None,
        }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "aim.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
