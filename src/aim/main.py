"""Main FastAPI application."""

import logging
import os

from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI

from aim.predictor import EmbeddingClassifier
from aim.routes import router


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:  # pragma: no cover
    """
    Lifespan context manager for FastAPI application.

    Handles XGBoost model loading on startup and cleanup on shutdown.
    """
    logger.info("Loading XGBoost embedding classifier...")

    # Get paths to model artifacts
    data_dir = Path(os.environ["MODEL_BASE_PATH"])

    model_path = data_dir / "xgboost_model.json"
    encoder_path = data_dir / "project_encoder.pkl"
    embedding_model_path = data_dir / "_emb"

    # Verify all required files exist
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found at {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Project encoder not found at {encoder_path}")
    if not embedding_model_path.exists():
        raise FileNotFoundError(f"Embedding model not found at {embedding_model_path}")

    # Load the classifier (threshold optimized from notebook: 0.10 for max recall)
    classifier = EmbeddingClassifier(
        model_path=model_path,
        encoder_path=encoder_path,
        embedding_model_path=embedding_model_path,
        threshold=0.10,
    )

    application.state.classifier = classifier
    logger.info("XGBoost embedding classifier loaded successfully")

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
    return {
        "status": "healthy",
        "classifier_loaded": classifier_loaded,
        "model_type": "xgboost_embedding" if classifier_loaded else None,
        "threshold": app.state.classifier.threshold if classifier_loaded else None,
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
