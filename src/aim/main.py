"""Main FastAPI application."""

import logging

import dspy

from contextlib import asynccontextmanager
from importlib.resources import files
from typing import AsyncGenerator

from fastapi import FastAPI

from aim.config import get_config
from aim.models import FlagClassifier
from aim.routes import router


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.

    Handles DSPy configuration and model loading on startup and cleanup on shutdown.
    """
    logger.info("Configuring DSPy with Azure OpenAI...")
    config = get_config()

    lm = dspy.LM(
        model=config.get_azure_model_path(),
        api_base=config.endpoint,
        api_version=config.api_version,
        api_key=config.api_key,
    )

    dspy.configure(lm=lm)
    dspy.settings.configure(lm=lm)
    logger.info("DSPy configured successfully with Azure OpenAI")

    logger.info("Loading DSPy model...")
    # Load model from the models package
    model_path = files("aim.model_definitions").joinpath("flag_classifier_optimized.json")
    loaded_model = FlagClassifier()
    loaded_model.load(str(model_path))
    application.state.model = loaded_model
    logger.info("Model loaded successfully!")

    yield

    logger.info("Shutting down application...")
    application.state.model = None


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
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return {"status": "healthy", "model_loaded": model_loaded}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "aim.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
