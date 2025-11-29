"""Main FastAPI application."""

import logging
import re

import dspy

from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
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

    logger.info("Loading DSPy models for each project...")
    # Load multiple models for each project
    models_dir = Path(str(files("aim.model_definitions")))
    project_models: dict[str, FlagClassifier] = {}

    # Pattern to match: flag_classifier_project_project_{n}.json
    pattern = re.compile(r"flag_classifier_project_project_(\d+)\.json")

    for model_file in models_dir.glob("flag_classifier_project_project_*.json"):
        match = pattern.match(model_file.name)
        if match:
            project_id = match.group(1)
            logger.info(f"Loading model for project {project_id} from {model_file.name}")

            model = FlagClassifier()
            model.load(str(model_file))
            project_models[project_id] = model

            logger.info(f"Model for project {project_id} loaded successfully")

    application.state.models = project_models
    logger.info(f"Loaded {len(project_models)} project models: {list(project_models.keys())}")

    yield

    logger.info("Shutting down application...")
    application.state.models = None


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
    models_loaded = hasattr(app.state, "models") and app.state.models is not None
    model_count = len(app.state.models) if models_loaded else 0
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "model_count": model_count,
        "project_ids": list(app.state.models.keys()) if models_loaded else [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "aim.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
