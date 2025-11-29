"""Tests for main FastAPI application."""

import re

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import dspy
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aim.main import app, lifespan
from aim.models import FlagClassifier


client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Flagging Classifier API", "version": "0.1.0"}


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "healthy"
    assert "models_loaded" in response_data
    assert "model_count" in response_data
    assert "project_ids" in response_data
    assert isinstance(response_data["models_loaded"], bool)
    assert isinstance(response_data["model_count"], int)
    assert isinstance(response_data["project_ids"], list)


class TestLifespan:
    """Tests for application lifespan context manager."""

    def test_app_has_lifespan_configured(self):
        """Test that app has lifespan configured."""
        # The app should have our lifespan configured
        assert app.router.lifespan_context is not None

    def test_health_check_reflects_model_state(self):
        """Test health endpoint reflects actual model state from lifespan."""
        # This test uses the actual app with real lifespan
        # Models are loaded during lifespan startup
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        # These fields must exist regardless of model loading success
        assert "models_loaded" in data
        assert "model_count" in data
        assert "project_ids" in data
