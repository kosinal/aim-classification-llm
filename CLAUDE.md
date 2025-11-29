# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **content recommendation classifier API** built with FastAPI and DSPy (Declarative Self-improving Language Programs). The system uses Azure OpenAI to classify content summaries and determine if they should be recommended to users, with separate fine-tuned models for different projects.

**Key Architecture Pattern**: Multi-project model loading - the application loads separate DSPy models at startup (one per project ID) from `src/aim/model_definitions/` using a filename pattern `flag_classifier_project_project_{n}.json`.

## Common Development Commands

### Environment Setup
```bash
# Install all dependencies (including dev dependencies)
poetry install

# Install pre-commit hooks (if configured)
pre-commit install
```

### Running the Application
```bash
# Development mode with auto-reload
make dev
# OR: poetry run uvicorn aim.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
make run
# OR: poetry run uvicorn aim.main:app --host 0.0.0.0 --port 8000
```

**API Documentation**: Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Testing
```bash
# Run all tests
make test
# OR: poetry run pytest

# Run tests with verbose output
make test-verbose
# OR: poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_routes.py

# Run specific test function
poetry run pytest tests/test_routes.py::test_assess_content_success

# Run with coverage report
poetry run pytest --cov=aim --cov-report=html
```

### Code Quality
```bash
# Run all linters and formatters
make lint

# Individual tools:
poetry run black .              # Code formatter
poetry run isort .              # Import sorter
poetry run ruff check --fix     # Linter with auto-fix
poetry run mypy --namespace-packages --explicit-package-bases src  # Type checker
```

**Quality Standards**:
- Test coverage minimum: 95% (configured in pyproject.toml)
- Line length: 100 characters (Black, isort)
- Type checking: Enabled with mypy (strict mode)

## Architecture and Design Patterns

### Application Structure
```
src/aim/
├── main.py              # FastAPI app, lifespan management, DSPy configuration
├── config.py            # Azure OpenAI configuration from environment
├── models.py            # DSPy model definitions (FlagAssessor, FlagClassifier)
├── routes.py            # API endpoints (/api/project/{id}/assess)
├── schemas.py           # Pydantic request/response models
└── model_definitions/   # Serialized DSPy models (*.json files)
```

### Multi-Model Loading Pattern

**Critical**: The application loads **multiple project-specific models** at startup, NOT a single model:

```python
# In main.py lifespan():
# 1. Scan model_definitions/ for files matching: flag_classifier_project_project_{n}.json
# 2. Extract project_id from filename using regex pattern
# 3. Load each model into app.state.models dict with project_id as key
# 4. Routes access models via: request.app.state.models[project_id_str]
```

**When adding new project models**:
1. Place model file in `src/aim/model_definitions/` following naming pattern
2. Restart the application - model will be auto-discovered and loaded
3. Verify via `/health` endpoint which shows loaded project IDs

### DSPy Integration

**DSPy Configuration**: Happens once at startup in `lifespan()`:
- Uses Azure OpenAI endpoint (not standard OpenAI)
- Configures global DSPy settings with `dspy.configure()` and `dspy.settings.configure()`
- Model path format: `azure/{model_name}` (not just model name)

**Model Architecture**:
- `FlagAssessor` (Signature): Defines input/output schema with descriptions
- `FlagClassifier` (Module): Uses ChainOfThought reasoning for predictions
- Models output: reasoning (string), prediction_score (float 0-1), prediction (positive/negative)

### API Design

**Single Endpoint**: `POST /api/project/{project_id}/assess`
- Path parameter: `project_id` (integer) - selects which trained model to use
- Request body: `{"summary": "text to classify"}`
- Response: `{"recommend": bool, "recommendation_score": float, "reasoning": str, "project_id": str}`

**Error Handling**:
- 503: Models not loaded yet (startup not complete)
- 404: No model found for requested project_id
- 500: Processing error during classification

### Environment Configuration

**Required Environment Variables** (see `.env.example`):
```bash
AIM_OPENAI_KEY=...                                              # Azure OpenAI API key
AZURE_ENDPOINT=https://aim-australia-east.openai.azure.com/     # Azure endpoint
AZURE_MODEL_NAME=gpt-5-mini-hiring                              # Model deployment name
AZURE_API_VERSION=2025-03-01-preview                            # API version
```

**Test Environment**: Uses `.env.test` for test runs (configured in pyproject.toml `pytest.ini_options`)

## Jupyter Notebooks

**Location**: `_notebooks/` directory

**Jupytext Integration**: Notebooks are paired with Python scripts (`.py` files) using percent format:
- Editing `.ipynb` auto-syncs to `.py` and vice versa
- Format: `ipynb,py:percent` (configured in pyproject.toml)
- **Never edit both files** - choose one and let jupytext sync

**Notebooks**:
- `00_EDA.py/ipynb`: Exploratory Data Analysis
- `01a_LLM_classifier.py/ipynb`: Single model training approach
- `01b_LLM_separate.py/ipynb`: Multi-model training (current approach)

## Data Directory

**Location**: `_data/` - Contains training/evaluation datasets (not in version control)

## Testing Patterns

**Test Structure**: Mirrors `src/` structure with `test_{module}.py` files

**Key Testing Approaches**:
1. **FastAPI TestClient**: Used for integration tests (see `test_main.py`, `test_routes.py`)
2. **Mocking DSPy**: Tests mock DSPy models to avoid Azure API calls
3. **Parameterized Tests**: Uses `@pytest.mark.parametrize` for multiple scenarios
4. **Async Tests**: Uses `pytest-asyncio` for async endpoint testing

**Example Test Pattern**:
```python
from fastapi.testclient import TestClient
from aim.main import app

client = TestClient(app)

def test_endpoint():
    response = client.get("/")
    assert response.status_code == 200
```

## Code Style and Conventions

**Python Version**: 3.11 (specified in pyproject.toml)

**Import Organization** (enforced by isort):
1. Standard library imports
2. Third-party imports (pytest, fastapi, etc.)
3. First-party imports (aim.*)
4. Blank line between import groups

**Type Annotations**:
- **Required** for all function signatures (enforced by mypy)
- Use modern syntax: `dict[str, int]` not `Dict[str, int]`
- Tests have relaxed type annotation requirements

**Ignored Ruff Rules**: See pyproject.toml `[tool.ruff.lint]` section for comprehensive list
- Most notable: D100-D104 (docstring requirements reduced), E501 (line length handled by Black)

## Common Gotchas

1. **Model Loading**: Models load at startup, not on-demand. If adding new models, restart the app.

2. **Project ID Type Mismatch**: URL path uses `int`, model dict keys are `str`. Always convert: `project_id_str = str(project_id)`

3. **LLM Output Parsing**: `prediction_score` may come as string from LLM - use robust parsing with try/except and fallback logic

4. **Environment Variables**: Different `.env` files for dev (`.env`) and test (`.env.test`). Tests won't use your dev config.

5. **DSPy Global State**: DSPy uses global configuration via `dspy.configure()`. Don't reconfigure in routes - it's set once at startup.

6. **Coverage Threshold**: 95% minimum. If adding new code, ensure comprehensive test coverage.

## File Naming Patterns

**Models**: `flag_classifier_project_project_{n}.json` where `{n}` is the numeric project ID
**Tests**: `test_{module_name}.py` for each module in `src/aim/`
**Notebooks**: Descriptive names with numeric prefixes for ordering (e.g., `01a_`, `01b_`)
