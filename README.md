# Content Recommendation Classifier API

A production-ready **content recommendation classifier API** built with FastAPI and DSPy (Declarative Self-improving Language Programs). The system uses Azure OpenAI to intelligently classify content summaries and determine if they should be recommended to users, supporting multiple project-specific fine-tuned models.

## 🚀 Features

- **Multi-Project Model Support**: Load and serve multiple project-specific DSPy models simultaneously
- **Azure OpenAI Integration**: Leverages Azure OpenAI for intelligent content classification
- **Chain-of-Thought Reasoning**: Uses DSPy's ChainOfThought module for explainable predictions
- **FastAPI Framework**: Modern, high-performance API with automatic OpenAPI documentation
- **Comprehensive Testing**: 95% test coverage requirement with extensive test suite
- **Type Safety**: Full mypy type checking with strict mode enabled
- **Code Quality**: Automated formatting (Black), linting (Ruff), and import sorting (isort)

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Testing](#-testing)
- [Code Quality](#-code-quality)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [Docker](#-docker)
- [Deployment](#-deployment)

## ⚡ Quick Start

### Prerequisites

- **Python 3.11** (specified in pyproject.toml)
- **Poetry** for dependency management
- **Azure OpenAI API key** and endpoint access

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kosinal/aim-classification-llm
   cd aim-classification-llm
   ```

2. **Install dependencies**
   ```bash
   make install
   # OR: poetry install
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Azure OpenAI credentials
   ```

4. **Add trained models**
   - Place DSPy model files in `src/aim/model_definitions/`
   - Follow naming pattern: `flag_classifier_project_project_{n}.json` where `{n}` is the project ID
   - Models are auto-discovered and loaded at startup

### Running the Application

**Development mode** (with auto-reload):
```bash
make dev
# OR: poetry run uvicorn aim.main:app --host 0.0.0.0 --port 8000 --reload
```

**Production mode**:
```bash
make run
# OR: poetry run uvicorn aim.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once running, interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Available Endpoints

#### `GET /health`
Health check endpoint that returns application status and loaded project models.

**Response:**
```json
{
  "status":"healthy",
  "models_loaded":true,
  "model_count":3,
  "project_ids":["2","3","4"]
}
```

#### `POST /api/project/{project_id}/assess`
Classify content summary for a specific project.

**Parameters:**
- `project_id` (path, integer): The project ID for which to use the trained model

**Request Body:**
```json
{
  "summary": "Your content summary text here"
}
```

**Response:**
```json
{
  "recommend": true,
  "recommendation_score": 0.85,
  "reasoning": "The content is highly relevant because...",
  "project_id": "1"
}
```

**Status Codes:**
- `200`: Success
- `404`: No model found for the requested project_id
- `503`: Models not loaded yet (startup in progress)
- `500`: Processing error during classification

## 📁 Project Structure

```
aim-assignment/
├── src/aim/                          # Main application package
│   ├── __init__.py
│   ├── main.py                       # FastAPI app, lifespan management, DSPy config
│   ├── config.py                     # Azure OpenAI configuration
│   ├── models.py                     # DSPy model definitions (FlagAssessor, FlagClassifier)
│   ├── routes.py                     # API endpoints
│   ├── schemas.py                    # Pydantic request/response models
│   └── model_definitions/            # Serialized DSPy models (*.json)
│       ├── __init__.py
│       └── flag_classifier_project_project_{n}.json
├── tests/                            # Test suite (mirrors src structure)
│   ├── __init__.py
│   ├── test_main.py                  # Application lifespan and startup tests
│   ├── test_routes.py                # API endpoint tests
│   ├── test_models.py                # DSPy model tests
│   ├── test_schemas.py               # Pydantic schema tests
│   └── test_config.py                # Configuration tests
├── _notebooks/                       # Jupyter notebooks (paired with .py files)
│   ├── 00_EDA.ipynb/py              # Exploratory Data Analysis
│   ├── 01a_LLM_classifier.ipynb/py  # Single model training approach
│   └── 01b_LLM_separate.ipynb/py    # Multi-model training (current approach)
├── _data/                            # Training/evaluation datasets (gitignored)
├── .env.example                      # Environment variable template
├── .env                              # Environment variables (gitignored)
├── .env.test                         # Test environment variables (gitignored)
├── pyproject.toml                    # Poetry dependencies and tool config
├── poetry.lock                       # Locked dependency versions
├── Makefile                          # Development commands
├── CLAUDE.md                         # AI assistant guidance
└── README.md                         # This file
```

## 🛠️ Development

### Makefile Commands

```bash
make help          # Show available commands
make install       # Install dependencies with Poetry
make dev           # Run with auto-reload (development)
make run           # Run in production mode
make test          # Run all tests
make test-verbose  # Run tests with verbose output
make lint          # Run all linters and formatters
make clean         # Remove cache and temporary files

# Docker commands
make docker-build  # Build Docker image
make docker-run    # Run application in Docker using docker-compose
make docker-stop   # Stop Docker containers
make docker-logs   # View Docker container logs
make docker-clean  # Remove Docker containers and images
make docker-shell  # Open shell in running container
```

### Code Quality Tools

The project enforces high code quality standards:

```bash
# Run all quality tools at once
make lint

# Or run individually:
poetry run black .                                              # Code formatter
poetry run isort .                                              # Import sorter
poetry run ruff check --fix                                     # Linter with auto-fix
poetry run mypy --namespace-packages --explicit-package-bases src  # Type checker
```

**Quality Standards:**
- Line length: 100 characters (Black, isort)
- Type checking: Enabled with mypy strict mode
- Test coverage: Minimum 95% (enforced by pytest-cov)

### Jupyter Notebooks

Notebooks are located in `_notebooks/` and use **Jupytext** for version control:

- Notebooks are paired with Python scripts (`.py` files) using percent format
- Editing `.ipynb` auto-syncs to `.py` and vice versa
- **Never edit both files** - choose one and let Jupytext sync automatically
- Format: `ipynb,py:percent` (configured in pyproject.toml)

**Available Notebooks:**
- `00_EDA`: Exploratory Data Analysis
- `01a_LLM_classifier`: Single model training approach
- `01b_LLM_separate`: Multi-model training (current production approach)

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test
# OR: poetry run pytest

# Verbose output
make test-verbose
# OR: poetry run pytest -v

# Specific test file
poetry run pytest tests/test_routes.py

# Specific test function
poetry run pytest tests/test_routes.py::test_assess_content_success

# With coverage report
poetry run pytest --cov=aim --cov-report=html
```

### Test Environment

- Uses `.env.test` for test configuration (configured in pyproject.toml)
- Test files mirror `src/` structure with `test_{module}.py` naming
- Coverage threshold: **95% minimum** (fails below this)
- Comprehensive mocking of DSPy models to avoid Azure API calls during tests

### Testing Approaches

1. **FastAPI TestClient**: Integration tests for API endpoints
2. **Mocking DSPy**: Avoids external Azure API calls
3. **Parameterized Tests**: Multiple scenarios via `@pytest.mark.parametrize`
4. **Async Tests**: Uses `pytest-asyncio` for async endpoint testing

## 🏗️ Architecture

### Multi-Model Loading Pattern

**Critical Design**: The application loads **multiple project-specific models** at startup:

1. At startup, `main.py` scans `src/aim/model_definitions/` for files matching pattern: `flag_classifier_project_project_{n}.json`
2. Extracts `project_id` from filename using regex
3. Loads each model into `app.state.models` dict with `project_id` as key
4. API routes access models via: `request.app.state.models[project_id_str]`

**Adding New Project Models:**
1. Place model file in `src/aim/model_definitions/` following naming pattern
2. Restart the application - models are auto-discovered
3. Verify via `/health` endpoint which shows loaded project IDs

### DSPy Integration

**Configuration** (happens once at startup in `lifespan()`):
- Uses Azure OpenAI endpoint (not standard OpenAI)
- Configures global DSPy settings with `dspy.configure()` and `dspy.settings.configure()`
- Model path format: `azure/{model_name}` (not just model name)

**Model Architecture:**
- `FlagAssessor` (Signature): Defines input/output schema with field descriptions
- `FlagClassifier` (Module): Uses ChainOfThought reasoning for explainable predictions
- Model outputs:
  - `reasoning` (string): Explanation of classification decision
  - `prediction_score` (float 0-1): Confidence score
  - `prediction` (string): "positive" or "negative"

### Request Flow

```
Client Request
    ↓
POST /api/project/{project_id}/assess
    ↓
Validate project_id exists in loaded models
    ↓
Load project-specific FlagClassifier model
    ↓
Process summary with DSPy ChainOfThought
    ↓
Azure OpenAI API call (via DSPy)
    ↓
Parse prediction, score, reasoning
    ↓
Return AssessmentResponse JSON
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
# Azure OpenAI Configuration (required)
AIM_OPENAI_KEY=your-azure-openai-api-key-here
AZURE_ENDPOINT=https://aim-australia-east.openai.azure.com/
AZURE_MODEL_NAME=gpt-5-mini-hiring
AZURE_API_VERSION=2025-03-01-preview
```

**Important:**
- Development uses `.env`
- Tests use `.env.test` (automatically loaded by pytest-dotenv)
- Never commit `.env` files to version control

### Configuration Files

- `pyproject.toml`: Dependencies, tool configuration, and project metadata
- `poetry.lock`: Locked dependency versions for reproducibility
- `.flake8`: Flake8 linter configuration (legacy, mostly replaced by Ruff)

## 🐳 Docker

### Running with Docker

The application includes Docker support for containerized deployment.

#### Quick Start with Docker

1. **Build the Docker image**
   ```bash
   make docker-build
   # OR: docker build -t aim-classifier-api:latest .
   ```

2. **Run with docker-compose** (recommended)
   ```bash
   make docker-run
   # OR: docker-compose up -d
   ```

3. **View logs**
   ```bash
   make docker-logs
   # OR: docker-compose logs -f
   ```

4. **Stop containers**
   ```bash
   make docker-stop
   # OR: docker-compose down
   ```

#### Docker Commands Reference

```bash
make docker-build   # Build Docker image
make docker-run     # Run application in Docker using docker-compose
make docker-stop    # Stop Docker containers
make docker-logs    # View Docker container logs
make docker-clean   # Remove Docker containers and images
make docker-shell   # Open shell in running container
```

#### Docker Environment Setup

When running with Docker, ensure you have:
- A `.env` file with Azure OpenAI credentials

The application will be available at `http://localhost:8000` after starting the container.

## 🚢 Deployment

### Production Considerations

1. **Environment Variables**: Set all required Azure OpenAI credentials
2. **Model Files**: Ensure all project model files are present in `src/aim/model_definitions/`
3. **Dependencies**: Install with `poetry install --no-dev` for production
4. **ASGI Server**: Use production ASGI server like Gunicorn with Uvicorn workers:
   ```bash
   gunicorn aim.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```
5. **Health Checks**: Use `/health` endpoint for container health probes
6. **Monitoring**: Monitor model loading status and Azure OpenAI API latency

## 📝 Common Gotchas

1. **Model Loading**: Models load at startup, not on-demand. Add new models → restart app.
2. **Project ID Type**: URL path uses `int`, model dict keys are `str`. Always convert: `project_id_str = str(project_id)`
3. **LLM Output Parsing**: `prediction_score` may be string from LLM - robust parsing with try/except
4. **Environment Files**: Dev uses `.env`, tests use `.env.test` - they're separate
5. **DSPy Global State**: DSPy configured once at startup - never reconfigure in routes
6. **Coverage Threshold**: 95% minimum - comprehensive tests required for new code
