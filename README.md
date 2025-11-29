# FastAPI Project

A FastAPI project with Pytest for unit testing.

## Setup

### Prerequisites
- Python 3.9+
- Poetry

### Installation

Install dependencies:
```bash
make install
```

Or manually:
```bash
poetry install
```

## Running the Application

### Development Mode (with auto-reload)
```bash
make dev
```

### Production Mode
```bash
make run
```

The API will be available at `http://localhost:8000`

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

Run all tests:
```bash
make test
```

Run tests with verbose output:
```bash
make test-verbose
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   └── main.py           # FastAPI application
├── tests/
│   ├── __init__.py
│   └── test_main.py      # Unit tests
├── Makefile              # Commands for running and testing
├── pyproject.toml        # Poetry configuration
└── README.md
```

## Available Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /items/{item_id}` - Get item by ID (with optional query parameter)

## Makefile Commands

- `make help` - Show available commands
- `make install` - Install dependencies
- `make run` - Run FastAPI application
- `make dev` - Run with auto-reload
- `make test` - Run tests
- `make test-verbose` - Run tests with verbose output
- `make clean` - Remove cache and temporary files
