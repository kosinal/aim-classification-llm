.PHONY: help install run dev test test-verbose clean

help:
	@echo Available commands:
	@echo   make install       - Install dependencies using Poetry
	@echo   make run           - Run FastAPI application
	@echo   make dev           - Run FastAPI with auto-reload
	@echo   make test          - Run tests with Pytest
	@echo   make test-verbose  - Run tests with verbose output
	@echo   make clean         - Remove cache and temporary files
	@echo   make lint          - Run lint on whole project

lint:
	poetry run black .
	poetry run isort .
	poetry run ruff check --fix
	poetry run mypy --namespace-packages --explicit-package-bases src

install:
	poetry install

run: install
	poetry run uvicorn aim.main:app --host 0.0.0.0 --port 8000

dev: install
	poetry run uvicorn aim.main:app --host 0.0.0.0 --port 8000 --reload

test: install
	poetry run pytest

test-verbose: install
	poetry run pytest -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

