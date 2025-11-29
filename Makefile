.PHONY: help install run dev test test-verbose clean docker-build docker-run docker-stop docker-logs docker-clean

help:
	@echo Available commands:
	@echo   make install         - Install dependencies using Poetry
	@echo   make run             - Run FastAPI application
	@echo   make dev             - Run FastAPI with auto-reload
	@echo   make test            - Run tests with Pytest
	@echo   make test-verbose    - Run tests with verbose output
	@echo   make clean           - Remove cache and temporary files
	@echo   make lint            - Run lint on whole project
	@echo ""
	@echo Docker commands:
	@echo   make docker-build    - Build Docker image
	@echo   make docker-run      - Run application in Docker using docker-compose
	@echo   make docker-stop     - Stop Docker containers
	@echo   make docker-logs     - View Docker container logs
	@echo   make docker-clean    - Remove Docker containers and images
	@echo   make docker-shell    - Open shell in running container

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

# Docker commands
docker-build:
	docker build -t aim-classifier-api:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker rmi aim-classifier-api:latest || true

docker-shell:
	docker-compose exec app /bin/bash

