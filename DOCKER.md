# Docker Deployment Guide

This document provides comprehensive instructions for building and running the AIM Classifier API using Docker.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# 1. Create .env file with your Azure OpenAI credentials
cp .env.example .env
# Edit .env and add your AIM_OPENAI_KEY

# 2. Build and run the application
docker-compose up -d

# 3. Check application health
curl http://localhost:8000/health

# 4. View logs
docker-compose logs -f

# 5. Stop the application
docker-compose down
```

### Using Docker CLI

```bash
# 1. Build the image
docker build -t aim-classifier-api:latest .

# 2. Run the container
docker run -d \
  --name aim-classifier-api \
  -p 8000:8000 \
  -e AIM_OPENAI_KEY="your-api-key" \
  -e AZURE_ENDPOINT="https://aim-australia-east.openai.azure.com/" \
  -e AZURE_MODEL_NAME="gpt-5-mini-hiring" \
  -e AZURE_API_VERSION="2025-03-01-preview" \
  aim-classifier-api:latest

# 3. Check logs
docker logs -f aim-classifier-api

# 4. Stop and remove container
docker stop aim-classifier-api
docker rm aim-classifier-api
```

## Docker Image Architecture

### Multi-Stage Build

The Dockerfile uses a multi-stage build for optimal image size and security:

**Stage 1: Builder**
- Base: `python:3.11-slim`
- Installs Poetry and build dependencies
- Installs Python dependencies (production only)
- Result: Virtual environment with all runtime dependencies

**Stage 2: Runtime**
- Base: `python:3.11-slim`
- Copies virtual environment from builder stage
- Copies application code
- Runs as non-root user (`appuser`)
- Minimal attack surface

### Image Optimizations

- **Multi-stage build**: Separates build dependencies from runtime
- **Poetry layer caching**: Dependencies installed before code copy
- **Production dependencies only**: No dev/test dependencies in image
- **Non-root user**: Enhanced security by running as `appuser`
- **.dockerignore**: Excludes unnecessary files from build context
- **Health checks**: Automatic container health monitoring

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AIM_OPENAI_KEY` | Yes | - | Azure OpenAI API key |
| `AZURE_ENDPOINT` | No | `https://aim-australia-east.openai.azure.com/` | Azure OpenAI endpoint URL |
| `AZURE_MODEL_NAME` | No | `gpt-5-mini-hiring` | Azure model deployment name |
| `AZURE_API_VERSION` | No | `2025-03-01-preview` | Azure API version |

**Note**: Model definitions are baked into the Docker image during build. The Dockerfile copies `src/aim/` to `/app/aim/`, including all model definition files.

### Network Configuration

The docker-compose.yml creates a custom bridge network (`app-network`) for container communication:

```yaml
networks:
  app-network:
    driver: bridge
```

This allows you to add additional services (databases, monitoring, etc.) that can communicate with the API container by name.

### Restart Policy

The default restart policy is `unless-stopped`, meaning the container will automatically restart unless explicitly stopped:

```yaml
restart: unless-stopped
```

For production, you may want to use `restart: always` in your production override file.

## Production Deployment

### Build Optimization

```bash
# Build with buildkit for better caching
DOCKER_BUILDKIT=1 docker build -t aim-classifier-api:v0.1.0 .

# Build with specific platform (for ARM/M1 Macs)
docker build --platform linux/amd64 -t aim-classifier-api:v0.1.0 .

# Scan for vulnerabilities
docker scan aim-classifier-api:v0.1.0
```

### Running in Production

```bash
# Production docker-compose with resource limits
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**docker-compose.prod.yml** (create this for production overrides):
```yaml
version: '3.8'

services:
  app:
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Health Monitoring

```bash
# Check container health status
docker inspect --format='{{.State.Health.Status}}' aim-classifier-api

# View health check logs
docker inspect --format='{{json .State.Health}}' aim-classifier-api | jq
```

## Performance Tuning

### Worker Configuration

The default configuration uses 4 Uvicorn workers. Adjust based on your CPU cores:

```dockerfile
# In Dockerfile, modify CMD:
CMD ["uvicorn", "aim.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "8",  # Adjust based on CPU cores
    "--log-level", "info", \
    "--no-access-log"]  # Disables access logs for better performance
```

**Recommended workers**: `(2 * CPU_CORES) + 1`

**Note**: The `--no-access-log` flag is enabled by default to reduce log verbosity and improve performance. Remove it if you need detailed access logs.

### Resource Limits

```bash
# Run with CPU and memory limits
docker run -d \
  --cpus="2.0" \
  --memory="2g" \
  --memory-swap="2g" \
  -p 8000:8000 \
  aim-classifier-api:latest
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs for errors
docker logs aim-classifier-api

# Common issues:
# 1. Missing environment variables
# 2. Port 8000 already in use
```

### Health Check Failing

```bash
# Test health endpoint manually
docker exec aim-classifier-api curl http://localhost:8000/health

# Check if models loaded
docker exec aim-classifier-api python -c "from aim.main import app; print(app.state.models)"
```

### Model Loading Issues

```bash
# Verify model files are accessible
docker exec aim-classifier-api ls -la /app/aim/model_definitions/

# Check file permissions
docker exec aim-classifier-api stat /app/aim/model_definitions/flag_classifier_project_project_2.json
```