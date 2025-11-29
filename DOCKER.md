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
| `AZURE_ENDPOINT` | Yes | - | Azure OpenAI endpoint URL |
| `AZURE_MODEL_NAME` | No | `gpt-5-mini-hiring` | Azure model deployment name |
| `AZURE_API_VERSION` | No | `2025-03-01-preview` | Azure API version |

### Volumes

**Model Definitions** (required):
```bash
# Mount model definitions directory
-v ./src/aim/model_definitions:/app/src/aim/model_definitions:ro
```

**Source Code** (development only):
```bash
# Mount source for live code updates
-v ./src:/app/src:ro
```

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
    # Remove development volume mounts
    volumes:
      - ./src/aim/model_definitions:/app/src/aim/model_definitions:ro
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
    "--log-level", "info"]
```

**Recommended workers**: `(2 * CPU_CORES) + 1`

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
# 2. Model files not mounted correctly
# 3. Port 8000 already in use
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
docker exec aim-classifier-api ls -la /app/src/aim/model_definitions/

# Check file permissions
docker exec aim-classifier-api stat /app/src/aim/model_definitions/flag_classifier_project_project_2.json
```

### Performance Issues

```bash
# Check resource usage
docker stats aim-classifier-api

# View running processes
docker top aim-classifier-api

# Increase resources if needed
docker update --cpus="4.0" --memory="4g" aim-classifier-api
```

## Security Best Practices

### Image Security

```bash
# Scan for vulnerabilities
docker scan aim-classifier-api:latest

# Use specific Python version (not latest)
# Already configured: python:3.11-slim

# Run as non-root user
# Already configured: USER appuser
```

### Runtime Security

```bash
# Run with read-only root filesystem
docker run -d \
  --read-only \
  --tmpfs /tmp \
  -p 8000:8000 \
  aim-classifier-api:latest

# Drop unnecessary capabilities
docker run -d \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  -p 8000:8000 \
  aim-classifier-api:latest
```

### Secrets Management

**DO NOT** hardcode secrets in Dockerfile or docker-compose.yml:

```bash
# Use Docker secrets (Swarm mode)
docker secret create aim_openai_key -
# Paste your key, then Ctrl+D

# Or use environment file
docker run -d --env-file .env.production aim-classifier-api:latest
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t aim-classifier-api:${{ github.sha }} .

      - name: Run tests in container
        run: |
          docker run --rm \
            -e AIM_OPENAI_KEY=${{ secrets.AIM_OPENAI_KEY }} \
            aim-classifier-api:${{ github.sha }} \
            pytest tests/

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_TOKEN }} | docker login -u ${{ secrets.DOCKER_USER }} --password-stdin
          docker push aim-classifier-api:${{ github.sha }}
```

## Maintenance

### Update Dependencies

```bash
# Rebuild with updated dependencies
docker-compose build --no-cache

# Or using Docker CLI
docker build --no-cache -t aim-classifier-api:latest .
```

### Clean Up

```bash
# Remove stopped containers
docker-compose down

# Remove images
docker rmi aim-classifier-api:latest

# Clean up build cache
docker builder prune

# Full cleanup (careful!)
docker system prune -a --volumes
```

## Monitoring and Logging

### View Logs

```bash
# Follow logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs -f app

# View last 100 lines
docker-compose logs --tail=100 app
```

### Export Logs

```bash
# Export logs to file
docker logs aim-classifier-api > app.log 2>&1

# Export with timestamps
docker logs -t aim-classifier-api > app.log 2>&1
```

## Advanced Topics

### Multi-Container Setup with Monitoring

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  app:
    # ... existing config ...

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - app-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    networks:
      - app-network
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aim-classifier-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aim-classifier-api
  template:
    metadata:
      labels:
        app: aim-classifier-api
    spec:
      containers:
      - name: api
        image: aim-classifier-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: AIM_OPENAI_KEY
          valueFrom:
            secretKeyRef:
              name: azure-openai-secrets
              key: api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "1"
          limits:
            memory: "2Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Support

For issues or questions:
1. Check container logs: `docker logs aim-classifier-api`
2. Review this documentation
3. Check application health: `curl http://localhost:8000/health`
4. Verify environment variables are set correctly
