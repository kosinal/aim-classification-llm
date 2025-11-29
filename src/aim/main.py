"""Main FastAPI application."""

from fastapi import FastAPI


app = FastAPI(title="Aim Home Assignment", description="Aim Home Assignment", version="0.1.0")


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {"message": "Hello World"}


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}
