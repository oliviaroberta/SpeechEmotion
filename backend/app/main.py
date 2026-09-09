"""FastAPI application foundation without model-loading side effects."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


API_TITLE = "Speech Emotion Recognition API"
API_VERSION = "0.1.0"
DEFAULT_ALLOWED_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


class HealthResponse(BaseModel):
    """Stable response returned by the service readiness probe."""

    status: str
    service: str
    version: str


def get_allowed_origins() -> list[str]:
    """Read a comma-separated CORS allowlist with safe local development defaults."""
    configured = os.environ.get("SER_ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or list(DEFAULT_ALLOWED_ORIGINS)


def create_app() -> FastAPI:
    """Create the API without importing TensorFlow or the selected CNN model."""
    application = FastAPI(title=API_TITLE, version=API_VERSION)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            service="speech-emotion-recognition-api",
            version=API_VERSION,
        )

    return application


app = create_app()
