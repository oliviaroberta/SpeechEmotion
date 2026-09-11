"""Vercel ASGI entry point for the frozen FastAPI inference service."""

from backend.app.main import app

__all__ = ["app"]
