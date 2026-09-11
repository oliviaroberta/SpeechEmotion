"""FastAPI application foundation without model-loading side effects."""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from .prediction_service import (
    PredictionArtifactsUnavailableError,
    PredictionFunction,
    PredictionInputUnavailableError,
    predict_audio,
)
from .schemas import PredictionResponse


API_TITLE = "Speech Emotion Recognition API"
API_VERSION = "0.1.0"
DEFAULT_ALLOWED_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
DEFAULT_MAX_AUDIO_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 64 * 1024


class HealthResponse(BaseModel):
    """Stable response returned by the service readiness probe."""

    status: str
    service: str
    version: str


def get_allowed_origins() -> list[str]:
    """Read a validated CORS allowlist with development-only local defaults."""
    configured = os.environ.get("ALLOWED_ORIGINS") or os.environ.get("SER_ALLOWED_ORIGINS", "")
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    if not origins:
        return [] if os.environ.get("SER_ENVIRONMENT", "development").lower() == "production" else list(DEFAULT_ALLOWED_ORIGINS)
    for origin in origins:
        parsed = urlparse(origin)
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.params
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("ALLOWED_ORIGINS must contain comma-separated http(s) origins without paths or credentials.")
    return origins


def get_max_audio_bytes() -> int:
    """Read a positive request-size limit, retaining the safe default on bad input."""
    try:
        configured = int(os.environ.get("SER_MAX_AUDIO_BYTES", str(DEFAULT_MAX_AUDIO_BYTES)))
    except ValueError:
        return DEFAULT_MAX_AUDIO_BYTES
    return configured if configured > 0 else DEFAULT_MAX_AUDIO_BYTES


def get_prediction_function() -> PredictionFunction:
    """Expose the real predictor as an overrideable dependency for API tests."""
    return predict_audio


def _is_riff_wave_header(header: bytes) -> bool:
    return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"


def _validate_wav_container(path: Path) -> None:
    """Reject malformed RIFF/WAVE containers before invoking the ML pipeline."""
    try:
        with wave.open(str(path), "rb") as recording:
            if recording.getnframes() <= 0:
                raise wave.Error("WAV file contains no audio frames")
    except (EOFError, OSError, wave.Error) as error:
        raise PredictionInputUnavailableError("The uploaded WAV is malformed or unreadable.") from error


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

    @application.post("/api/v1/predict", response_model=PredictionResponse, tags=["prediction"])
    async def predict(
        file: Annotated[UploadFile | None, File()] = None,
        predictor: PredictionFunction = Depends(get_prediction_function),
    ) -> PredictionResponse:
        """Stream one WAV upload through the frozen CPU prediction pipeline."""
        if file is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="An audio file is required.")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
                temporary_path = Path(temporary_file.name)
                total_bytes = 0
                header = bytearray()
                while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                    total_bytes += len(chunk)
                    if total_bytes > get_max_audio_bytes():
                        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Audio upload exceeds the size limit.")
                    if len(header) < 12:
                        header.extend(chunk[: 12 - len(header)])
                    temporary_file.write(chunk)

            if total_bytes == 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio upload is empty.")
            if not _is_riff_wave_header(bytes(header)):
                raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only RIFF/WAVE audio is supported.")

            _validate_wav_container(temporary_path)
            result = await run_in_threadpool(predictor, temporary_path)
            try:
                return PredictionResponse.model_validate(result)
            except ValidationError as error:
                raise RuntimeError("The prediction service returned an invalid result.") from error
        except HTTPException:
            raise
        except PredictionInputUnavailableError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The WAV audio could not be processed.") from None
        except PredictionArtifactsUnavailableError:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Prediction service is unavailable.") from None
        except Exception:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed unexpectedly.") from None
        finally:
            await file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return application


app = create_app()
