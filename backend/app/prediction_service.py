"""Lazy adapter for the frozen local audio-prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


class PredictionInputUnavailableError(ValueError):
    """The submitted recording cannot be processed as valid audio."""


class PredictionArtifactsUnavailableError(RuntimeError):
    """The selected model or its required normalization data is unavailable."""


PredictionFunction = Callable[[Path], dict[str, Any]]


def predict_audio(path: Path) -> dict[str, Any]:
    """Run local inference, importing TensorFlow only when prediction is requested."""
    from ml.src.predict_audio import (
        PredictionArtifactError,
        PredictionInputError,
        PredictionIntegrityError,
        predict_audio_file,
    )

    try:
        return predict_audio_file(path)
    except PredictionInputError as error:
        raise PredictionInputUnavailableError("The uploaded audio could not be processed.") from error
    except (PredictionArtifactError, PredictionIntegrityError) as error:
        raise PredictionArtifactsUnavailableError("Prediction artifacts are unavailable or invalid.") from error
