"""Reusable single-audio inference for the frozen selected CNN."""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(ROOT / "ml" / ".cache" / "matplotlib")

import numpy as np
import soundfile as sf
import tensorflow as tf

try:  # Supports both ``python -m ml.src.predict_audio`` and source-path imports.
    from .audio_features import extract_log_mel_spectrogram, load_feature_config
    from .audio_preprocessing import load_preprocessing_config, preprocess_audio_file
    from .cnn_model import EXPECTED_CLASSES
except ImportError:  # pragma: no cover - exercised by the existing source-path test setup.
    from audio_features import extract_log_mel_spectrogram, load_feature_config
    from audio_preprocessing import load_preprocessing_config, preprocess_audio_file
    from cnn_model import EXPECTED_CLASSES


MODEL_PATH = ROOT / "ml" / "models" / "cnn" / "cnn_reduced_regularization.keras"
NORMALIZATION_PATH = ROOT / "ml" / "data" / "processed" / "cnn_baseline" / "normalization.npz"
MODEL_CONFIG_PATH = ROOT / "ml" / "config" / "cnn_reduced_regularization.json"
MODEL_IDENTIFIER = "cnn_reduced_regularization_no_dropout"
EXPECTED_FEATURE_SHAPE = (64, 219, 1)

_resource_lock = threading.Lock()
_resources: tuple[tf.keras.Model, np.ndarray, np.ndarray, str] | None = None


class PredictionInputError(ValueError):
    """Raised when a supplied recording cannot be processed safely."""


class PredictionArtifactError(RuntimeError):
    """Raised when a required frozen inference artifact is missing or invalid."""


class PredictionIntegrityError(RuntimeError):
    """Raised when a model input or output breaks the inference contract."""


def _load_normalization(path: Path = NORMALIZATION_PATH) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise PredictionArtifactError(f"Training normalization artifact is missing: {path}")
    try:
        with np.load(path) as data:
            mean = np.asarray(data["mean"], dtype=np.float32)
            std = np.asarray(data["std"], dtype=np.float32)
            epsilon = float(data["epsilon"])
    except (OSError, KeyError, ValueError) as error:
        raise PredictionArtifactError(f"Training normalization artifact is unreadable: {path}") from error
    if mean.shape != (64,) or std.shape != (64,) or not np.isclose(epsilon, 1e-6, rtol=0, atol=1e-12):
        raise PredictionArtifactError("Training normalization statistics do not match the frozen CNN contract.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise PredictionArtifactError("Training normalization statistics contain invalid values.")
    return mean, std


def _load_model(path: Path = MODEL_PATH) -> tf.keras.Model:
    if not path.is_file():
        raise PredictionArtifactError(f"Frozen CNN model artifact is missing: {path}")
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except (OSError, ValueError, tf.errors.OpError) as error:
        raise PredictionArtifactError(f"Frozen CNN model artifact cannot be loaded: {path}") from error
    if model.input_shape != (None, *EXPECTED_FEATURE_SHAPE) or model.output_shape != (None, len(EXPECTED_CLASSES)):
        raise PredictionArtifactError("Frozen CNN model input or output shape is invalid.")
    if model.count_params() != 102344 or any(layer.rate != 0.0 for layer in model.layers if layer.__class__.__name__ == "Dropout"):
        raise PredictionArtifactError("Model artifact is not the selected reduced-regularization/no-dropout CNN.")
    return model


def _model_version() -> str:
    config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("model_variant") != "cnn_reduced_regularization" or config.get("version") != "1.0.0":
        raise PredictionArtifactError("Frozen CNN configuration is invalid.")
    return str(config["version"])


def load_prediction_resources() -> tuple[tf.keras.Model, np.ndarray, np.ndarray, str]:
    """Lazily load and reuse the frozen CNN plus training-only normalization."""
    global _resources
    with _resource_lock:
        if _resources is None:
            model = _load_model()
            mean, std = _load_normalization()
            _resources = (model, mean, std, _model_version())
    return _resources


def clear_prediction_resource_cache() -> None:
    """Clear process-local resources, primarily for isolated tests or controlled shutdown."""
    global _resources
    with _resource_lock:
        _resources = None


def prepare_audio_file(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen preprocessing, Log-Mel extraction, and training normalization."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise PredictionInputError(f"Audio file was not found: {audio_path}")
    _, mean, std, _ = load_prediction_resources()
    try:
        waveform, audit = preprocess_audio_file(audio_path, load_preprocessing_config())
        log_mel = extract_log_mel_spectrogram(waveform, config=load_feature_config())
    except (OSError, ValueError, sf.LibsndfileError) as error:
        raise PredictionInputError(f"Audio file cannot be processed: {audio_path}") from error
    if waveform.shape != (56000,) or waveform.dtype != np.float32 or not np.isfinite(waveform).all():
        raise PredictionIntegrityError("Frozen preprocessing did not produce a finite 56,000-sample waveform.")
    if log_mel.shape != EXPECTED_FEATURE_SHAPE[:2] or log_mel.dtype != np.float32 or not np.isfinite(log_mel).all():
        raise PredictionIntegrityError("Frozen feature extraction did not produce a finite 64x219 Log-Mel matrix.")
    normalized = np.ascontiguousarray((log_mel - mean[:, None]) / std[:, None], dtype=np.float32)
    feature = np.ascontiguousarray(normalized[:, :, None][None, ...], dtype=np.float32)
    if feature.shape != (1, *EXPECTED_FEATURE_SHAPE) or not np.isfinite(feature).all():
        raise PredictionIntegrityError("Normalized CNN feature has an invalid shape or non-finite values.")
    return feature, audit


def build_prediction_result(probabilities: np.ndarray, model_version: str) -> dict[str, Any]:
    """Validate one softmax row and return an API-ready prediction dictionary."""
    values = np.asarray(probabilities, dtype=np.float32)
    if values.shape == (1, len(EXPECTED_CLASSES)):
        values = values[0]
    if values.shape != (len(EXPECTED_CLASSES),) or not np.isfinite(values).all():
        raise PredictionIntegrityError("CNN output must contain eight finite probabilities for one recording.")
    if not np.allclose(float(values.sum()), 1.0, rtol=0, atol=1e-6):
        raise PredictionIntegrityError("CNN output probabilities do not sum to one.")
    index = int(np.argmax(values))
    probability_map = {label: float(values[position]) for position, label in enumerate(EXPECTED_CLASSES)}
    return {
        "emotion": EXPECTED_CLASSES[index],
        "class_index": index,
        "confidence": float(values[index]),
        "probabilities": probability_map,
        "model": {"identifier": MODEL_IDENTIFIER, "version": model_version},
    }


def predict_audio_file(path: str | Path) -> dict[str, Any]:
    """Predict one readable PCM_16 WAV recording with the frozen selected CNN."""
    feature, _ = prepare_audio_file(path)
    model, _, _, version = load_prediction_resources()
    probabilities = np.asarray(model(feature, training=False).numpy(), dtype=np.float32)
    return build_prediction_result(probabilities, version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one local prediction with the frozen speech-emotion CNN.")
    parser.add_argument("audio_path", type=Path, help="Path to a readable PCM_16 WAV file.")
    arguments = parser.parse_args()
    print(json.dumps(predict_audio_file(arguments.audio_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
