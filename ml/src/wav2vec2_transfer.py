"""Shared validation and pooling utilities for the frozen Wav2Vec2 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "ml" / "config" / "wav2vec2_transfer.json"
EXPECTED_LABELS = (
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
)


def load_transfer_config(path: Path | None = None) -> dict[str, Any]:
    """Load the fixed transfer-learning configuration and validate its contract."""
    config = json.loads((path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("version") != "1.0.0":
        raise ValueError("Unsupported transfer-learning configuration version.")
    if config.get("model_identifier") != "facebook/wav2vec2-base" or not config.get("encoder_frozen"):
        raise ValueError("The transfer experiment requires the frozen facebook/wav2vec2-base encoder.")
    if tuple(config.get("emotion_labels", ())) != EXPECTED_LABELS:
        raise ValueError("The transfer emotion-label order is invalid.")
    if config.get("input", {}).get("sample_rate") != 16000 or config["input"].get("waveform_samples") != 56000:
        raise ValueError("The transfer input contract must be 16 kHz and 56,000 samples.")
    if config.get("embedding", {}).get("hidden_size") != 768 or config["embedding"].get("shape") != [1536]:
        raise ValueError("The transfer embedding contract is invalid.")
    if config.get("classifier", {}).get("C") != 1.0 or config["classifier"].get("class_weight") != "balanced":
        raise ValueError("The fixed transfer classifier configuration is invalid.")
    return config


def validate_waveform(waveform: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Return a contiguous finite mono waveform without modifying the caller's array."""
    array = np.asarray(waveform)
    expected_samples = int(config["input"]["waveform_samples"])
    if array.ndim != 1 or array.shape != (expected_samples,):
        raise ValueError(f"Expected a mono waveform with shape ({expected_samples},).")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Waveform samples must be finite numerical values.")
    return np.ascontiguousarray(array, dtype=np.float32)


def pool_hidden_states(hidden_states: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    """Pool final Wav2Vec2 states into mean and population-standard-deviation features."""
    config = config or load_transfer_config()
    states = np.asarray(hidden_states)
    hidden_size = int(config["embedding"]["hidden_size"])
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] != hidden_size:
        raise ValueError(f"Expected non-empty final hidden states shaped (frames, {hidden_size}).")
    if not np.issubdtype(states.dtype, np.number) or not np.isfinite(states).all():
        raise ValueError("Final hidden states must be finite numerical values.")
    pooled = np.concatenate((states.mean(axis=0, dtype=np.float64), states.std(axis=0, dtype=np.float64, ddof=0)))
    result = np.ascontiguousarray(pooled, dtype=np.float32)
    if result.shape != (1536,) or not np.isfinite(result).all():
        raise ValueError("Invalid pooled Wav2Vec2 embedding.")
    return result
