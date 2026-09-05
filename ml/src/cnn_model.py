"""Deterministic CNN baseline definition for Log-Mel speech inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tensorflow as tf


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "ml" / "config" / "cnn_baseline.json"
EXPECTED_INPUT_SHAPE = (64, 219, 1)
EXPECTED_CLASSES = (
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
)


def load_cnn_config(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the fixed baseline CNN configuration."""
    config = json.loads((path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("version") != "1.0.0":
        raise ValueError("Unsupported CNN baseline configuration version.")
    if tuple(config.get("input_shape", ())) != EXPECTED_INPUT_SHAPE:
        raise ValueError("CNN baseline input shape must be [64, 219, 1].")
    if tuple(config.get("classes", ())) != EXPECTED_CLASSES:
        raise ValueError("CNN baseline class order is invalid.")
    compile_config = config.get("compile", {})
    if compile_config.get("optimizer") != "adam" or compile_config.get("learning_rate") != 0.001:
        raise ValueError("CNN baseline must use Adam with a 0.001 learning rate.")
    if compile_config.get("loss") != "sparse_categorical_crossentropy":
        raise ValueError("CNN baseline loss is invalid.")
    if config.get("random_seed") != 42:
        raise ValueError("CNN baseline random seed must be 42.")
    return config


def build_cnn_baseline(config: dict[str, Any] | None = None) -> tf.keras.Model:
    """Build and compile the fixed eight-class Log-Mel CNN baseline."""
    config = config or load_cnn_config()
    tf.keras.utils.set_random_seed(config["random_seed"])
    inputs = tf.keras.Input(shape=tuple(config["input_shape"]), name="log_mel")
    features = inputs
    for filters, dropout_rate in ((32, 0.20), (64, 0.25), (128, 0.30)):
        features = tf.keras.layers.Conv2D(filters, (3, 3), padding="valid")(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.ReLU()(features)
        features = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(features)
        features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dense(64, activation="relu")(features)
    features = tf.keras.layers.Dropout(0.40)(features)
    outputs = tf.keras.layers.Dense(len(config["classes"]), activation="softmax", name="emotion")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["compile"]["learning_rate"]),
        loss=config["compile"]["loss"],
        metrics=config["compile"]["metrics"],
    )
    return model
