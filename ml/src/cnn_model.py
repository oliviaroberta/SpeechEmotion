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


@tf.keras.utils.register_keras_serializable(package="speech_emotion")
class SpectrogramMasking(tf.keras.layers.Layer):
    """Apply one deterministic frequency and time mask during training only."""

    def __init__(self, maximum_frequency_width: int = 6, maximum_time_width: int = 18, mask_value: float = 0.0, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.maximum_frequency_width = maximum_frequency_width
        self.maximum_time_width = maximum_time_width
        self.mask_value = mask_value
        self.seed = seed

    def call(self, inputs: tf.Tensor, training: bool | tf.Tensor | None = None) -> tf.Tensor:
        def apply_masks() -> tf.Tensor:
            frequency_width = tf.random.stateless_uniform([], seed=(self.seed, 0), minval=1, maxval=self.maximum_frequency_width + 1, dtype=tf.int32)
            frequency_start = tf.random.stateless_uniform([], seed=(self.seed, 1), minval=0, maxval=65 - frequency_width, dtype=tf.int32)
            time_width = tf.random.stateless_uniform([], seed=(self.seed, 2), minval=1, maxval=self.maximum_time_width + 1, dtype=tf.int32)
            time_start = tf.random.stateless_uniform([], seed=(self.seed, 3), minval=0, maxval=220 - time_width, dtype=tf.int32)
            frequencies = tf.range(64)
            times = tf.range(219)
            frequency_keep = tf.logical_or(frequencies < frequency_start, frequencies >= frequency_start + frequency_width)
            time_keep = tf.logical_or(times < time_start, times >= time_start + time_width)
            mask = tf.cast(tf.reshape(frequency_keep, (1, 64, 1, 1)), inputs.dtype) * tf.cast(tf.reshape(time_keep, (1, 1, 219, 1)), inputs.dtype)
            return inputs * mask + tf.cast(self.mask_value, inputs.dtype) * (1 - mask)

        if training is None:
            return inputs
        if isinstance(training, bool):
            return apply_masks() if training else inputs
        return tf.cond(tf.cast(training, tf.bool), apply_masks, lambda: inputs)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "maximum_frequency_width": self.maximum_frequency_width, "maximum_time_width": self.maximum_time_width, "mask_value": self.mask_value, "seed": self.seed}


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


def build_cnn_reduced_regularization(config: dict[str, Any] | None = None) -> tf.keras.Model:
    """Build the baseline architecture with every diagnostic dropout rate set to zero."""
    config = config or load_cnn_config()
    tf.keras.utils.set_random_seed(config["random_seed"])
    inputs = tf.keras.Input(shape=tuple(config["input_shape"]), name="log_mel")
    features = inputs
    for filters in (32, 64, 128):
        features = tf.keras.layers.Conv2D(filters, (3, 3), padding="valid")(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.ReLU()(features)
        features = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(features)
        features = tf.keras.layers.Dropout(0.0)(features)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dense(64, activation="relu")(features)
    features = tf.keras.layers.Dropout(0.0)(features)
    outputs = tf.keras.layers.Dense(len(config["classes"]), activation="softmax", name="emotion")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_reduced_regularization")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["compile"]["learning_rate"]), loss=config["compile"]["loss"], metrics=config["compile"]["metrics"])
    return model


def build_cnn_regularized_augmented(config: dict[str, Any] | None = None) -> tf.keras.Model:
    """Build the mild-dropout CNN with serializable training-only spectrogram masking."""
    config = config or load_cnn_config()
    tf.keras.utils.set_random_seed(config["random_seed"])
    inputs = tf.keras.Input(shape=tuple(config["input_shape"]), name="log_mel")
    features = SpectrogramMasking()(inputs)
    for filters, dropout_rate in ((32, 0.05), (64, 0.10), (128, 0.15)):
        features = tf.keras.layers.Conv2D(filters, (3, 3), padding="valid")(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.ReLU()(features)
        features = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(features)
        features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dense(64, activation="relu")(features)
    features = tf.keras.layers.Dropout(0.20)(features)
    outputs = tf.keras.layers.Dense(len(config["classes"]), activation="softmax", name="emotion")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_regularized_augmented")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["compile"]["learning_rate"]), loss=config["compile"]["loss"], metrics=config["compile"]["metrics"])
    return model
