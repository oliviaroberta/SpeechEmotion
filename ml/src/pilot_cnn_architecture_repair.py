"""Evaluate two in-memory CNN architecture repairs on a balanced training subset."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf

from cnn_model import EXPECTED_CLASSES


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data" / "processed" / "cnn_baseline"
REPORT_PATH = ROOT / "ml" / "metadata" / "cnn_architecture_repair_pilot.json"
CONTROL_ACCURACY = 0.28125


def feature_fingerprint(feature: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(feature, dtype="<f4").tobytes()).hexdigest()


def convolutional_blocks(inputs: tf.Tensor, dropout_rates: tuple[float, float, float]) -> tf.Tensor:
    features = inputs
    for filters, dropout_rate in zip((32, 64, 128), dropout_rates):
        features = tf.keras.layers.Conv2D(filters, (3, 3), padding="valid")(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.ReLU()(features)
        features = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(features)
        features = tf.keras.layers.Dropout(dropout_rate)(features)
    return features


def compile_model(inputs: tf.Tensor, outputs: tf.Tensor, name: str) -> tf.keras.Model:
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_candidate_a() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(64, 219, 1), name="log_mel")
    features = convolutional_blocks(inputs, (0.0, 0.0, 0.0))
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dense(64, activation="relu")(features)
    features = tf.keras.layers.Dropout(0.0)(features)
    outputs = tf.keras.layers.Dense(8, activation="softmax", name="emotion")(features)
    return compile_model(inputs, outputs, "cnn_repair_reduced_regularisation")


def build_candidate_b() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(64, 219, 1), name="log_mel")
    features = convolutional_blocks(inputs, (0.10, 0.10, 0.10))
    features = tf.keras.layers.AveragePooling2D(pool_size=(2, 3))(features)
    features = tf.keras.layers.Flatten()(features)
    features = tf.keras.layers.Dense(64, activation="relu")(features)
    features = tf.keras.layers.Dropout(0.20)(features)
    outputs = tf.keras.layers.Dense(8, activation="softmax", name="emotion")(features)
    return compile_model(inputs, outputs, "cnn_repair_spatial_information")


class StopOnAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        super().__init__()
        self.features = features
        self.labels = labels
        self.evaluation_accuracy: list[float] = []
        self.evaluation_loss: list[float] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        metrics = self.model.evaluate(self.features, self.labels, batch_size=8, verbose=0, return_dict=True)
        accuracy = float(metrics["accuracy"])
        self.evaluation_accuracy.append(accuracy)
        self.evaluation_loss.append(float(metrics["loss"]))
        if accuracy >= 0.95:
            self.model.stop_training = True


def run_candidate(name: str, factory: Callable[[], tf.keras.Model], features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = factory()
    callback = StopOnAccuracy(features, labels)
    started = time.perf_counter()
    history = model.fit(features, labels, batch_size=8, epochs=200, verbose=0, callbacks=[callback], shuffle=True)
    final_metrics = model.evaluate(features, labels, batch_size=8, verbose=0, return_dict=True)
    result = {
        "name": name,
        "parameter_count": int(model.count_params()),
        "batch_size": 8,
        "maximum_epochs": 200,
        "epochs_completed": len(history.history["loss"]),
        "final_loss": float(final_metrics["loss"]),
        "final_accuracy": float(final_metrics["accuracy"]),
        "reached_95_percent": float(final_metrics["accuracy"]) >= 0.95,
        "training_duration_seconds": time.perf_counter() - started,
        "training_history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "evaluation_accuracy_history": callback.evaluation_accuracy,
        "evaluation_loss_history": callback.evaluation_loss,
        "model_saved": False,
    }
    del model
    return result


def main() -> None:
    features = np.load(DATA_DIR / "train_log_mel.npy")
    labels = np.load(DATA_DIR / "train_labels.npy")
    if features.shape != (960, 64, 219, 1) or labels.shape != (960,):
        raise RuntimeError("Unexpected prepared training-array shapes.")
    indices = np.concatenate([np.flatnonzero(labels == class_id)[:4] for class_id in range(8)])
    subset = features[indices]
    subset_labels = labels[indices]
    fingerprints = [feature_fingerprint(feature) for feature in subset]
    if not np.isfinite(subset).all() or any(np.ptp(feature) == 0 for feature in subset):
        raise RuntimeError("The balanced pilot subset contains non-finite or constant features.")
    if len(set(fingerprints)) != 32:
        raise RuntimeError("The balanced pilot subset contains duplicate feature arrays.")
    if not np.array_equal(np.bincount(subset_labels, minlength=8), np.full(8, 4)):
        raise RuntimeError("The pilot subset is not balanced across eight emotions.")

    candidate_a = run_candidate("candidate_a_reduced_regularisation", build_candidate_a, subset, subset_labels)
    candidate_b = run_candidate("candidate_b_preserve_spatial_information", build_candidate_b, subset, subset_labels)
    successes = [candidate for candidate in (candidate_a, candidate_b) if candidate["reached_95_percent"]]
    if candidate_a["reached_95_percent"] and candidate_b["reached_95_percent"]:
        recommendation = min(successes, key=lambda candidate: (candidate["parameter_count"], candidate["epochs_completed"]))["name"]
        likely_cause = "Excessive regularisation is likely involved because Candidate A succeeded; Candidate B also succeeds, so spatial compression is not necessary to explain the repair."
    elif candidate_a["reached_95_percent"]:
        recommendation = candidate_a["name"]
        likely_cause = "Excessive regularisation is likely involved because Candidate A succeeded."
    elif candidate_b["reached_95_percent"]:
        recommendation = candidate_b["name"]
        likely_cause = "Spatial compression is likely the main problem because Candidate B succeeded while Candidate A failed."
    else:
        recommendation = None
        likely_cause = "Neither candidate memorised the subset; further pipeline investigation is required."
    report = {
        "control_baseline_memorisation_accuracy": CONTROL_ACCURACY,
        "subset": {
            "source": "training arrays only", "size": 32, "examples_per_emotion": 4,
            "class_order": list(EXPECTED_CLASSES), "indices": [int(index) for index in indices],
            "all_finite": True, "all_non_constant": True, "duplicates_found": False,
            "feature_fingerprints": fingerprints,
        },
        "candidates": {"candidate_a": candidate_a, "candidate_b": candidate_b},
        "likely_cause": likely_cause,
        "provisional_recommendation": recommendation,
        "validation_data_accessed": False,
        "test_actors_21_to_24_accessed": False,
        "existing_model_or_report_modified": False,
        "temporary_models_saved": False,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CNN architecture repair pilot completed: {recommendation or 'no candidate selected'}")


if __name__ == "__main__":
    main()
