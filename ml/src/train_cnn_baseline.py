"""Train the fixed CNN baseline using prepared train and validation arrays only."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(ROOT / "ml" / ".cache" / "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from cnn_model import EXPECTED_CLASSES, build_cnn_baseline, load_cnn_config


DATA_DIR = ROOT / "ml" / "data" / "processed" / "cnn_baseline"
DATA_SUMMARY_PATH = ROOT / "ml" / "metadata" / "cnn_dataset_summary.json"
TRAINING_CONFIG_PATH = ROOT / "ml" / "config" / "cnn_training.json"
MODEL_PATH = ROOT / "ml" / "models" / "cnn" / "cnn_baseline.keras"
HISTORY_FIGURE_PATH = ROOT / "ml" / "reports" / "figures" / "cnn_baseline_training_history.png"
CONFUSION_FIGURE_PATH = ROOT / "ml" / "reports" / "figures" / "cnn_baseline_validation_confusion_matrix.png"
REPORT_PATH = ROOT / "ml" / "metadata" / "cnn_baseline_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_config() -> dict[str, Any]:
    config = json.loads(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    if config != {
        "batch_size": 32,
        "class_weights": "balanced",
        "early_stopping": {"monitor": "val_loss", "patience": 6, "restore_best_weights": True},
        "maximum_epochs": 40,
        "random_seed": 42,
        "reduce_learning_rate": {"factor": 0.5, "minimum_learning_rate": 1e-5, "monitor": "val_loss", "patience": 3},
        "version": "1.0.0",
    }:
        raise ValueError("CNN training configuration differs from the approved baseline.")
    return config


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_features = np.load(DATA_DIR / "train_log_mel.npy")
    train_labels = np.load(DATA_DIR / "train_labels.npy")
    validation_features = np.load(DATA_DIR / "validation_log_mel.npy")
    validation_labels = np.load(DATA_DIR / "validation_labels.npy")
    if train_features.shape != (960, 64, 219, 1) or validation_features.shape != (240, 64, 219, 1):
        raise RuntimeError("Prepared CNN feature shapes are invalid.")
    if train_features.dtype != np.float32 or validation_features.dtype != np.float32:
        raise RuntimeError("Prepared CNN feature dtype is invalid.")
    if not all(np.isfinite(values).all() for values in (train_features, validation_features)):
        raise RuntimeError("Prepared CNN features contain non-finite values.")
    if not all(np.issubdtype(values.dtype, np.integer) and values.min() >= 0 and values.max() <= 7 for values in (train_labels, validation_labels)):
        raise RuntimeError("Prepared CNN labels are invalid.")
    return train_features, train_labels, validation_features, validation_labels


def balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(classes, np.arange(8)):
        raise RuntimeError("Training labels do not include all eight classes.")
    return {int(class_id): float(len(labels) / (len(classes) * count)) for class_id, count in zip(classes, counts)}


def write_history_figure(history: dict[str, list[float]]) -> None:
    HISTORY_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["loss"], label="Training")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="CNN baseline loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["accuracy"], label="Training")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set(title="CNN baseline accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(HISTORY_FIGURE_PATH, dpi=150)
    plt.close(figure)


def write_confusion_figure(matrix: np.ndarray) -> None:
    CONFUSION_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Recording count")
    axis.set(xticks=range(8), yticks=range(8), xticklabels=EXPECTED_CLASSES, yticklabels=EXPECTED_CLASSES, xlabel="Predicted emotion", ylabel="Actual emotion", title="CNN baseline validation confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2
    for row_index in range(8):
        for column_index in range(8):
            count = matrix[row_index, column_index]
            axis.text(column_index, row_index, str(count), ha="center", va="center", color="white" if count > threshold else "black")
    figure.tight_layout()
    figure.savefig(CONFUSION_FIGURE_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    config = load_training_config()
    dataset_summary = json.loads(DATA_SUMMARY_PATH.read_text(encoding="utf-8"))
    if dataset_summary["test_actors_accessed"] or dataset_summary["model_trained"]:
        raise RuntimeError("Prepared CNN dataset safety status is invalid.")
    if MODEL_PATH.exists():
        raise RuntimeError(f"Refusing to overwrite existing CNN candidate model: {MODEL_PATH}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_features, train_labels, validation_features, validation_labels = load_arrays()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["random_seed"])
    model = build_cnn_baseline(load_cnn_config())
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["early_stopping"]["patience"], restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=config["reduce_learning_rate"]["patience"], factor=config["reduce_learning_rate"]["factor"], min_lr=config["reduce_learning_rate"]["minimum_learning_rate"]),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
    ]
    started = time.perf_counter()
    history = model.fit(train_features, train_labels, validation_data=(validation_features, validation_labels), batch_size=config["batch_size"], epochs=config["maximum_epochs"], class_weight=balanced_class_weights(train_labels), callbacks=callbacks, verbose=2)
    duration_seconds = time.perf_counter() - started
    probabilities = model.predict(validation_features, batch_size=config["batch_size"], verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    reloaded_model = tf.keras.models.load_model(MODEL_PATH)
    reloaded_probabilities = reloaded_model.predict(validation_features, batch_size=config["batch_size"], verbose=0)
    max_reload_difference = float(np.max(np.abs(probabilities - reloaded_probabilities)))
    if not np.allclose(probabilities, reloaded_probabilities, rtol=0, atol=1e-7):
        raise RuntimeError("Reloaded CNN validation predictions differ from the in-memory best model.")
    matrix = confusion_matrix(validation_labels, predictions, labels=range(8))
    precision, recall, f1, support = precision_recall_fscore_support(validation_labels, predictions, labels=range(8), zero_division=0)
    macro = precision_recall_fscore_support(validation_labels, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(validation_labels, predictions, average="weighted", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(validation_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(validation_labels, predictions)),
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
    }
    history_values = {key: [float(value) for value in values] for key, values in history.history.items()}
    best_epoch = int(np.argmin(history_values["val_loss"]) + 1)
    write_history_figure(history_values)
    write_confusion_figure(matrix)
    report = {
        "configuration_hashes": {"cnn_architecture_configuration_hash": sha256_file(ROOT / "ml/config/cnn_baseline.json"), "cnn_dataset_configuration_hash": sha256_file(ROOT / "ml/config/cnn_dataset.json"), "cnn_training_configuration_hash": sha256_file(TRAINING_CONFIG_PATH), "dataset_summary_hash": sha256_file(DATA_SUMMARY_PATH)},
        "actor_ids": {"train": [f"{actor:02d}" for actor in range(1, 17)], "validation": [f"{actor:02d}" for actor in range(17, 21)], "test_excluded": True},
        "feature_shapes": {"train": list(train_features.shape), "validation": list(validation_features.shape)},
        "class_weights": balanced_class_weights(train_labels),
        "epochs_completed": len(history_values["loss"]), "best_epoch": best_epoch, "final_learning_rate": float(tf.keras.backend.get_value(model.optimizer.learning_rate)), "training_duration_seconds": duration_seconds,
        "metrics": metrics,
        "per_class": {label: {"precision": float(p), "recall": float(r), "f1": float(score), "support": int(count)} for label, p, r, score, count in zip(EXPECTED_CLASSES, precision, recall, f1, support)},
        "confusion_matrix": matrix.tolist(),
        "baseline_svm_comparison": {"svm_accuracy": 0.4791666666666667, "svm_macro_f1": 0.45973667988035805, "cnn_accuracy": metrics["accuracy"], "cnn_macro_f1": metrics["macro"]["f1"], "accuracy_delta": metrics["accuracy"] - 0.4791666666666667, "macro_f1_delta": metrics["macro"]["f1"] - 0.45973667988035805, "cnn_outperformed_svm": metrics["accuracy"] > 0.4791666666666667 and metrics["macro"]["f1"] > 0.45973667988035805},
        "model_relative_path": MODEL_PATH.relative_to(ROOT).as_posix(), "history_figure_relative_path": HISTORY_FIGURE_PATH.relative_to(ROOT).as_posix(), "confusion_figure_relative_path": CONFUSION_FIGURE_PATH.relative_to(ROOT).as_posix(),
        "reload_predictions_match": True, "reload_probability_max_abs_difference": max_reload_difference, "model_trained_on_test_data": False,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CNN baseline training completed in {duration_seconds:.2f} seconds.")


if __name__ == "__main__":
    main()
