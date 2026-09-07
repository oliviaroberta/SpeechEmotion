"""Train the fixed no-dropout CNN variant on prepared train and validation arrays."""

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from cnn_model import EXPECTED_CLASSES, build_cnn_reduced_regularization


DATA_DIR = ROOT / "ml" / "data" / "processed" / "cnn_baseline"
CONFIG_PATH = ROOT / "ml" / "config" / "cnn_reduced_regularization.json"
MODEL_PATH = ROOT / "ml" / "models" / "cnn" / "cnn_reduced_regularization.keras"
REPORT_PATH = ROOT / "ml" / "metadata" / "cnn_reduced_regularization_validation.json"
HISTORY_PATH = ROOT / "ml" / "reports" / "figures" / "cnn_reduced_regularization_history.png"
CONFUSION_PATH = ROOT / "ml" / "reports" / "figures" / "cnn_reduced_regularization_validation_confusion_matrix.png"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version") != "1.0.0" or config.get("dropout_rates") != [0.0, 0.0, 0.0, 0.0] or config.get("maximum_epochs") != 80:
        raise ValueError("CNN reduced-regularization configuration is invalid.")
    return config


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = (np.load(DATA_DIR / "train_log_mel.npy"), np.load(DATA_DIR / "train_labels.npy"), np.load(DATA_DIR / "validation_log_mel.npy"), np.load(DATA_DIR / "validation_labels.npy"))
    if arrays[0].shape != (960, 64, 219, 1) or arrays[2].shape != (240, 64, 219, 1) or not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("Prepared CNN arrays are invalid.")
    return arrays


def class_weights(labels: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(classes, np.arange(8)):
        raise RuntimeError("Training labels must contain every emotion.")
    return {int(class_id): float(len(labels) / (len(classes) * count)) for class_id, count in zip(classes, counts)}


class ValidationMacroF1(tf.keras.callbacks.Callback):
    """Compute validation macro F1 before checkpoint, stopping, and LR callbacks."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        super().__init__()
        self.features = features
        self.labels = labels
        self.records: list[dict[str, float]] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs if logs is not None else {}
        probabilities = self.model.predict(self.features, batch_size=32, verbose=0)
        macro_f1 = float(f1_score(self.labels, np.argmax(probabilities, axis=1), average="macro", zero_division=0))
        logs["val_macro_f1"] = macro_f1
        logs["learning_rate"] = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.records.append({key: float(logs[key]) for key in ("loss", "accuracy", "val_loss", "val_accuracy", "val_macro_f1", "learning_rate")})


def write_history(records: list[dict[str, float]]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(records) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [record["loss"] for record in records], label="Training")
    axes[0].plot(epochs, [record["val_loss"] for record in records], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss"); axes[0].legend()
    axes[1].plot(epochs, [record["accuracy"] for record in records], label="Training")
    axes[1].plot(epochs, [record["val_accuracy"] for record in records], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy"); axes[1].legend()
    axes[2].plot(epochs, [record["val_macro_f1"] for record in records], label="Validation macro F1")
    axes[2].set(title="Validation macro F1", xlabel="Epoch", ylabel="Macro F1"); axes[2].legend()
    figure.tight_layout(); figure.savefig(HISTORY_PATH, dpi=150); plt.close(figure)


def write_confusion(matrix: np.ndarray) -> None:
    CONFUSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8)); image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Recording count")
    axis.set(xticks=range(8), yticks=range(8), xticklabels=EXPECTED_CLASSES, yticklabels=EXPECTED_CLASSES, xlabel="Predicted emotion", ylabel="Actual emotion", title="Reduced-regularization CNN validation confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(8):
        for column in range(8): axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    figure.tight_layout(); figure.savefig(CONFUSION_PATH, dpi=150); plt.close(figure)


def metrics_for(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=range(8), zero_division=0)
    macro = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)), "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)), "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])}, "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])}, "per_class": {emotion: {"precision": float(p), "recall": float(r), "f1": float(score), "support": int(count)} for emotion, p, r, score, count in zip(EXPECTED_CLASSES, precision, recall, f1, support)}}


def main() -> None:
    config = load_config()
    if MODEL_PATH.exists(): raise RuntimeError(f"Refusing to overwrite existing model: {MODEL_PATH}")
    train_features, train_labels, validation_features, validation_labels = load_arrays()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(42)
    model = build_cnn_reduced_regularization()
    macro_callback = ValidationMacroF1(validation_features, validation_labels)
    callbacks = [macro_callback, tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_macro_f1", mode="max", save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=12, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", patience=4, factor=0.5, min_lr=1e-5)]
    started = time.perf_counter()
    model.fit(train_features, train_labels, validation_data=(validation_features, validation_labels), batch_size=config["batch_size"], epochs=config["maximum_epochs"], class_weight=class_weights(train_labels), callbacks=callbacks, verbose=2)
    duration = time.perf_counter() - started
    probabilities = model.predict(validation_features, batch_size=32, verbose=0); predictions = np.argmax(probabilities, axis=1)
    reloaded = tf.keras.models.load_model(MODEL_PATH); reloaded_probabilities = reloaded.predict(validation_features, batch_size=32, verbose=0)
    difference = float(np.max(np.abs(probabilities - reloaded_probabilities)))
    if not np.allclose(probabilities, reloaded_probabilities, rtol=0, atol=1e-7): raise RuntimeError("Reloaded model differs from restored best model.")
    train_predictions = np.argmax(model.predict(train_features, batch_size=32, verbose=0), axis=1)
    validation_metrics = metrics_for(validation_labels, predictions); train_metrics = metrics_for(train_labels, train_predictions)
    matrix = confusion_matrix(validation_labels, predictions, labels=range(8)); best_epoch = int(np.argmax([record["val_macro_f1"] for record in macro_callback.records]) + 1)
    assessment = "overfitting" if train_metrics["macro"]["f1"] - validation_metrics["macro"]["f1"] >= 0.15 else "underfitting" if train_metrics["accuracy"] < 0.75 else "inconclusive"
    write_history(macro_callback.records); write_confusion(matrix)
    report = {"configuration_hashes": {"variant_configuration_hash": sha256_file(CONFIG_PATH), "baseline_architecture_configuration_hash": sha256_file(ROOT / "ml/config/cnn_baseline.json"), "dataset_summary_hash": sha256_file(ROOT / "ml/metadata/cnn_dataset_summary.json")}, "actor_ids": {"train": [f"{actor:02d}" for actor in range(1, 17)], "validation": [f"{actor:02d}" for actor in range(17, 21)], "test_actors_21_to_24_accessed": False}, "parameter_count": int(model.count_params()), "feature_shapes": {"train": list(train_features.shape), "validation": list(validation_features.shape)}, "class_weights": class_weights(train_labels), "epochs_completed": len(macro_callback.records), "best_epoch": best_epoch, "training_duration_seconds": duration, "final_learning_rate": float(tf.keras.backend.get_value(model.optimizer.learning_rate)), "epoch_history": macro_callback.records, "training_metrics": train_metrics, "validation_metrics": validation_metrics, "confusion_matrix": matrix.tolist(), "generalization_assessment": assessment, "comparisons": {"svm": {"accuracy": 0.4791666666666667, "macro_f1": 0.45973667988035805}, "original_cnn": {"accuracy": 0.20416666666666666, "macro_f1": 0.1131718124642653}, "reduced_regularization_cnn": {"accuracy": validation_metrics["accuracy"], "macro_f1": validation_metrics["macro"]["f1"]}}, "model_relative_path": MODEL_PATH.relative_to(ROOT).as_posix(), "history_figure_relative_path": HISTORY_PATH.relative_to(ROOT).as_posix(), "confusion_figure_relative_path": CONFUSION_PATH.relative_to(ROOT).as_posix(), "reload_probabilities_match": True, "reload_probability_max_abs_difference": difference}
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Reduced-regularization CNN training completed in {duration:.2f} seconds.")


if __name__ == "__main__": main()
