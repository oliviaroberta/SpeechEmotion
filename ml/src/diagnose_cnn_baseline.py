"""Diagnose the saved CNN baseline without altering it or accessing test actors."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from cnn_model import EXPECTED_CLASSES, build_cnn_baseline


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data" / "processed" / "cnn_baseline"
MANIFEST_PATH = ROOT / "ml" / "metadata" / "ravdess_manifest.csv"
BASELINE_REPORT_PATH = ROOT / "ml" / "metadata" / "cnn_baseline_validation.json"
MODEL_PATH = ROOT / "ml" / "models" / "cnn" / "cnn_baseline.keras"
REPORT_PATH = ROOT / "ml" / "metadata" / "cnn_baseline_diagnosis.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest_split(split: str) -> list[dict[str, str]]:
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    expected = 960 if split == "train" else 240
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} manifest rows for {split}.")
    return rows


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = (
        np.load(DATA_DIR / "train_log_mel.npy"),
        np.load(DATA_DIR / "train_labels.npy"),
        np.load(DATA_DIR / "validation_log_mel.npy"),
        np.load(DATA_DIR / "validation_labels.npy"),
    )
    if arrays[0].shape != (960, 64, 219, 1) or arrays[2].shape != (240, 64, 219, 1):
        raise RuntimeError("CNN dataset feature shapes are invalid.")
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("CNN dataset arrays contain non-finite values.")
    return arrays


def assert_alignment(rows: list[dict[str, str]], labels: np.ndarray, split: str) -> list[dict[str, Any]]:
    class_ids = {emotion: index for index, emotion in enumerate(EXPECTED_CLASSES)}
    expected = np.asarray([class_ids[row["emotion"]] for row in rows])
    if not np.array_equal(labels, expected):
        raise RuntimeError(f"{split} labels do not match manifest order.")
    samples = []
    for emotion in EXPECTED_CLASSES:
        index = next(index for index, row in enumerate(rows) if row["emotion"] == emotion)
        samples.append({
            "index": index,
            "relative_path": rows[index]["relative_path"],
            "actor_id": rows[index]["actor_id"],
            "emotion": emotion,
            "label": int(labels[index]),
        })
    return samples


def evaluate(model: tf.keras.Model, features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    results = model.evaluate(features, labels, batch_size=32, verbose=0, return_dict=True)
    probabilities = model.predict(features, batch_size=32, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    return {
        "loss": float(results["loss"]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "prediction_counts": {emotion: int(np.sum(predictions == index)) for index, emotion in enumerate(EXPECTED_CLASSES)},
        "mean_prediction_confidence": float(np.mean(np.max(probabilities, axis=1))),
        "predictions": predictions,
    }


class StopOnMemorisation(tf.keras.callbacks.Callback):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        super().__init__()
        self.features = features
        self.labels = labels
        self.accuracy_history: list[float] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        predictions = np.argmax(self.model.predict(self.features, verbose=0), axis=1)
        accuracy = float(np.mean(predictions == self.labels))
        self.accuracy_history.append(accuracy)
        if accuracy >= 0.95:
            self.model.stop_training = True


def run_memorisation_test(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    indices = np.concatenate([np.flatnonzero(labels == class_id)[:4] for class_id in range(8)])
    if len(indices) != 32:
        raise RuntimeError("Memorisation subset is not balanced with 32 examples.")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = build_cnn_baseline()
    callback = StopOnMemorisation(features[indices], labels[indices])
    started = time.perf_counter()
    history = model.fit(features[indices], labels[indices], batch_size=32, epochs=200, verbose=0, callbacks=[callback])
    probabilities = model.predict(features[indices], verbose=0)
    accuracy = float(np.mean(np.argmax(probabilities, axis=1) == labels[indices]))
    return {
        "subset_size": 32,
        "examples_per_emotion": 4,
        "maximum_epochs": 200,
        "epochs_completed": len(history.history["loss"]),
        "accuracy": accuracy,
        "reached_95_percent": accuracy >= 0.95,
        "duration_seconds": time.perf_counter() - started,
        "accuracy_history": callback.accuracy_history,
        "model_saved": False,
        "validation_or_test_data_used": False,
    }


def main() -> None:
    if not MODEL_PATH.is_file():
        raise RuntimeError("Saved CNN baseline model is missing.")
    baseline_report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    train_rows = load_manifest_split("train")
    validation_rows = load_manifest_split("validation")
    train_features, train_labels, validation_features, validation_labels = load_arrays()
    train_samples = assert_alignment(train_rows, train_labels, "train")
    validation_samples = assert_alignment(validation_rows, validation_labels, "validation")
    saved_model_hash_before = sha256_file(MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    training = evaluate(model, train_features, train_labels)
    validation = evaluate(model, validation_features, validation_labels)
    per_actor = {}
    for actor_id in ("17", "18", "19", "20"):
        actor_indices = np.asarray([index for index, row in enumerate(validation_rows) if row["actor_id"] == actor_id])
        actor_predictions = validation["predictions"][actor_indices]
        actor_labels = validation_labels[actor_indices]
        per_actor[actor_id] = {
            "recordings": int(len(actor_indices)),
            "accuracy": float(accuracy_score(actor_labels, actor_predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(actor_labels, actor_predictions)),
            "macro_f1": float(f1_score(actor_labels, actor_predictions, average="macro", zero_division=0)),
        }
    memorisation = run_memorisation_test(train_features, train_labels)
    saved_model_hash_after = sha256_file(MODEL_PATH)
    if saved_model_hash_before != saved_model_hash_after:
        raise RuntimeError("The saved CNN baseline model was modified during diagnosis.")
    if memorisation["reached_95_percent"] and training["accuracy"] < 0.75:
        diagnosis = "Underfitting/training-policy problem"
        rationale = "The temporary model memorised the balanced subset, while the saved baseline has low training accuracy."
    elif not memorisation["reached_95_percent"]:
        diagnosis = "Pipeline or learning-capacity problem"
        rationale = "The fixed architecture did not memorise the balanced 32-example subset within 200 epochs."
    elif training["accuracy"] - validation["accuracy"] >= 0.15:
        diagnosis = "Overfitting/speaker-generalisation problem"
        rationale = "The model memorised the tiny subset and has a material training-versus-validation accuracy gap."
    else:
        diagnosis = "Inconclusive"
        rationale = "The saved-model and tiny-subset measurements do not isolate a single likely cause."
    report = {
        "baseline_model_sha256": saved_model_hash_before,
        "baseline_model_unchanged": True,
        "fixed_emotion_order": list(EXPECTED_CLASSES),
        "labels_match_fixed_emotion_order": True,
        "saved_model_metrics": {"train": {key: value for key, value in training.items() if key != "predictions"}, "validation": {key: value for key, value in validation.items() if key != "predictions"}},
        "validation_per_actor": per_actor,
        "configured_class_weights": baseline_report["class_weights"],
        "recorded_epoch_history": {"available": False, "reason": "Not recorded in the committed baseline validation report.", "epochs_completed": baseline_report["epochs_completed"], "best_epoch": baseline_report["best_epoch"], "final_learning_rate": baseline_report["final_learning_rate"]},
        "deterministic_alignment_samples": {"train": train_samples, "validation": validation_samples},
        "memorisation_test": memorisation,
        "diagnosis": diagnosis,
        "diagnosis_rationale": rationale,
        "test_actors_21_to_24_accessed": False,
        "temporary_model_saved": False,
        "baseline_model_retrained": False,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CNN baseline diagnosis completed: {diagnosis}")


if __name__ == "__main__":
    main()
