"""One-time held-out evaluation for the frozen reduced-regularization CNN."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(ROOT / "ml" / ".cache" / "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from audio_features import extract_log_mel_spectrogram, load_feature_config
from audio_preprocessing import load_preprocessing_config, preprocess_audio_file
from cnn_model import EXPECTED_CLASSES


MANIFEST_PATH = ROOT / "ml" / "metadata" / "ravdess_manifest.csv"
MODEL_PATH = ROOT / "ml" / "models" / "cnn" / "cnn_reduced_regularization.keras"
NORMALIZATION_PATH = ROOT / "ml" / "data" / "processed" / "cnn_baseline" / "normalization.npz"
VALIDATION_REPORT_PATH = ROOT / "ml" / "metadata" / "cnn_reduced_regularization_validation.json"
MODEL_CONFIG_PATH = ROOT / "ml" / "config" / "cnn_reduced_regularization.json"
DATASET_SUMMARY_PATH = ROOT / "ml" / "metadata" / "cnn_dataset_summary.json"
REPORT_PATH = ROOT / "ml" / "metadata" / "cnn_reduced_regularization_test.json"
FIGURE_PATH = ROOT / "ml" / "reports" / "figures" / "cnn_reduced_regularization_test_confusion_matrix.png"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_ignored(path: Path) -> None:
    result = subprocess.run(["git", "check-ignore", "-q", str(path.relative_to(ROOT))], cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Refusing to write a Git-tracked generated figure: {path}")


def load_test_rows() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    expected_actors = {"21", "22", "23", "24"}
    if len(rows) != 240 or {row["actor_id"] for row in rows} != expected_actors:
        raise RuntimeError("The final evaluation requires exactly the 240 records from actors 21-24.")
    if any(row["emotion"] not in EXPECTED_CLASSES for row in rows):
        raise RuntimeError("The test manifest contains an unexpected emotion label.")
    return rows


def load_training_normalization() -> tuple[np.ndarray, np.ndarray]:
    with np.load(NORMALIZATION_PATH) as data:
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)
        epsilon = float(data["epsilon"])
    if mean.shape != (64,) or std.shape != (64,) or not np.isclose(epsilon, 1e-6, rtol=0, atol=1e-12):
        raise RuntimeError("Training-only normalization statistics do not match the frozen CNN dataset contract.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("Training-only normalization statistics are invalid.")
    return mean, std


def verify_frozen_model() -> tf.keras.Model:
    validation = json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
    if validation["validation_metrics"]["accuracy"] != 0.5541666666666667 or validation["validation_metrics"]["macro"]["f1"] != 0.5369751467634547:
        raise RuntimeError("The selected-model validation provenance does not match the frozen selection record.")
    hashes = validation["configuration_hashes"]
    if hashes["variant_configuration_hash"] != sha256_file(MODEL_CONFIG_PATH) or hashes["dataset_summary_hash"] != sha256_file(DATASET_SUMMARY_PATH):
        raise RuntimeError("The selected model configuration or training-only dataset provenance has changed.")
    model = tf.keras.models.load_model(MODEL_PATH)
    if model.input_shape != (None, 64, 219, 1) or model.output_shape != (None, 8) or model.count_params() != 102344:
        raise RuntimeError("The selected model architecture does not match the frozen validation artifact.")
    if any(layer.rate != 0.0 for layer in model.layers if layer.__class__.__name__ == "Dropout"):
        raise RuntimeError("The selected model is not the frozen no-dropout CNN.")
    return model


def build_test_features(rows: list[dict[str, str]], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.empty((len(rows), 64, 219, 1), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    label_ids = {label: index for index, label in enumerate(EXPECTED_CLASSES)}
    preprocessing = load_preprocessing_config()
    feature_config = load_feature_config()
    for index, row in enumerate(rows):
        waveform, audit = preprocess_audio_file(ROOT / row["relative_path"], preprocessing)
        if waveform.shape != (56000,) or waveform.dtype != np.float32 or not np.isfinite(waveform).all():
            raise RuntimeError(f"Frozen preprocessing failed for {row['relative_path']}.")
        if audit["source_sample_rate"] != 48000 or audit["final_action"] not in {"padded", "truncated", "unchanged"}:
            raise RuntimeError(f"Unexpected source or final preprocessing audit for {row['relative_path']}.")
        log_mel = extract_log_mel_spectrogram(waveform, config=feature_config)
        if log_mel.shape != (64, 219) or log_mel.dtype != np.float32 or not np.isfinite(log_mel).all():
            raise RuntimeError(f"Frozen feature extraction failed for {row['relative_path']}.")
        features[index, :, :, 0] = (log_mel - mean[:, None]) / std[:, None]
        labels[index] = label_ids[row["emotion"]]
    if not np.isfinite(features).all() or features.shape != (240, 64, 219, 1):
        raise RuntimeError("Final test features are invalid.")
    return features, labels


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=range(8), zero_division=0)
    macro = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
        "per_emotion": {label: {"precision": float(p), "recall": float(r), "f1": float(score), "support": int(count)} for label, p, r, score, count in zip(EXPECTED_CLASSES, precision, recall, f1, support)},
    }


def write_confusion_matrix(matrix: np.ndarray) -> None:
    require_ignored(FIGURE_PATH)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Recording count")
    axis.set(xticks=range(8), yticks=range(8), xticklabels=EXPECTED_CLASSES, yticklabels=EXPECTED_CLASSES, xlabel="Predicted emotion", ylabel="Actual emotion", title="Final held-out test confusion matrix: reduced-regularization CNN")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(8):
        for column in range(8):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    if REPORT_PATH.exists() or FIGURE_PATH.exists():
        raise RuntimeError("Final test artifacts already exist; refusing to repeat held-out evaluation.")
    model = verify_frozen_model()
    rows = load_test_rows()
    mean, std = load_training_normalization()
    features, labels = build_test_features(rows, mean, std)
    probabilities = model.predict(features, batch_size=32, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    if probabilities.shape != (240, 8) or predictions.shape != (240,) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise RuntimeError("Final test predictions are invalid.")
    matrix = confusion_matrix(labels, predictions, labels=range(8))
    metrics = calculate_metrics(labels, predictions)
    if int(matrix.sum()) != 240 or sum(item["support"] for item in metrics["per_emotion"].values()) != 240:
        raise RuntimeError("Final test confusion matrix or supports do not reconcile.")
    reloaded = tf.keras.models.load_model(MODEL_PATH)
    reloaded_probabilities = reloaded.predict(features, batch_size=32, verbose=0)
    if not np.allclose(probabilities, reloaded_probabilities, rtol=0, atol=1e-7):
        raise RuntimeError("Reloaded frozen model does not reproduce final test probabilities.")
    write_confusion_matrix(matrix)
    report = {
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "tensorflow": tf.__version__},
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "feature_identity": {"configuration_hash": sha256_file(ROOT / "ml" / "config" / "features.json"), "shape": [64, 219, 1]},
        "final_test_only": True,
        "model": {"artifact_relative_path": MODEL_PATH.relative_to(ROOT).as_posix(), "artifact_sha256": sha256_file(MODEL_PATH), "identity": "cnn_reduced_regularization_no_dropout", "parameter_count": 102344, "selected_validation_accuracy": 0.5541666666666667, "selected_validation_macro_f1": 0.5369751467634547},
        "normalization": {"artifact_relative_path": NORMALIZATION_PATH.relative_to(ROOT).as_posix(), "artifact_sha256": sha256_file(NORMALIZATION_PATH), "method": "training_only_per_mel_band_mean_std", "statistics_shape": [64]},
        "preprocessing_identity": {"configuration_hash": sha256_file(ROOT / "ml" / "config" / "preprocessing.json"), "output_shape": [56000]},
        "prediction_distribution": {label: int(np.sum(predictions == index)) for index, label in enumerate(EXPECTED_CLASSES)},
        "reload": {"probabilities_match": True, "probability_max_abs_difference": float(np.max(np.abs(probabilities - reloaded_probabilities)))},
        "selected_before_test": True,
        "test": {"actor_ids": ["21", "22", "23", "24"], "feature_shape": list(features.shape), "metrics": metrics, "recording_count": len(rows), "true_distribution": {label: int(np.sum(labels == index)) for index, label in enumerate(EXPECTED_CLASSES)}},
        "test_confusion_matrix": matrix.tolist(),
        "test_figure_relative_path": FIGURE_PATH.relative_to(ROOT).as_posix(),
        "training_or_tuning_after_test": False,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Final held-out test evaluation completed exactly once.")


if __name__ == "__main__":
    main()
