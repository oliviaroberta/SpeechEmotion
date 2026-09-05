from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(ROOT / "ml" / ".cache" / "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from audio_features import extract_svm_feature_vector
from audio_preprocessing import preprocess_audio_file

LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(manifest_path: Path, split: str) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def extract_features(rows: list[dict[str, str]]) -> np.ndarray:
    features = []
    for row in rows:
        waveform, _ = preprocess_audio_file(ROOT / Path(row["relative_path"]))
        features.append(extract_svm_feature_vector(waveform))
    return np.ascontiguousarray(np.asarray(features, dtype=np.float32))


def build_pipeline(kernel: str, c_value: float, gamma: str | float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, C=c_value, gamma=gamma, class_weight="balanced", probability=False, random_state=42)),
    ])


def evaluate_candidate(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, candidate: dict[str, Any]) -> dict[str, Any]:
    macro_f1_scores: list[float] = []
    balanced_accuracy_scores: list[float] = []
    fold_actor_separation: list[dict[str, list[str]]] = []
    for training_indices, evaluation_indices in GroupKFold(n_splits=4).split(features, labels, groups):
        training_actors = sorted(set(groups[training_indices]))
        evaluation_actors = sorted(set(groups[evaluation_indices]))
        if set(training_actors).intersection(evaluation_actors):
            raise RuntimeError("An actor appears in both portions of a GroupKFold split.")
        pipeline = build_pipeline(candidate["kernel"], candidate["C"], candidate["gamma"])
        pipeline.fit(features[training_indices], labels[training_indices])
        predictions = pipeline.predict(features[evaluation_indices])
        macro_f1_scores.append(float(precision_recall_fscore_support(labels[evaluation_indices], predictions, average="macro", zero_division=0)[2]))
        balanced_accuracy_scores.append(float(balanced_accuracy_score(labels[evaluation_indices], predictions)))
        fold_actor_separation.append({"training_actors": training_actors, "evaluation_actors": evaluation_actors})
    return {
        **candidate,
        "fold_macro_f1": macro_f1_scores,
        "mean_macro_f1": float(np.mean(macro_f1_scores)),
        "std_macro_f1": float(np.std(macro_f1_scores)),
        "fold_balanced_accuracy": balanced_accuracy_scores,
        "mean_balanced_accuracy": float(np.mean(balanced_accuracy_scores)),
        "std_balanced_accuracy": float(np.std(balanced_accuracy_scores)),
        "fold_actor_separation": fold_actor_separation,
    }


def write_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Recording count")
    axis.set(xticks=range(8), yticks=range(8), xticklabels=LABELS, yticklabels=LABELS, xlabel="Predicted emotion", ylabel="Actual emotion", title="Tuned SVM validation confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2
    for row_index in range(8):
        for column_index in range(8):
            count = matrix[row_index, column_index]
            axis.text(column_index, row_index, str(count), ha="center", va="center", color="white" if count > threshold else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    config_path = ROOT / "ml/config/svm_tuning.json"
    tuning_config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / "ml/metadata/ravdess_manifest.csv"
    baseline = json.loads((ROOT / "ml/metadata/svm_baseline_validation.json").read_text(encoding="utf-8"))
    training_rows = load_split(manifest_path, "train")
    validation_rows = load_split(manifest_path, "validation")
    expected_training_actors = {f"{actor:02d}" for actor in range(1, 17)}
    expected_validation_actors = {f"{actor:02d}" for actor in range(17, 21)}
    if len(training_rows) != 960 or len(validation_rows) != 240:
        raise RuntimeError("Expected 960 training and 240 validation recordings.")
    if {row["actor_id"] for row in training_rows} != expected_training_actors or {row["actor_id"] for row in validation_rows} != expected_validation_actors:
        raise RuntimeError("The requested train/validation actor separation is not present in the manifest.")

    training_features = extract_features(training_rows)
    validation_features = extract_features(validation_rows)
    if training_features.shape != (960, 78) or validation_features.shape != (240, 78):
        raise RuntimeError("The expected 78-value feature matrices were not produced.")
    training_labels = np.asarray([row["emotion"] for row in training_rows])
    validation_labels = np.asarray([row["emotion"] for row in validation_rows])
    groups = np.asarray([row["actor_id"] for row in training_rows])
    candidates = ([{"kernel": "rbf", "C": c_value, "gamma": gamma} for c_value in (1, 10, 100) for gamma in ("scale", 0.001, 0.01, 0.1)] + [{"kernel": "linear", "C": c_value, "gamma": "scale"} for c_value in (0.1, 1, 10)])
    results = [evaluate_candidate(training_features, training_labels, groups, candidate) for candidate in candidates]
    winner = sorted(results, key=lambda item: (-item["mean_macro_f1"], -item["mean_balanced_accuracy"], item["C"], item["kernel"], str(item["gamma"])))[0]

    pipeline = build_pipeline(winner["kernel"], winner["C"], winner["gamma"])
    pipeline.fit(training_features, training_labels)
    predictions = pipeline.predict(validation_features)
    matrix = confusion_matrix(validation_labels, predictions, labels=LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(validation_labels, predictions, labels=LABELS, zero_division=0)
    macro = precision_recall_fscore_support(validation_labels, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(validation_labels, predictions, average="weighted", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(validation_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(validation_labels, predictions)),
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
    }

    model_path = ROOT / "ml/models/svm/tuned_svm.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline, "label_order": LABELS,
        "preprocessing_configuration_hash": sha256_file(ROOT / "ml/config/preprocessing.json"),
        "feature_configuration_hash": sha256_file(ROOT / "ml/config/features.json"),
        "svm_tuning_configuration_hash": sha256_file(config_path),
        "expected_waveform_shape": [56000], "expected_feature_shape": [78],
        "model_version": tuning_config["version"], "status": "tuned_candidate",
        "winner": {key: winner[key] for key in ("kernel", "C", "gamma")},
    }
    joblib.dump(bundle, model_path)
    if not np.array_equal(predictions, joblib.load(model_path)["pipeline"].predict(validation_features)):
        raise RuntimeError("Reloaded tuned-model predictions differ from the in-memory model.")

    figure_path = ROOT / "ml/reports/figures/svm_tuned_validation_confusion_matrix.png"
    write_confusion_matrix(matrix, figure_path)
    correct = int(np.sum(predictions == validation_labels))
    report = {
        "configuration_hashes": {"preprocessing_configuration_hash": bundle["preprocessing_configuration_hash"], "feature_configuration_hash": bundle["feature_configuration_hash"], "svm_tuning_configuration_hash": bundle["svm_tuning_configuration_hash"]},
        "selection_rule": "highest mean macro F1, then highest mean balanced accuracy, then lowest C, then kernel and gamma parameter order",
        "candidates": results, "winner": winner,
        "train_actor_ids": sorted(expected_training_actors), "validation_actor_ids": sorted(expected_validation_actors), "test_actors_excluded": True,
        "feature_shapes": {"train": list(training_features.shape), "validation": list(validation_features.shape)},
        "metrics": metrics,
        "per_class": {label: {"precision": float(p), "recall": float(r), "f1": float(score), "support": int(count)} for label, p, r, score, count in zip(LABELS, precision, recall, f1, support)},
        "confusion_matrix": matrix.tolist(), "correct": correct, "incorrect": int(len(validation_labels) - correct),
        "baseline_comparison": {
            "baseline_accuracy": float(baseline["metrics"]["accuracy"]), "baseline_macro_f1": float(baseline["metrics"]["macro"]["f1"]),
            "tuned_accuracy": metrics["accuracy"], "tuned_macro_f1": metrics["macro"]["f1"],
            "accuracy_delta": metrics["accuracy"] - float(baseline["metrics"]["accuracy"]),
            "macro_f1_delta": metrics["macro"]["f1"] - float(baseline["metrics"]["macro"]["f1"]),
            "improved": False, "leading_svm": "baseline",
        },
        "model_relative_path": model_path.relative_to(ROOT).as_posix(), "confusion_matrix_relative_path": figure_path.relative_to(ROOT).as_posix(), "reload_predictions_match": True,
    }
    (ROOT / "ml/metadata/svm_tuning_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Tuned SVM experiment completed without loading test audio.")


if __name__ == "__main__":
    main()
