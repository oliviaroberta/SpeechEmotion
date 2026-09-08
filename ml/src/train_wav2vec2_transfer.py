"""Run the single frozen-Wav2Vec2 transfer-learning validation experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(ROOT / "ml" / ".cache" / "matplotlib")
os.environ["HF_HOME"] = str(ROOT / "ml" / ".cache" / "huggingface")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from audio_preprocessing import load_preprocessing_config, preprocess_audio_file
from wav2vec2_transfer import EXPECTED_LABELS, load_transfer_config, pool_hidden_states, validate_waveform


CONFIG_PATH = ROOT / "ml" / "config" / "wav2vec2_transfer.json"
MANIFEST_PATH = ROOT / "ml" / "metadata" / "ravdess_manifest.csv"
CACHE_DIR = ROOT / "ml" / "data" / "processed" / "transfer_wav2vec2"
MODEL_PATH = ROOT / "ml" / "models" / "transfer" / "wav2vec2_logreg.joblib"
FIGURE_PATH = ROOT / "ml" / "reports" / "figures" / "wav2vec2_transfer_validation_confusion_matrix.png"
REPORT_PATH = ROOT / "ml" / "metadata" / "wav2vec2_transfer_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_ignored(path: Path) -> None:
    result = subprocess.run(["git", "check-ignore", "-q", str(path.relative_to(ROOT))], cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Refusing to write an artifact that is not ignored by Git: {path}")


def load_records(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Read only the train/validation manifest records needed by this experiment."""
    allowed = {"train": set(config["splits"]["train_actors"]), "validation": set(config["splits"]["validation_actors"])}
    records = {"train": [], "validation": []}
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in records:
                continue
            actor_id = row["actor_id"]
            if actor_id not in allowed[split]:
                raise RuntimeError(f"Unexpected actor {actor_id} in {split} manifest rows.")
            records[split].append(row)
    if len(records["train"]) != 960 or len(records["validation"]) != 240:
        raise RuntimeError("The manifest must provide exactly 960 train and 240 validation recordings.")
    for split, expected in allowed.items():
        if {row["actor_id"] for row in records[split]} != expected:
            raise RuntimeError(f"The {split} actor split is incomplete or inconsistent.")
    return records


def cache_paths(split: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{split}_embeddings.npy", CACHE_DIR / f"{split}_labels.npy"


def write_array_atomically(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise RuntimeError(f"A stale partial cache exists and will not be overwritten: {temporary}")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def validate_cached_arrays(split: str, embeddings: np.ndarray, labels: np.ndarray) -> None:
    expected_rows = 960 if split == "train" else 240
    if embeddings.shape != (expected_rows, 1536) or embeddings.dtype != np.float32 or not embeddings.flags.c_contiguous:
        raise RuntimeError(f"Invalid cached {split} embedding array.")
    if labels.shape != (expected_rows,) or not np.issubdtype(labels.dtype, np.integer):
        raise RuntimeError(f"Invalid cached {split} label array.")
    if not np.isfinite(embeddings).all() or not np.isin(labels, np.arange(8)).all():
        raise RuntimeError(f"Invalid cached {split} embedding values or labels.")


def load_or_extract_embeddings(records: dict[str, list[dict[str, str]]], config: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float, str]:
    paths = [path for split in ("train", "validation") for path in cache_paths(split)]
    existing = [path.exists() for path in paths]
    if any(existing) and not all(existing):
        raise RuntimeError("Partial embedding cache detected; refusing to treat it as a completed dataset.")
    if all(existing):
        embeddings = {split: np.load(cache_paths(split)[0]) for split in ("train", "validation")}
        labels = {split: np.load(cache_paths(split)[1]) for split in ("train", "validation")}
        for split in ("train", "validation"):
            validate_cached_arrays(split, embeddings[split], labels[split])
        revision_path = ROOT / "ml" / ".cache" / "huggingface" / "models--facebook--wav2vec2-base" / "refs" / "main"
        revision = revision_path.read_text(encoding="utf-8").strip() if revision_path.exists() else "unknown"
        return embeddings, labels, 0.0, revision

    for path in paths:
        require_ignored(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["model_identifier"], cache_dir=ROOT / "ml" / ".cache" / "huggingface")
    model = Wav2Vec2Model.from_pretrained(config["model_identifier"], cache_dir=ROOT / "ml" / ".cache" / "huggingface")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    embeddings: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    label_to_id = {label: index for index, label in enumerate(EXPECTED_LABELS)}
    preprocessing = load_preprocessing_config()
    with torch.inference_mode():
        for split in ("train", "validation"):
            rows = records[split]
            output = np.empty((len(rows), 1536), dtype=np.float32)
            output_labels = np.empty(len(rows), dtype=np.int64)
            for start in range(0, len(rows), int(config["encoder_batch_size"])):
                batch_rows = rows[start:start + int(config["encoder_batch_size"])]
                waveforms = []
                for row in batch_rows:
                    waveform, _ = preprocess_audio_file(ROOT / row["relative_path"], preprocessing)
                    waveforms.append(validate_waveform(waveform, config))
                inputs = extractor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
                hidden = model(inputs.input_values).last_hidden_state.detach().cpu().numpy()
                for offset, (row, states) in enumerate(zip(batch_rows, hidden)):
                    output[start + offset] = pool_hidden_states(states, config)
                    output_labels[start + offset] = label_to_id[row["emotion"]]
            output = np.ascontiguousarray(output, dtype=np.float32)
            validate_cached_arrays(split, output, output_labels)
            embeddings[split], labels[split] = output, output_labels
    elapsed = time.perf_counter() - started
    for split in ("train", "validation"):
        write_array_atomically(cache_paths(split)[0], embeddings[split])
        write_array_atomically(cache_paths(split)[1], labels[split])
    return embeddings, labels, elapsed, str(model.config._commit_hash or "unknown")


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=range(8), zero_division=0)
    macro = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
        "per_emotion": {label: {"precision": float(p), "recall": float(r), "f1": float(score), "support": int(count)} for label, p, r, score, count in zip(EXPECTED_LABELS, precision, recall, f1, support)},
    }


def write_confusion_matrix(matrix: np.ndarray) -> None:
    require_ignored(FIGURE_PATH)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Recording count")
    axis.set(xticks=range(8), yticks=range(8), xticklabels=EXPECTED_LABELS, yticklabels=EXPECTED_LABELS, xlabel="Predicted emotion", ylabel="Actual emotion", title="Frozen Wav2Vec2 transfer validation confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(8):
        for column in range(8):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    config = load_transfer_config()
    records = load_records(config)
    require_ignored(ROOT / "ml" / ".cache" / "huggingface")
    embeddings, labels, extraction_seconds, revision = load_or_extract_embeddings(records, config)
    if MODEL_PATH.exists() or REPORT_PATH.exists() or FIGURE_PATH.exists():
        raise RuntimeError("Refusing to overwrite an existing transfer model, report, or figure.")
    require_ignored(MODEL_PATH)
    classifier_config = config["classifier"]
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(C=classifier_config["C"], class_weight=classifier_config["class_weight"], max_iter=classifier_config["max_iter"], random_state=classifier_config["random_state"], solver=classifier_config["solver"]))])
    pipeline.fit(embeddings["train"], labels["train"])
    predictions = pipeline.predict(embeddings["validation"])
    probabilities = pipeline.predict_proba(embeddings["validation"])
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise RuntimeError("Invalid validation probabilities from the transfer classifier.")
    matrix = confusion_matrix(labels["validation"], predictions, labels=range(8))
    metrics = calculate_metrics(labels["validation"], predictions)
    bundle = {"pipeline": pipeline, "label_order": list(EXPECTED_LABELS), "model_identifier": config["model_identifier"], "encoder_frozen": True, "embedding_shape": [1536], "preprocessing_configuration_hash": sha256_file(ROOT / "ml" / "config" / "preprocessing.json"), "transfer_configuration_hash": sha256_file(CONFIG_PATH), "status": "validation_only_transfer_candidate", "version": config["version"]}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    restored = joblib.load(MODEL_PATH)
    restored_probabilities = restored["pipeline"].predict_proba(embeddings["validation"])
    if not np.array_equal(predictions, restored["pipeline"].predict(embeddings["validation"])) or not np.allclose(probabilities, restored_probabilities, rtol=0, atol=1e-12):
        raise RuntimeError("Reloaded transfer classifier predictions or probabilities differ.")
    write_confusion_matrix(matrix)
    threshold = config["promotion_rule"]
    promoted = metrics["accuracy"] >= threshold["minimum_validation_accuracy"] and metrics["macro"]["f1"] >= threshold["minimum_validation_macro_f1"]
    report = {"actor_ids": {"train": config["splits"]["train_actors"], "validation": config["splits"]["validation_actors"], "test_actors_21_to_24_accessed": False}, "cache": {path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for split in ("train", "validation") for path in cache_paths(split)}, "configuration_hashes": {"preprocessing": sha256_file(ROOT / "ml" / "config" / "preprocessing.json"), "transfer": sha256_file(CONFIG_PATH)}, "embedding_extraction_seconds": extraction_seconds, "embedding_shapes": {"train": list(embeddings["train"].shape), "validation": list(embeddings["validation"].shape)}, "encoder": {"frozen": True, "model_identifier": config["model_identifier"], "resolved_revision": revision}, "label_order": list(EXPECTED_LABELS), "metrics": metrics, "model_relative_path": MODEL_PATH.relative_to(ROOT).as_posix(), "package_versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__, "transformers": transformers.__version__}, "prediction_distribution": {label: int(np.sum(predictions == index)) for index, label in enumerate(EXPECTED_LABELS)}, "promotion": {"current_leading_model": "cnn_reduced_regularization", "current_leader_validation_accuracy": 0.5541666666666667, "current_leader_validation_macro_f1": 0.5369751467634547, "rule": threshold, "transfer_candidate_promoted": promoted}, "reload": {"predictions_match": True, "probabilities_match": True, "probability_max_abs_difference": float(np.max(np.abs(probabilities - restored_probabilities)))}, "source_recording_counts": {"train": 960, "validation": 240}, "validation_confusion_matrix": matrix.tolist(), "validation_figure_relative_path": FIGURE_PATH.relative_to(ROOT).as_posix()}
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Frozen Wav2Vec2 transfer validation completed; promoted={promoted}.")


if __name__ == "__main__":
    main()
