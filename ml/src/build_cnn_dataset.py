"""Build deterministic, normalized Log-Mel arrays for CNN training and validation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from audio_features import extract_log_mel_spectrogram
from audio_preprocessing import preprocess_audio_file


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "ml" / "data" / "processed" / "cnn_baseline"
TEMP_OUTPUT_DIR = ROOT / "ml" / "data" / "processed" / "cnn_baseline.tmp"
SUMMARY_PATH = ROOT / "ml" / "metadata" / "cnn_dataset_summary.json"
MANIFEST_PATH = ROOT / "ml" / "metadata" / "ravdess_manifest.csv"
CONFIG_PATH = ROOT / "ml" / "config" / "cnn_dataset.json"
REQUIRED_FILES = (
    "train_log_mel.npy",
    "train_labels.npy",
    "validation_log_mel.npy",
    "validation_labels.npy",
    "normalization.npz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") != "1.0.0" or config.get("feature_shape") != [64, 219, 1]:
        raise ValueError("Invalid CNN dataset configuration.")
    if config.get("epsilon") != 1e-6 or len(config.get("class_order", [])) != 8:
        raise ValueError("Invalid CNN normalization or label configuration.")
    return config


def load_rows(split: str, expected_actors: set[str]) -> list[dict[str, str]]:
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if {row["actor_id"] for row in rows} != expected_actors:
        raise RuntimeError(f"Unexpected actor selection for {split} split.")
    return rows


def is_complete_directory(path: Path) -> bool:
    return path.is_dir() and {entry.name for entry in path.iterdir()} == set(REQUIRED_FILES)


def write_split(rows: list[dict[str, str]], destination: Path, class_ids: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    feature_path = destination / ("train_log_mel.npy" if rows[0]["split"] == "train" else "validation_log_mel.npy")
    label_path = destination / ("train_labels.npy" if rows[0]["split"] == "train" else "validation_labels.npy")
    features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32, shape=(len(rows), 64, 219, 1))
    labels = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        waveform, _ = preprocess_audio_file(ROOT / row["relative_path"])
        log_mel = extract_log_mel_spectrogram(waveform)
        if log_mel.shape != (64, 219) or not np.isfinite(log_mel).all():
            raise RuntimeError(f"Invalid Log-Mel output for {row['relative_path']}.")
        features[index, :, :, 0] = log_mel
        labels[index] = class_ids[row["emotion"]]
    features.flush()
    np.save(label_path, labels)
    return features, labels


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    for index in range(len(features)):
        features[index, :, :, 0] = (features[index, :, :, 0] - mean[:, None]) / std[:, None]
    features.flush()


def describe_file(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_summary(config: dict[str, Any], train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]]) -> dict[str, Any]:
    train_features = np.load(OUTPUT_DIR / "train_log_mel.npy", mmap_mode="r")
    validation_features = np.load(OUTPUT_DIR / "validation_log_mel.npy", mmap_mode="r")
    train_labels = np.load(OUTPUT_DIR / "train_labels.npy", mmap_mode="r")
    validation_labels = np.load(OUTPUT_DIR / "validation_labels.npy", mmap_mode="r")
    with np.load(OUTPUT_DIR / "normalization.npz") as normalization:
        mean = normalization["mean"]
        std = normalization["std"]
    classes = config["class_order"]
    return {
        "configuration_hash": sha256_file(CONFIG_PATH),
        "manifest_hash": sha256_file(MANIFEST_PATH),
        "output_directory": "ml/data/processed/cnn_baseline",
        "deterministic_manifest_order": True,
        "test_actors_accessed": False,
        "splits": {
            "train": {"actors": config["splits"]["train_actors"], "shape": list(train_features.shape), "dtype": str(train_features.dtype), "class_counts": {name: int(np.sum(train_labels == index)) for index, name in enumerate(classes)}},
            "validation": {"actors": config["splits"]["validation_actors"], "shape": list(validation_features.shape), "dtype": str(validation_features.dtype), "class_counts": {name: int(np.sum(validation_labels == index)) for index, name in enumerate(classes)}},
        },
        "labels": {"dtype": str(train_labels.dtype), "range": [int(train_labels.min()), int(train_labels.max())], "class_order": classes},
        "normalization": {
            "method": config["normalization"], "training_only": True, "epsilon": config["epsilon"],
            "mean_shape": list(mean.shape), "std_shape": list(std.shape), "dtype": str(mean.dtype),
            "normalized_train_mean_max_abs": float(np.max(np.abs(np.mean(train_features, axis=(0, 2, 3))))),
            "normalized_train_std_max_abs_error": float(np.max(np.abs(np.std(train_features, axis=(0, 2, 3)) - 1.0))),
        },
        "files": {name: describe_file(OUTPUT_DIR / name) for name in REQUIRED_FILES},
        "source_recordings": {"train": len(train_rows), "validation": len(validation_rows)},
        "model_trained": False,
        "raw_audio_modified": False,
    }


def main() -> None:
    config = load_config()
    if TEMP_OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to use stale temporary dataset output: {TEMP_OUTPUT_DIR}")
    if OUTPUT_DIR.exists():
        if not is_complete_directory(OUTPUT_DIR):
            raise RuntimeError(f"Existing dataset directory is incomplete: {OUTPUT_DIR}")
        print("CNN dataset already complete; reusing existing deterministic arrays.")
    else:
        train_rows = load_rows("train", set(config["splits"]["train_actors"]))
        validation_rows = load_rows("validation", set(config["splits"]["validation_actors"]))
        TEMP_OUTPUT_DIR.mkdir(parents=True)
        class_ids = {name: index for index, name in enumerate(config["class_order"])}
        try:
            train_features, _ = write_split(train_rows, TEMP_OUTPUT_DIR, class_ids)
            mean = np.asarray(np.mean(train_features, axis=(0, 2, 3), dtype=np.float64), dtype=np.float32)
            std = np.asarray(np.std(train_features, axis=(0, 2, 3), dtype=np.float64), dtype=np.float32)
            std = np.maximum(std, np.float32(config["epsilon"]))
            normalize_features(train_features, mean, std)
            del train_features
            validation_features, _ = write_split(validation_rows, TEMP_OUTPUT_DIR, class_ids)
            normalize_features(validation_features, mean, std)
            del validation_features
            np.savez(TEMP_OUTPUT_DIR / "normalization.npz", mean=mean, std=std, epsilon=np.float32(config["epsilon"]))
        except Exception:
            raise
        if not is_complete_directory(TEMP_OUTPUT_DIR):
            raise RuntimeError("Temporary CNN dataset output is incomplete.")
        TEMP_OUTPUT_DIR.rename(OUTPUT_DIR)
        print("CNN dataset arrays built atomically.")
    train_rows = load_rows("train", set(config["splits"]["train_actors"]))
    validation_rows = load_rows("validation", set(config["splits"]["validation_actors"]))
    SUMMARY_PATH.write_text(json.dumps(build_summary(config, train_rows, validation_rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CNN dataset summary written.")


if __name__ == "__main__":
    main()
