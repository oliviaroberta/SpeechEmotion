"""Verify the deterministic CNN dataset arrays without opening raw audio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_cnn_dataset import CONFIG_PATH, OUTPUT_DIR, REQUIRED_FILES, SUMMARY_PATH, load_config, load_rows, sha256_file


def main() -> None:
    config = load_config()
    if not SUMMARY_PATH.is_file() or not OUTPUT_DIR.is_dir() or {entry.name for entry in OUTPUT_DIR.iterdir()} != set(REQUIRED_FILES):
        raise RuntimeError("CNN dataset outputs or summary are incomplete.")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["configuration_hash"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("CNN dataset configuration hash does not match the summary.")
    for name in REQUIRED_FILES:
        path = OUTPUT_DIR / name
        if summary["files"][name] != {"bytes": path.stat().st_size, "sha256": sha256_file(path)}:
            raise RuntimeError(f"CNN dataset file hash mismatch: {name}")
    train = np.load(OUTPUT_DIR / "train_log_mel.npy", mmap_mode="r")
    validation = np.load(OUTPUT_DIR / "validation_log_mel.npy", mmap_mode="r")
    train_labels = np.load(OUTPUT_DIR / "train_labels.npy", mmap_mode="r")
    validation_labels = np.load(OUTPUT_DIR / "validation_labels.npy", mmap_mode="r")
    if train.shape != (960, 64, 219, 1) or validation.shape != (240, 64, 219, 1):
        raise RuntimeError("CNN feature array shapes are invalid.")
    if train.dtype != np.float32 or validation.dtype != np.float32 or train_labels.dtype.kind not in "iu" or validation_labels.dtype.kind not in "iu":
        raise RuntimeError("CNN dataset dtypes are invalid.")
    if not np.isfinite(train).all() or not np.isfinite(validation).all() or not np.isfinite(train_labels).all() or not np.isfinite(validation_labels).all():
        raise RuntimeError("CNN dataset contains non-finite values.")
    if train_labels.min() < 0 or train_labels.max() > 7 or validation_labels.min() < 0 or validation_labels.max() > 7:
        raise RuntimeError("CNN labels are outside class IDs 0 through 7.")
    class_ids = {name: index for index, name in enumerate(config["class_order"])}
    expected_train_labels = np.asarray([class_ids[row["emotion"]] for row in load_rows("train", set(config["splits"]["train_actors"]))])
    expected_validation_labels = np.asarray([class_ids[row["emotion"]] for row in load_rows("validation", set(config["splits"]["validation_actors"]))])
    if not np.array_equal(train_labels, expected_train_labels) or not np.array_equal(validation_labels, expected_validation_labels):
        raise RuntimeError("CNN labels do not preserve deterministic manifest order.")
    mean_error = float(np.max(np.abs(np.mean(train, axis=(0, 2, 3)))))
    std_error = float(np.max(np.abs(np.std(train, axis=(0, 2, 3)) - 1.0)))
    if mean_error > 1e-5 or std_error > 1e-5:
        raise RuntimeError("Training normalization does not have approximately zero mean and unit standard deviation.")
    if summary["splits"]["train"]["class_counts"] != {name: int(np.sum(train_labels == index)) for index, name in enumerate(config["class_order"])}:
        raise RuntimeError("Training label counts do not match the summary.")
    if summary["splits"]["validation"]["class_counts"] != {name: int(np.sum(validation_labels == index)) for index, name in enumerate(config["class_order"])}:
        raise RuntimeError("Validation label counts do not match the summary.")
    if summary["test_actors_accessed"] or summary["model_trained"] or summary["raw_audio_modified"]:
        raise RuntimeError("CNN dataset summary contains an invalid safety status.")
    print(f"CNN dataset verification passed: mean error={mean_error:.3e}, std error={std_error:.3e}")


if __name__ == "__main__":
    main()
