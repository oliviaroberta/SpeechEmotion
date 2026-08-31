from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


CSV_COLUMNS = [
    "relative_path", "filename", "modality_id", "vocal_channel_id", "emotion_id", "emotion",
    "intensity_id", "intensity", "statement_id", "repetition_id", "actor_id", "gender", "split",
    "sample_rate", "channels", "subtype", "frames", "duration_seconds",
]
EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
SPLIT_ACTORS = {"train": list(range(1, 17)), "validation": list(range(17, 21)), "test": list(range(21, 25))}
EXPECTED_SPLITS = {
    "train": {"recordings": 960, "actors": 16, "gender": {"male": 480, "female": 480}, "emotion": {"neutral": 64, **{name: 128 for name in list(EMOTIONS.values())[1:]}}, "intensity": {"normal": 512, "strong": 448}, "channel": {"mono": 957, "stereo": 3}},
    "validation": {"recordings": 240, "actors": 4, "gender": {"male": 120, "female": 120}, "emotion": {"neutral": 16, **{name: 32 for name in list(EMOTIONS.values())[1:]}}, "intensity": {"normal": 128, "strong": 112}, "channel": {"mono": 238, "stereo": 2}},
    "test": {"recordings": 240, "actors": 4, "gender": {"male": 120, "female": 120}, "emotion": {"neutral": 16, **{name: 32 for name in list(EMOTIONS.values())[1:]}}, "intensity": {"normal": 128, "strong": 112}, "channel": {"mono": 240, "stereo": 0}},
}
EXPECTED_ACTORS = {f"Actor_{number:02d}" for number in range(1, 25)}
DURATION_PRECISION = 6


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def split_for_actor(actor_id: int) -> str:
    for split, actor_ids in SPLIT_ACTORS.items():
        if actor_id in actor_ids:
            return split
    raise ValueError(f"Actor ID is outside the approved split ranges: {actor_id}")


def parse_filename(filename: str, actor_directory: str) -> tuple[str, str, str, str, str, str, str]:
    if not filename.endswith(".wav"):
        raise ValueError(f"Expected a WAV filename: {filename}")
    fields = filename[:-4].split("-")
    if len(fields) != 7:
        raise ValueError(f"Expected seven filename fields: {filename}")
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = fields
    if modality != "03" or vocal_channel != "01" or emotion not in EMOTIONS:
        raise ValueError(f"Invalid modality, channel, or emotion in {filename}")
    if intensity not in {"01", "02"} or (emotion == "01" and intensity == "02"):
        raise ValueError(f"Invalid intensity in {filename}")
    if statement not in {"01", "02"} or repetition not in {"01", "02"}:
        raise ValueError(f"Invalid statement or repetition in {filename}")
    if actor not in {f"{number:02d}" for number in range(1, 25)}:
        raise ValueError(f"Invalid actor ID in {filename}")
    if actor_directory != f"Actor_{actor}":
        raise ValueError(f"Actor directory does not match filename: {actor_directory}/{filename}")
    return modality, vocal_channel, emotion, intensity, statement, repetition, actor


def inspect_file(path: Path) -> tuple[int, int, str, int, float]:
    info = sf.info(path)
    if info.format != "WAV" or info.samplerate != 48_000 or info.subtype != "PCM_16":
        raise ValueError(f"Unexpected audio format properties for {path}")
    if info.channels not in {1, 2} or info.frames <= 0:
        raise ValueError(f"Unexpected channel count or empty audio for {path}")
    duration = info.frames / info.samplerate
    if duration <= 0:
        raise ValueError(f"Non-positive duration for {path}")
    return info.samplerate, info.channels, info.subtype, info.frames, duration


def build_rows(dataset_directory: Path) -> list[dict[str, str]]:
    if not dataset_directory.is_dir():
        raise FileNotFoundError(f"Extracted dataset directory does not exist: {dataset_directory}")
    actor_directories = {path.name for path in dataset_directory.iterdir() if path.is_dir()}
    if actor_directories != EXPECTED_ACTORS:
        raise ValueError(f"Actor directory mismatch: {sorted(actor_directories)}")

    rows: list[dict[str, str]] = []
    root = repository_root()
    for actor_number in range(1, 25):
        actor_directory = f"Actor_{actor_number:02d}"
        actor_path = dataset_directory / actor_directory
        files = sorted(actor_path.glob("*.wav"))
        if len(files) != 60 or any(not path.is_file() for path in files):
            raise ValueError(f"Expected 60 WAV files in {actor_directory}, found {len(files)}")
        if len(list(actor_path.iterdir())) != 60:
            raise ValueError(f"Unexpected entry in {actor_directory}")
        for path in files:
            modality, vocal_channel, emotion, intensity, statement, repetition, actor = parse_filename(path.name, actor_directory)
            sample_rate, channels, subtype, frames, duration = inspect_file(path)
            rows.append({
                "relative_path": path.relative_to(root).as_posix(), "filename": path.name,
                "modality_id": modality, "vocal_channel_id": vocal_channel, "emotion_id": emotion,
                "emotion": EMOTIONS[emotion], "intensity_id": intensity,
                "intensity": "normal" if intensity == "01" else "strong", "statement_id": statement,
                "repetition_id": repetition, "actor_id": actor,
                "gender": "male" if actor_number % 2 else "female", "split": split_for_actor(actor_number),
                "sample_rate": str(sample_rate), "channels": str(channels), "subtype": subtype,
                "frames": str(frames), "duration_seconds": f"{duration:.{DURATION_PRECISION}f}",
            })
    if len(rows) != 1_440:
        raise ValueError(f"Expected 1,440 rows, found {len(rows)}")
    return rows


def duration_summary(durations: list[float], include_total: bool) -> dict[str, float]:
    result = {"minimum": round(min(durations), DURATION_PRECISION), "maximum": round(max(durations), DURATION_PRECISION), "mean": round(statistics.mean(durations), DURATION_PRECISION), "median": round(statistics.median(durations), DURATION_PRECISION)}
    if include_total:
        result["total"] = round(sum(durations), DURATION_PRECISION)
    return result


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def summarise_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    splits: dict[str, object] = {}
    for split in SPLIT_ACTORS:
        split_rows = by_split[split]
        splits[split] = {
            "actor_ids": [f"{number:02d}" for number in SPLIT_ACTORS[split]],
            "recording_count": len(split_rows), "actor_count": len(SPLIT_ACTORS[split]),
            "gender_counts": counter_dict(Counter(row["gender"] for row in split_rows)),
            "emotion_counts": counter_dict(Counter(row["emotion"] for row in split_rows)),
            "intensity_counts": counter_dict(Counter(row["intensity"] for row in split_rows)),
            "channel_counts": {"mono": sum(row["channels"] == "1" for row in split_rows), "stereo": sum(row["channels"] == "2" for row in split_rows)},
            "duration_seconds": duration_summary([float(row["duration_seconds"]) for row in split_rows], False),
        }
    return {
        "total_recording_count": len(rows), "total_actor_count": 24, "splits": splits,
        "overall": {
            "gender_counts": counter_dict(Counter(row["gender"] for row in rows)),
            "emotion_counts": counter_dict(Counter(row["emotion"] for row in rows)),
            "intensity_counts": counter_dict(Counter(row["intensity"] for row in rows)),
            "channel_counts": {"mono": sum(row["channels"] == "1" for row in rows), "stereo": sum(row["channels"] == "2" for row in rows)},
            "duration_seconds": duration_summary([float(row["duration_seconds"]) for row in rows], True),
        },
    }


def validate_expected(summary: dict[str, object]) -> None:
    if summary["total_recording_count"] != 1_440 or summary["total_actor_count"] != 24:
        raise ValueError("Unexpected overall recording or actor count")
    splits = summary["splits"]
    assert isinstance(splits, dict)
    for split, expected in EXPECTED_SPLITS.items():
        actual = splits[split]
        assert isinstance(actual, dict)
        comparisons = {"recording_count": expected["recordings"], "actor_count": expected["actors"], "gender_counts": expected["gender"], "emotion_counts": expected["emotion"], "intensity_counts": expected["intensity"], "channel_counts": expected["channel"]}
        for key, value in comparisons.items():
            if actual[key] != value:
                raise ValueError(f"Unexpected {split} {key}: {actual[key]}")


def write_outputs(rows: list[dict[str, str]], summary: dict[str, object], manifest_path: Path, summary_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.parent != manifest_path.parent:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(f"{manifest_path.name}.tmp")
    temporary_summary = summary_path.with_name(f"{summary_path.name}.tmp")
    if temporary_manifest.exists() or temporary_summary.exists():
        raise RuntimeError("A temporary metadata output already exists; inspect it manually before rebuilding.")
    try:
        with temporary_manifest.open("w", encoding="utf-8", newline="") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        with temporary_summary.open("w", encoding="utf-8", newline="\n") as summary_file:
            json.dump(summary, summary_file, indent=2, sort_keys=True)
            summary_file.write("\n")
            summary_file.flush()
            os.fsync(summary_file.fileno())
        temporary_manifest.replace(manifest_path)
        temporary_summary.replace(summary_path)
    finally:
        for temporary_path in (temporary_manifest, temporary_summary):
            if temporary_path.exists():
                temporary_path.unlink()


def parse_arguments() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description="Build a deterministic RAVDESS metadata manifest.")
    parser.add_argument("dataset_directory", nargs="?", type=Path, default=root / "ml" / "data" / "raw" / "ravdess" / "extracted")
    parser.add_argument("manifest_path", nargs="?", type=Path, default=root / "ml" / "metadata" / "ravdess_manifest.csv")
    parser.add_argument("summary_path", nargs="?", type=Path, default=root / "ml" / "metadata" / "ravdess_split_summary.json")
    return parser.parse_args()


def main() -> int:
    try:
        arguments = parse_arguments()
        rows = build_rows(arguments.dataset_directory)
        summary = summarise_rows(rows)
        validate_expected(summary)
        write_outputs(rows, summary, arguments.manifest_path, arguments.summary_path)
    except Exception as error:
        print(f"RAVDESS manifest build failed: {error}", file=sys.stderr)
        return 1
    print("RAVDESS manifest build completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
