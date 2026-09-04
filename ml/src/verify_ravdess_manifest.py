from __future__ import annotations

import argparse
import csv
import json
import ntpath
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

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
DURATION_TOLERANCE_SECONDS = 0.000001


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def split_for_actor(actor_id: int) -> str:
    for split, actor_ids in SPLIT_ACTORS.items():
        if actor_id in actor_ids:
            return split
    raise ValueError(f"Actor ID is outside the approved split ranges: {actor_id}")


def validate_filename_fields(row: dict[str, str], actor_directory: str) -> None:
    filename = row["filename"]
    if not filename.endswith(".wav"):
        raise ValueError(f"Expected a WAV filename: {filename}")
    fields = filename[:-4].split("-")
    if len(fields) != 7:
        raise ValueError(f"Expected seven filename fields: {filename}")
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = fields
    expected = [row["modality_id"], row["vocal_channel_id"], row["emotion_id"], row["intensity_id"], row["statement_id"], row["repetition_id"], row["actor_id"]]
    if fields != expected:
        raise ValueError(f"Manifest fields disagree with filename: {filename}")
    if modality != "03" or vocal_channel != "01" or emotion not in EMOTIONS:
        raise ValueError(f"Invalid modality, channel, or emotion in {filename}")
    if intensity not in {"01", "02"} or (emotion == "01" and intensity == "02"):
        raise ValueError(f"Invalid intensity in {filename}")
    if statement not in {"01", "02"} or repetition not in {"01", "02"}:
        raise ValueError(f"Invalid statement or repetition in {filename}")
    if actor not in {f"{number:02d}" for number in range(1, 25)} or actor_directory != f"Actor_{actor}":
        raise ValueError(f"Actor does not match directory for {filename}")
    if row["emotion"] != EMOTIONS[emotion] or row["intensity"] != ("normal" if intensity == "01" else "strong"):
        raise ValueError(f"Manifest labels disagree with filename: {filename}")


def validate_audio_row(row: dict[str, str], audio_path: Path) -> None:
    info = sf.info(audio_path)
    if info.format != "WAV" or info.samplerate != 48_000 or info.subtype != "PCM_16" or info.channels not in {1, 2} or info.frames <= 0:
        raise ValueError(f"Unexpected audio properties for {audio_path}")
    expected = [str(info.samplerate), str(info.channels), info.subtype, str(info.frames)]
    actual = [row["sample_rate"], row["channels"], row["subtype"], row["frames"]]
    if actual != expected:
        raise ValueError(f"Manifest audio fields disagree with header: {audio_path}")
    try:
        manifest_duration = float(row["duration_seconds"])
    except ValueError as error:
        raise ValueError(f"Invalid duration value for {audio_path}") from error
    if f"{manifest_duration:.{DURATION_PRECISION}f}" != row["duration_seconds"]:
        raise ValueError(f"Duration does not use fixed precision for {audio_path}")
    if abs(manifest_duration - (info.frames / info.samplerate)) > DURATION_TOLERANCE_SECONDS:
        raise ValueError(f"Duration differs from header for {audio_path}")


def duration_summary(durations: list[float], include_total: bool) -> dict[str, float]:
    result = {"minimum": round(min(durations), DURATION_PRECISION), "maximum": round(max(durations), DURATION_PRECISION), "mean": round(statistics.mean(durations), DURATION_PRECISION), "median": round(statistics.median(durations), DURATION_PRECISION)}
    if include_total:
        result["total"] = round(sum(durations), DURATION_PRECISION)
    return result


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def summarise_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["split"]].append(row)
    splits: dict[str, object] = {}
    for split in SPLIT_ACTORS:
        split_rows = grouped[split]
        splits[split] = {
            "actor_ids": [f"{number:02d}" for number in SPLIT_ACTORS[split]], "recording_count": len(split_rows), "actor_count": len(SPLIT_ACTORS[split]),
            "gender_counts": counter_dict(Counter(row["gender"] for row in split_rows)), "emotion_counts": counter_dict(Counter(row["emotion"] for row in split_rows)),
            "intensity_counts": counter_dict(Counter(row["intensity"] for row in split_rows)), "channel_counts": {"mono": sum(row["channels"] == "1" for row in split_rows), "stereo": sum(row["channels"] == "2" for row in split_rows)},
            "duration_seconds": duration_summary([float(row["duration_seconds"]) for row in split_rows], False),
        }
    return {"total_recording_count": len(rows), "total_actor_count": 24, "splits": splits, "overall": {
        "gender_counts": counter_dict(Counter(row["gender"] for row in rows)), "emotion_counts": counter_dict(Counter(row["emotion"] for row in rows)),
        "intensity_counts": counter_dict(Counter(row["intensity"] for row in rows)), "channel_counts": {"mono": sum(row["channels"] == "1" for row in rows), "stereo": sum(row["channels"] == "2" for row in rows)},
        "duration_seconds": duration_summary([float(row["duration_seconds"]) for row in rows], True),
    }}


def validate_expected_counts(summary: dict[str, object], actor_sets: dict[str, set[str]]) -> None:
    if summary["total_recording_count"] != 1_440 or summary["total_actor_count"] != 24:
        raise ValueError("Unexpected overall recording or actor count")
    if any(actor_sets[first] & actor_sets[second] for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Speaker leakage detected between split actor sets")
    if set().union(*actor_sets.values()) != {f"{number:02d}" for number in range(1, 25)}:
        raise ValueError("Split actor sets do not contain all actors")
    splits = summary["splits"]
    if not isinstance(splits, dict):
        raise ValueError("Invalid split summary structure")
    for split, expected in EXPECTED_SPLITS.items():
        actual = splits[split]
        if not isinstance(actual, dict):
            raise ValueError(f"Invalid {split} summary structure")
        checks = {"recording_count": expected["recordings"], "actor_count": expected["actors"], "gender_counts": expected["gender"], "emotion_counts": expected["emotion"], "intensity_counts": expected["intensity"], "channel_counts": expected["channel"]}
        for key, expected_value in checks.items():
            if actual[key] != expected_value:
                raise ValueError(f"Unexpected {split} {key}: {actual[key]}")


def validate_manifest(dataset_directory: Path, manifest_path: Path, summary_path: Path) -> None:
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Manifest CSV or split summary JSON is missing")
    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"Unexpected CSV header: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 1_440:
        raise ValueError(f"Expected 1,440 manifest rows, found {len(rows)}")

    root = repository_root().resolve()
    dataset_root = dataset_directory.resolve()
    paths: set[str] = set()
    filenames: set[str] = set()
    actor_counts: Counter[str] = Counter()
    actor_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if set(row) != set(CSV_COLUMNS) or None in row.values():
            raise ValueError("Malformed manifest row")
        relative_path = row["relative_path"]
        if "\\" in relative_path or relative_path in paths or row["filename"] in filenames:
            raise ValueError(f"Duplicate or non-portable manifest path: {relative_path}")
        portable_path = PurePosixPath(relative_path)
        if portable_path.is_absolute() or ".." in portable_path.parts or ntpath.splitdrive(relative_path)[0]:
            raise ValueError(f"Absolute or unsafe manifest path: {relative_path}")
        audio_path = (root / Path(*portable_path.parts)).resolve()
        if not audio_path.is_relative_to(root) or not audio_path.is_relative_to(dataset_root) or not audio_path.is_file():
            raise ValueError(f"Manifest path is outside the extracted dataset or missing: {relative_path}")
        if audio_path.relative_to(root).as_posix() != relative_path:
            raise ValueError(f"Manifest path is not repository-relative: {relative_path}")
        validate_filename_fields(row, audio_path.parent.name)
        actor_id = int(row["actor_id"])
        if row["gender"] != ("male" if actor_id % 2 else "female") or row["split"] != split_for_actor(actor_id):
            raise ValueError(f"Invalid gender or split for {relative_path}")
        validate_audio_row(row, audio_path)
        paths.add(relative_path)
        filenames.add(row["filename"])
        actor_counts[row["actor_id"]] += 1
        actor_sets[row["split"]].add(row["actor_id"])

    extracted_wavs = {path.relative_to(root).as_posix() for path in dataset_root.rglob("*.wav")}
    if paths != extracted_wavs:
        raise ValueError("Manifest and extracted WAV file sets differ")
    if set(actor_counts) != {f"{number:02d}" for number in range(1, 25)} or any(count != 60 for count in actor_counts.values()):
        raise ValueError("Unexpected manifest rows per actor")

    summary = summarise_rows(rows)
    validate_expected_counts(summary, actor_sets)
    with summary_path.open("r", encoding="utf-8") as summary_file:
        saved_summary = json.load(summary_file)
    if saved_summary != summary:
        raise ValueError("Split summary does not match values independently recalculated from the manifest")
    serialized_summary = json.dumps(saved_summary)
    if "C:\\" in serialized_summary or "olivia.dogbey" in serialized_summary.lower() or "\\\\" in serialized_summary:
        raise ValueError("Split summary contains a local absolute path or environment-specific value")
    metadata_directory = manifest_path.parent
    required_metadata = {manifest_path.name, summary_path.name}
    metadata_files = {path.name for path in metadata_directory.iterdir() if path.is_file()}
    if not required_metadata.issubset(metadata_files):
        raise ValueError("Required manifest metadata files are missing.")
    unsafe_suffixes = {
        ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".zip", ".npy", ".npz",
        ".pkl", ".joblib", ".h5", ".keras", ".onnx", ".pt", ".pth", ".ckpt",
        ".tmp", ".part",
    }
    for entry in metadata_directory.iterdir():
        if entry.is_dir() or entry.name.lower() in {".cache", "cache", "__pycache__"}:
            raise ValueError(f"Unsafe directory in metadata directory: {entry.name}")
        if entry.suffix.lower() in unsafe_suffixes or entry.suffix.lower() not in {".csv", ".json"}:
            raise ValueError(f"Unsafe or unsupported metadata file: {entry.name}")
    print("Split intersections: train/validation=[], train/test=[], validation/test=[]")
    print(f"Duration tolerance seconds: {DURATION_TOLERANCE_SECONDS}")


def parse_arguments() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description="Independently verify RAVDESS metadata manifest and split summary.")
    parser.add_argument("dataset_directory", nargs="?", type=Path, default=root / "ml" / "data" / "raw" / "ravdess" / "extracted")
    parser.add_argument("manifest_path", nargs="?", type=Path, default=root / "ml" / "metadata" / "ravdess_manifest.csv")
    parser.add_argument("summary_path", nargs="?", type=Path, default=root / "ml" / "metadata" / "ravdess_split_summary.json")
    return parser.parse_args()


def main() -> int:
    try:
        arguments = parse_arguments()
        validate_manifest(arguments.dataset_directory, arguments.manifest_path, arguments.summary_path)
    except Exception as error:
        print(f"RAVDESS manifest verification failed: {error}", file=sys.stderr)
        return 1
    print("RAVDESS manifest verification completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
