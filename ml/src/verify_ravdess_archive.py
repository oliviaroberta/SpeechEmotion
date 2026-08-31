from __future__ import annotations

import argparse
import hashlib
import ntpath
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


EXPECTED_MD5 = "bc696df654c87fed845eb13823edef8a"
EXPECTED_WAV_COUNT = 1_440
EXPECTED_ACTORS = {f"Actor_{actor_id:02d}" for actor_id in range(1, 25)}
EXPECTED_EMOTION_COUNTS = {
    "neutral": 96,
    "calm": 192,
    "happy": 192,
    "sad": 192,
    "angry": 192,
    "fearful": 192,
    "disgust": 192,
    "surprised": 192,
}
EMOTION_NAMES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def calculate_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_zip_path(name: str) -> PurePosixPath:
    if not name:
        raise ValueError("ZIP entry has an empty name.")
    if name.startswith(("/", "\\")) or ntpath.splitdrive(name)[0]:
        raise ValueError(f"Unsafe absolute or drive-qualified ZIP path: {name!r}")

    normalised = name.replace("\\", "/")
    path = PurePosixPath(normalised)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP path traversal: {name!r}")
    return path


def validate_filename(filename: str, actor_directory: str) -> str:
    fields = filename.removesuffix(".wav").split("-")
    if len(fields) != 7:
        raise ValueError(f"Expected seven filename fields: {filename}")

    modality, vocal_channel, emotion, intensity, statement, repetition, actor = fields
    if modality != "03":
        raise ValueError(f"Invalid modality in {filename}: {modality}")
    if vocal_channel != "01":
        raise ValueError(f"Invalid vocal channel in {filename}: {vocal_channel}")
    if emotion not in EMOTION_NAMES:
        raise ValueError(f"Invalid emotion in {filename}: {emotion}")
    if intensity not in {"01", "02"}:
        raise ValueError(f"Invalid intensity in {filename}: {intensity}")
    if emotion == "01" and intensity == "02":
        raise ValueError(f"Neutral recording has strong intensity: {filename}")
    if statement not in {"01", "02"}:
        raise ValueError(f"Invalid statement in {filename}: {statement}")
    if repetition not in {"01", "02"}:
        raise ValueError(f"Invalid repetition in {filename}: {repetition}")
    if actor not in {f"{actor_id:02d}" for actor_id in range(1, 25)}:
        raise ValueError(f"Invalid actor ID in {filename}: {actor}")
    if actor_directory != f"Actor_{actor}":
        raise ValueError(
            f"Actor directory does not match filename actor ID: {actor_directory}/{filename}"
        )
    return emotion


def validate_archive(archive_path: Path) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive does not exist: {archive_path}")

    archive_size = archive_path.stat().st_size
    archive_md5 = calculate_md5(archive_path)
    print(f"Archive path: {archive_path}")
    print(f"Archive byte size: {archive_size}")
    print(f"Archive MD5: {archive_md5}")
    if archive_md5.lower() != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: expected {EXPECTED_MD5}, got {archive_md5}")

    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        for entry in entries:
            if entry.flag_bits & 0x1:
                raise RuntimeError(f"Encrypted ZIP entry is not allowed: {entry.filename}")
            normalise_zip_path(entry.filename)

        failed_member = archive.testzip()
        if failed_member is not None:
            raise RuntimeError(f"ZIP CRC/integrity validation failed for: {failed_member}")
        print("ZIP CRC/integrity validation: passed")

        wav_entries: list[tuple[PurePosixPath, zipfile.ZipInfo]] = []
        unexpected_files: list[str] = []
        for entry in entries:
            path = normalise_zip_path(entry.filename)
            if entry.is_dir() or entry.filename.endswith("/"):
                continue
            if path.suffix.lower() != ".wav":
                unexpected_files.append(entry.filename)
                continue
            wav_entries.append((path, entry))

    if unexpected_files:
        raise RuntimeError(
            "Unexpected non-WAV files: " + ", ".join(sorted(unexpected_files))
        )
    print("Unexpected non-WAV files: none")

    if len(wav_entries) != EXPECTED_WAV_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_WAV_COUNT} WAV files, found {len(wav_entries)}")

    actor_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    gender_counts: Counter[str] = Counter()
    actor_directories: set[str] = set()
    for path, _entry in wav_entries:
        if len(path.parts) != 2:
            raise RuntimeError(f"WAV entry must be directly inside an actor directory: {path}")
        actor_directory, filename = path.parts
        actor_directories.add(actor_directory)
        if actor_directory not in EXPECTED_ACTORS:
            raise RuntimeError(f"Unexpected actor directory for WAV entry: {path}")

        emotion = validate_filename(filename, actor_directory)
        actor_counts[actor_directory] += 1
        emotion_counts[EMOTION_NAMES[emotion]] += 1
        actor_id = int(actor_directory.removeprefix("Actor_"))
        gender_counts["male" if actor_id % 2 else "female"] += 1

    if actor_directories != EXPECTED_ACTORS:
        missing = sorted(EXPECTED_ACTORS - actor_directories)
        unexpected = sorted(actor_directories - EXPECTED_ACTORS)
        raise RuntimeError(f"Actor directory mismatch; missing={missing}, unexpected={unexpected}")
    if any(actor_counts[actor] != 60 for actor in EXPECTED_ACTORS):
        counts = ", ".join(f"{actor}={actor_counts[actor]}" for actor in sorted(EXPECTED_ACTORS))
        raise RuntimeError(f"Expected 60 WAV files per actor; found {counts}")
    if dict(emotion_counts) != EXPECTED_EMOTION_COUNTS:
        raise RuntimeError(
            f"Emotion totals differ from expectation: {dict(sorted(emotion_counts.items()))}"
        )

    print(f"WAV entries: {len(wav_entries)}")
    print(f"Actor directories: {len(actor_directories)}")
    print("WAV files per actor: 60 each")
    print(f"Gender counts: male={gender_counts['male']}, female={gender_counts['female']}")
    print("Emotion distribution:")
    for emotion, expected_count in EXPECTED_EMOTION_COUNTS.items():
        print(f"  {emotion}: {emotion_counts[emotion]}")


def parse_arguments() -> argparse.Namespace:
    default_archive = (
        repository_root()
        / "ml"
        / "data"
        / "raw"
        / "ravdess"
        / "archive"
        / "Audio_Speech_Actors_01-24.zip"
    )
    parser = argparse.ArgumentParser(description="Validate the RAVDESS speech ZIP archive without extraction.")
    parser.add_argument("archive", nargs="?", type=Path, default=default_archive)
    return parser.parse_args()


def main() -> int:
    try:
        validate_archive(parse_arguments().archive)
    except Exception as error:
        print(f"RAVDESS archive verification failed: {error}", file=sys.stderr)
        return 1

    print("RAVDESS archive verification completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
