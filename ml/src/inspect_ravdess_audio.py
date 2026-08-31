from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from verify_ravdess_archive import EMOTION_NAMES, EXPECTED_ACTORS, EXPECTED_EMOTION_COUNTS, validate_filename


EXPECTED_WAV_COUNT = 1_440


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def duration_statistics(durations: list[float]) -> dict[str, float]:
    return {
        "minimum": min(durations),
        "maximum": max(durations),
        "mean": statistics.mean(durations),
        "median": statistics.median(durations),
    }


def inspect_audio_file(path: Path) -> tuple[float, int, int, str, str, bool | None, int]:
    with sf.SoundFile(path) as audio_file:
        if audio_file.format != "WAV":
            raise RuntimeError(f"Unexpected audio format for {path}: {audio_file.format}")
        if audio_file.samplerate != 48_000:
            raise RuntimeError(f"Unexpected sample rate for {path}: {audio_file.samplerate}")
        if audio_file.channels not in {1, 2}:
            raise RuntimeError(f"Unexpected channel count for {path}: {audio_file.channels}")
        if audio_file.subtype != "PCM_16":
            raise RuntimeError(f"Unexpected audio subtype for {path}: {audio_file.subtype}")
        if audio_file.frames <= 0:
            raise RuntimeError(f"Audio file has no frames: {path}")

        frames_read = 0
        channels_are_identical: bool | None = True if audio_file.channels == 2 else None
        maximum_channel_difference = 0
        while True:
            samples = audio_file.read(frames=65_536, dtype="int16", always_2d=True)
            if samples.size == 0:
                break
            if not np.isfinite(samples).all():
                raise RuntimeError(f"Audio contains non-finite samples: {path}")
            if audio_file.channels == 2:
                channel_differences = np.abs(samples[:, 0].astype(np.int32) - samples[:, 1].astype(np.int32))
                if np.any(channel_differences):
                    channels_are_identical = False
                maximum_channel_difference = max(maximum_channel_difference, int(channel_differences.max()))
            frames_read += samples.shape[0]
        if frames_read != audio_file.frames:
            raise RuntimeError(
                f"Read frame count differs from metadata for {path}: {frames_read} != {audio_file.frames}"
            )

        duration = audio_file.frames / audio_file.samplerate
        if duration <= 0:
            raise RuntimeError(f"Audio file has non-positive duration: {path}")
        return (
            duration,
            audio_file.samplerate,
            audio_file.channels,
            audio_file.subtype,
            audio_file.format,
            channels_are_identical,
            maximum_channel_difference,
        )


def inspect_dataset(dataset_directory: Path) -> None:
    if not dataset_directory.is_dir():
        raise FileNotFoundError(f"Extracted dataset directory does not exist: {dataset_directory}")

    root_entries = list(dataset_directory.iterdir())
    actor_directories = {entry.name for entry in root_entries if entry.is_dir()}
    unexpected_root_entries = [entry.name for entry in root_entries if not entry.is_dir()]
    if unexpected_root_entries:
        raise RuntimeError(f"Unexpected files in dataset root: {sorted(unexpected_root_entries)}")
    if actor_directories != EXPECTED_ACTORS:
        raise RuntimeError(
            f"Actor directory mismatch; missing={sorted(EXPECTED_ACTORS - actor_directories)}, "
            f"unexpected={sorted(actor_directories - EXPECTED_ACTORS)}"
        )

    actor_counts: Counter[str] = Counter()
    gender_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    intensity_counts: Counter[str] = Counter()
    sample_rate_counts: Counter[int] = Counter()
    channel_counts: Counter[int] = Counter()
    subtype_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    durations: list[float] = []
    emotion_durations: defaultdict[str, list[float]] = defaultdict(list)
    emotion_channel_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    actor_channel_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    statement_channel_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    identical_channel_files = 0
    differing_channel_files = 0
    maximum_channel_difference = 0

    for actor_directory in sorted(EXPECTED_ACTORS):
        actor_path = dataset_directory / actor_directory
        actor_entries = list(actor_path.iterdir())
        unexpected_entries = [entry.name for entry in actor_entries if not entry.is_file() or entry.suffix.lower() != ".wav"]
        if unexpected_entries:
            raise RuntimeError(f"Unexpected entries in {actor_directory}: {sorted(unexpected_entries)}")
        if len(actor_entries) != 60:
            raise RuntimeError(f"Expected 60 WAV files in {actor_directory}, found {len(actor_entries)}")

        for audio_path in sorted(actor_entries):
            emotion_code = validate_filename(audio_path.name, actor_directory)
            (
                duration,
                sample_rate,
                channels,
                subtype,
                audio_format,
                channels_are_identical,
                file_maximum_channel_difference,
            ) = inspect_audio_file(audio_path)
            actor_counts[actor_directory] += 1
            actor_id = int(actor_directory.removeprefix("Actor_"))
            gender_counts["male" if actor_id % 2 else "female"] += 1
            emotion_name = EMOTION_NAMES[emotion_code]
            emotion_counts[emotion_name] += 1
            intensity_name = "normal" if audio_path.name.split("-")[3] == "01" else "strong"
            intensity_counts[intensity_name] += 1
            sample_rate_counts[sample_rate] += 1
            channel_counts[channels] += 1
            subtype_counts[subtype] += 1
            format_counts[audio_format] += 1
            durations.append(duration)
            emotion_durations[emotion_name].append(duration)
            emotion_channel_counts[emotion_name][channels] += 1
            actor_channel_counts[actor_directory][channels] += 1
            statement_channel_counts[audio_path.name.split("-")[4]][channels] += 1
            if channels == 2:
                if channels_are_identical:
                    identical_channel_files += 1
                else:
                    differing_channel_files += 1
                maximum_channel_difference = max(maximum_channel_difference, file_maximum_channel_difference)

    if sum(actor_counts.values()) != EXPECTED_WAV_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_WAV_COUNT} WAV files, found {sum(actor_counts.values())}")
    if any(actor_counts[actor] != 60 for actor in EXPECTED_ACTORS):
        raise RuntimeError(f"Unexpected actor file counts: {dict(sorted(actor_counts.items()))}")
    if dict(emotion_counts) != EXPECTED_EMOTION_COUNTS:
        raise RuntimeError(f"Unexpected emotion counts: {dict(sorted(emotion_counts.items()))}")
    if dict(sample_rate_counts) != {48_000: EXPECTED_WAV_COUNT}:
        raise RuntimeError(f"Unexpected sample-rate counts: {dict(sample_rate_counts)}")
    if sum(channel_counts.values()) != EXPECTED_WAV_COUNT or not set(channel_counts).issubset({1, 2}):
        raise RuntimeError(f"Unexpected channel counts: {dict(channel_counts)}")
    if dict(subtype_counts) != {"PCM_16": EXPECTED_WAV_COUNT}:
        raise RuntimeError(f"Unexpected subtype counts: {dict(subtype_counts)}")
    if dict(format_counts) != {"WAV": EXPECTED_WAV_COUNT}:
        raise RuntimeError(f"Unexpected format counts: {dict(format_counts)}")

    overall = duration_statistics(durations)
    print(f"Total readable WAV files: {len(durations)}")
    print(f"Actor counts: {dict(sorted(actor_counts.items()))}")
    print(f"Gender counts: {dict(sorted(gender_counts.items()))}")
    print(f"Emotion counts: {dict(emotion_counts)}")
    print(f"Intensity counts: {dict(sorted(intensity_counts.items()))}")
    print(f"Sample-rate counts: {dict(sorted(sample_rate_counts.items()))}")
    print(f"Channel counts: {dict(sorted(channel_counts.items()))}")
    print(f"Mono files: {channel_counts[1]}")
    print(f"Stereo files: {channel_counts[2]}")
    print(f"Audio subtype counts: {dict(sorted(subtype_counts.items()))}")
    print(f"Audio format counts: {dict(sorted(format_counts.items()))}")
    print(
        "Channel counts by emotion: "
        f"{ {emotion: dict(sorted(emotion_channel_counts[emotion].items())) for emotion in EXPECTED_EMOTION_COUNTS} }"
    )
    print(
        "Channel counts by actor: "
        f"{ {actor: dict(sorted(actor_channel_counts[actor].items())) for actor in sorted(EXPECTED_ACTORS)} }"
    )
    print(
        "Channel counts by statement: "
        f"{ {statement: dict(sorted(statement_channel_counts[statement].items())) for statement in sorted(statement_channel_counts)} }"
    )
    print(
        "Stereo channel comparison: "
        f"identical_files={identical_channel_files}, differing_files={differing_channel_files}, "
        f"maximum_absolute_sample_difference={maximum_channel_difference}"
    )
    print(
        "Overall duration seconds: "
        f"min={overall['minimum']:.6f}, max={overall['maximum']:.6f}, "
        f"mean={overall['mean']:.6f}, median={overall['median']:.6f}, total={sum(durations):.6f}"
    )
    print("Per-emotion duration seconds:")
    for emotion_name in EXPECTED_EMOTION_COUNTS:
        values = duration_statistics(emotion_durations[emotion_name])
        print(
            f"  {emotion_name}: min={values['minimum']:.6f}, max={values['maximum']:.6f}, "
            f"mean={values['mean']:.6f}, median={values['median']:.6f}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect every extracted RAVDESS WAV file without modifying it.")
    parser.add_argument(
        "dataset_directory",
        nargs="?",
        type=Path,
        default=repository_root() / "ml" / "data" / "raw" / "ravdess" / "extracted",
    )
    return parser.parse_args()


def main() -> int:
    try:
        inspect_dataset(parse_arguments().dataset_directory)
    except Exception as error:
        print(f"RAVDESS audio inspection failed: {error}", file=sys.stderr)
        return 1

    print("RAVDESS audio inspection completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
