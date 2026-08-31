from __future__ import annotations

import argparse, csv, hashlib, json, os, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

MANIFEST_HASH = "079884EA3724C1ADAB1C53E33D9EB669B58FCC58E4F577379BBFF807A46C398D"
TOP_DBS = (20, 25, 30, 35, 40)
DURATIONS = (2.5, 3.0, 3.5, 4.0)
TARGET_SR, FRAME_LENGTH, HOP_LENGTH = 16_000, 2_048, 512

def root() -> Path: return Path(__file__).resolve().parents[2]
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest().upper()
def stats(values: list[float]) -> dict[str, float]:
    return {"minimum": round(min(values), 6), "maximum": round(max(values), 6), "mean": round(statistics.mean(values), 6), "median": round(statistics.median(values), 6), **{f"p{p}": round(float(np.percentile(values, p)), 6) for p in (5, 25, 75, 90, 95, 99)}}
def grouped(measures: list[dict[str, object]], field: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for measure in measures: groups[str(measure[field])].append(float(measure["retained_seconds"]))
    return {name: stats(values) for name, values in sorted(groups.items())}
def count_groups(measures: list[dict[str, object]], field: str, predicate: str) -> dict[str, int]:
    result: Counter[str] = Counter()
    for measure in measures:
        if bool(measure[predicate]): result[str(measure[field])] += 1
    return dict(sorted(result.items()))

def load_training_rows(manifest: Path) -> list[dict[str, str]]:
    if sha256(manifest) != MANIFEST_HASH: raise ValueError("Source manifest SHA-256 does not match the approved value")
    with manifest.open(encoding="utf-8", newline="") as file: rows = list(csv.DictReader(file))
    train = [row for row in rows if row["split"] == "train"]
    if len(train) != 960 or {row["actor_id"] for row in train} != {f"{n:02d}" for n in range(1, 17)}: raise ValueError("Training manifest rows do not match the approved actor split")
    if any(row["split"] != "train" for row in train): raise ValueError("Non-training row selected")
    if any(row["actor_id"] in {f"{n:02d}" for n in range(17, 25)} for row in train): raise ValueError("Evaluation actor selected")
    return train

def read_signal(path: Path) -> tuple[np.ndarray, int, int, float, float]:
    info = sf.info(path)
    if info.format != "WAV" or info.samplerate != 48_000 or info.subtype != "PCM_16" or info.channels not in {1, 2} or info.frames <= 0: raise ValueError(f"Invalid source audio: {path}")
    signal, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != info.samplerate or signal.size == 0 or not np.isfinite(signal).all(): raise ValueError(f"Unreadable or non-finite source audio: {path}")
    return signal, sr, info.channels, float(np.max(np.abs(signal))), float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))

def trim_summary(measures: list[dict[str, object]]) -> dict[str, object]:
    original = [float(x["original_seconds"]) for x in measures]; retained = [float(x["retained_seconds"]) for x in measures]
    return {"original_duration_seconds": stats(original), "retained_duration_seconds": stats(retained), "leading_removed_seconds": stats([float(x["leading_seconds"]) for x in measures]), "trailing_removed_seconds": stats([float(x["trailing_seconds"]) for x in measures]), "total_removed_seconds": stats([float(x["removed_seconds"]) for x in measures]), "retained_percentage": stats([float(x["retained_percent"]) for x in measures]), "no_trimming": {"count": sum(bool(x["no_trim"]) for x in measures), "percentage": round(100 * sum(bool(x["no_trim"]) for x in measures) / len(measures), 6)}, "retaining_less_than": {key: {"count": sum(float(x["retained_percent"]) < threshold for x in measures), "percentage": round(100 * sum(float(x["retained_percent"]) < threshold for x in measures) / len(measures), 6)} for key, threshold in (("95_percent", 95), ("90_percent", 90), ("80_percent", 80), ("70_percent", 70))}, "empty_signal_count": sum(float(x["retained_seconds"]) == 0 for x in measures), "per_emotion": grouped(measures, "emotion"), "per_gender": grouped(measures, "gender"), "per_intensity": grouped(measures, "intensity")}

def fixed_summary(measures: list[dict[str, object]], samples: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for duration in DURATIONS:
        target = round(duration * samples); padding = [max(0, target - int(x["retained_samples"])) / samples for x in measures]; truncation = [max(0, int(x["retained_samples"]) - target) / samples for x in measures]
        exact = [abs(int(x["retained_samples"]) - target) <= 1 for x in measures]
        entry = {"padding": {"count": sum(x > 0 for x in padding), "percentage": round(100 * sum(x > 0 for x in padding) / len(measures), 6), "total_seconds": round(sum(padding), 6), "mean_seconds": round(statistics.mean(padding), 6)}, "truncation": {"count": sum(x > 0 for x in truncation), "percentage": round(100 * sum(x > 0 for x in truncation) / len(measures), 6), "total_seconds": round(sum(truncation), 6), "mean_seconds": round(statistics.mean(truncation), 6)}, "exact_fit_count": sum(exact), "per_emotion": {}, "per_gender": {}, "per_intensity": {}}
        for field, key in (("emotion", "per_emotion"), ("gender", "per_gender"), ("intensity", "per_intensity")):
            groups: dict[str, dict[str, int]] = defaultdict(lambda: {"padding": 0, "truncation": 0, "exact_fit": 0})
            for measure, pad, cut, fit in zip(measures, padding, truncation, exact): groups[str(measure[field])]["padding"] += pad > 0; groups[str(measure[field])]["truncation"] += cut > 0; groups[str(measure[field])]["exact_fit"] += fit
            entry[key] = dict(sorted(groups.items()))
        result[f"{duration:.1f}"] = entry
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description="Run an in-memory, training-only RAVDESS preprocessing pilot.")
    parser.add_argument("manifest", nargs="?", type=Path, default=root() / "ml/metadata/ravdess_manifest.csv"); parser.add_argument("output", nargs="?", type=Path, default=root() / "ml/metadata/ravdess_preprocessing_pilot.json")
    args = parser.parse_args()
    try:
        rows = load_training_rows(args.manifest); trim_data: dict[str, list[dict[str, object]]] = {str(db): [] for db in TOP_DBS}; original_peaks=[]; original_rms=[]; resampled_peaks=[]; resampled_rms=[]; mono_count=stereo_count=identical_stereo=0
        for index, row in enumerate(rows, 1):
            path = root() / Path(*Path(row["relative_path"]).parts); signal, sr, channels, peak, rms = read_signal(path); original_peaks.append(peak); original_rms.append(rms)
            if channels == 2:
                stereo_count += 1
                if not np.array_equal(signal[:, 0], signal[:, 1]): raise ValueError(f"Stereo channels differ: {row['filename']}")
                mono = np.mean(signal, axis=1, dtype=np.float32); identical_stereo += 1
                if not np.allclose(mono, signal[:, 0], rtol=0, atol=1e-7): raise ValueError(f"Identical stereo downmix changed values: {row['filename']}")
            else: mono_count += 1; mono = signal[:, 0]
            resampled = librosa.resample(mono, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
            if resampled.ndim != 1 or resampled.size == 0 or not np.isfinite(resampled).all() or abs(len(resampled) - round(len(mono) * TARGET_SR / sr)) > 2: raise ValueError(f"Invalid resampling result: {row['filename']}")
            resampled_peaks.append(float(np.max(np.abs(resampled)))); resampled_rms.append(float(np.sqrt(np.mean(np.square(resampled, dtype=np.float64)))) )
            for db in TOP_DBS:
                trimmed, bounds = librosa.effects.trim(resampled, top_db=db, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
                start, end = map(int, bounds); retained = len(trimmed); original_seconds = len(resampled) / TARGET_SR
                trim_data[str(db)].append({"emotion": row["emotion"], "gender": row["gender"], "intensity": row["intensity"], "original_seconds": original_seconds, "retained_seconds": retained / TARGET_SR, "retained_samples": retained, "leading_seconds": start / TARGET_SR, "trailing_seconds": (len(resampled) - end) / TARGET_SR, "removed_seconds": (len(resampled) - retained) / TARGET_SR, "retained_percent": 100 * retained / len(resampled), "no_trim": retained == len(resampled)})
            if index % 240 == 0: print(f"Analysed training recordings: {index}/960")
        trim_results = {db: trim_summary(values) for db, values in trim_data.items()}; selected_db = "40"; selected_fixed = next((duration for duration in DURATIONS if trim_results[selected_db]["fixed_duration_candidates"][f"{duration:.1f}"]["truncation"]["percentage"] <= 5), None) if False else None
        fixed = {db: fixed_summary(values, TARGET_SR) for db, values in trim_data.items()}
        selected_fixed = next((duration for duration in DURATIONS if fixed[selected_db][f"{duration:.1f}"]["truncation"]["percentage"] <= 5), None)
        report = {"source_manifest_sha256": MANIFEST_HASH, "analysis_scope": {"split": "train", "actor_ids": [f"{n:02d}" for n in range(1,17)], "recording_count": 960, "validation_and_test_excluded": True}, "audio_written": False, "configuration": {"target_sample_rate": TARGET_SR, "resample_method": "soxr_hq", "frame_length": FRAME_LENGTH, "hop_length": HOP_LENGTH, "duration_precision": 6}, "input_channels": {"mono": mono_count, "stereo": stereo_count, "identical_stereo_files": identical_stereo}, "resampling_checks": {"all_passed": True, "length_tolerance_samples": 2, "out_of_range_count": sum(x > 1.0 for x in resampled_peaks)}, "silence_trimming": {db: {**trim_results[db], "fixed_duration_candidates": fixed[db]} for db in trim_data}, "amplitude": {"original_peak": stats(original_peaks), "original_rms": stats(original_rms), "resampled_peak": stats(resampled_peaks), "resampled_rms": stats(resampled_rms), "silent_file_count": sum(x == 0 for x in original_rms), "potentially_clipped_file_count": sum(x >= 0.999 for x in original_peaks), "normalization_decision": "Do not apply per-file peak normalization; decoded floating-point PCM is not normalization and loudness may be informative."}, "provisional_recommendation": {"status": "provisional_not_applied_pending_review", "top_db": int(selected_db), "frame_length": FRAME_LENGTH, "hop_length": HOP_LENGTH, "fixed_duration_seconds": selected_fixed, "sample_rate": TARGET_SR, "stereo_to_mono": "arithmetic mean of channels", "padding": "zero-pad deterministically, split between beginning and end", "truncation": "deterministic truncation; no random crop", "amplitude_normalization": "do not apply per-file peak normalization", "evidence": {"selected_trim_retained": trim_results[selected_db]["retained_duration_seconds"], "selected_trim_empty_count": trim_results[selected_db]["empty_signal_count"], "selected_duration_truncation_percent": None if selected_fixed is None else fixed[selected_db][f"{selected_fixed:.1f}"]["truncation"]["percentage"]}, "risks": "Energy trimming approximates boundaries and may remove low-energy speech; settings require review before application."}}
        temporary = args.output.with_name(args.output.name + ".tmp"); args.output.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists(): raise RuntimeError(f"Temporary pilot output exists: {temporary}")
        with temporary.open("w", encoding="utf-8", newline="\n") as file: json.dump(report, file, indent=2, sort_keys=True); file.write("\n"); file.flush(); os.fsync(file.fileno())
        temporary.replace(args.output); print("Training-only preprocessing pilot completed successfully.")
    except Exception as error:
        print(f"Preprocessing pilot failed: {error}", file=sys.stderr); return 1
    return 0
if __name__ == "__main__": raise SystemExit(main())
