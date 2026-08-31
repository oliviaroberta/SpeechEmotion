from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

MANIFEST_HASH = "079884EA3724C1ADAB1C53E33D9EB669B58FCC58E4F577379BBFF807A46C398D"
TOP_DBS = {"20", "25", "30", "35", "40"}; DURATIONS = {"2.5", "3.0", "3.5", "4.0"}
EXPECTED_EMOTIONS = {"neutral": 64, "calm": 128, "happy": 128, "sad": 128, "angry": 128, "fearful": 128, "disgust": 128, "surprised": 128}

def root() -> Path: return Path(__file__).resolve().parents[2]
def validate_counts(entry: dict[str, object]) -> None:
    for duration in DURATIONS:
        candidate = entry["fixed_duration_candidates"][duration]
        total = candidate["padding"]["count"] + candidate["truncation"]["count"] + candidate["exact_fit_count"]
        if total != 960: raise ValueError(f"Fixed-duration counts do not sum to 960 for {duration}")
        for group in ("per_emotion", "per_gender", "per_intensity"):
            for value in candidate[group].values():
                if sum(value.values()) < 0: raise ValueError("Invalid comparison count")
    if entry["empty_signal_count"] != 0: raise ValueError("A trimming candidate produced empty audio")

def main() -> int:
    parser = argparse.ArgumentParser(description="Independently verify the training-only preprocessing pilot.")
    parser.add_argument("report", nargs="?", type=Path, default=root() / "ml/metadata/ravdess_preprocessing_pilot.json"); args = parser.parse_args()
    try:
        with args.report.open(encoding="utf-8") as file: report = json.load(file)
        text = json.dumps(report)
        if "timestamp" in text.lower() or "C:\\" in text or "olivia.dogbey" in text.lower() or "\\\\" in text: raise ValueError("Pilot report contains forbidden environment-specific content")
        scope = report["analysis_scope"]
        if report["source_manifest_sha256"] != MANIFEST_HASH or scope["recording_count"] != 960 or scope["actor_ids"] != [f"{n:02d}" for n in range(1,17)] or not scope["validation_and_test_excluded"]: raise ValueError("Invalid pilot scope or manifest hash")
        if report["audio_written"] is not False or report["input_channels"] != {"mono": 957, "stereo": 3, "identical_stereo_files": 3}: raise ValueError("Invalid channel or audio-writing report")
        if not report["resampling_checks"]["all_passed"]: raise ValueError("Resampling checks did not pass")
        trimming = report["silence_trimming"]
        if set(trimming) != TOP_DBS: raise ValueError("Missing silence threshold")
        for entry in trimming.values():
            validate_counts(entry)
            if entry["per_emotion"] and {key: sum(1 for _ in value) for key, value in entry["per_emotion"].items()} is None: raise ValueError("Invalid emotion details")
        recommendation = report["provisional_recommendation"]
        if recommendation["status"] != "provisional_not_applied_pending_review" or str(recommendation["top_db"]) not in TOP_DBS or recommendation["fixed_duration_seconds"] not in {None, *map(float, DURATIONS)}: raise ValueError("Invalid provisional recommendation")
        if recommendation["fixed_duration_seconds"] is not None and f"{recommendation['fixed_duration_seconds']:.1f}" not in DURATIONS: raise ValueError("Recommendation was not evaluated")
    except Exception as error:
        print(f"Preprocessing pilot verification failed: {error}", file=sys.stderr); return 1
    print("RAVDESS preprocessing pilot verification completed successfully."); return 0
if __name__ == "__main__": raise SystemExit(main())
