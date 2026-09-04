# RAVDESS Preprocessing Plan

## 1. Purpose

This document records a deterministic training-only preprocessing pilot. No preprocessing has been applied and no processed audio exists.

## 2. Data Used for the Pilot

The pilot used only training actors `01-16` and 960 recordings. Validation actors `17-20` and test actors `21-24` were excluded from all pilot measurements and recommendations. The source manifest SHA-256 was `079884EA3724C1ADAB1C53E33D9EB669B58FCC58E4F577379BBFF807A46C398D`. Restricting decisions to training voices avoids evaluation leakage.

## 3. Planned Processing Order

1. Load PCM audio as floating-point samples.
2. Convert stereo to mono by averaging channels.
3. Resample from 48 kHz to 16 kHz using `soxr_hq`.
4. Apply the selected silence-trimming rule.
5. Pad or truncate deterministically to the selected duration.
6. Verify output shape and numerical validity.

No processed files currently exist.

## 4. Silence-Trimming Analysis

All candidates used `frame_length=2048` and `hop_length=512`. Energy trimming is only an approximation of spoken-word boundaries.

| top_db | Mean retained seconds | No trim | Under 90% retained | Empty |
|---:|---:|---:|---:|---:|
| 20 | 1.714490 | 0 | 960 | 0 |
| 25 | 1.790231 | 0 | 960 | 0 |
| 30 | 1.885190 | 0 | 957 | 0 |
| 35 | 2.003113 | 2 | 940 | 0 |
| 40 | 2.158558 | 13 | 911 | 0 |

Higher thresholds retained more audio and were therefore less aggressive. All candidates avoided empty signals, but the pilot does not claim exact word boundaries.

## 5. Fixed-Duration Analysis

Padding/truncation counts after each trimming candidate were:

| top_db | 2.5 s pad/truncate | 3.0 s pad/truncate | 3.5 s pad/truncate | 4.0 s pad/truncate |
|---:|---:|---:|---:|---:|
| 20 | 931 / 29 | 959 / 1 | 960 / 0 | 960 / 0 |
| 25 | 908 / 52 | 957 / 3 | 960 / 0 | 960 / 0 |
| 30 | 874 / 86 | 943 / 17 | 958 / 2 | 960 / 0 |
| 35 | 816 / 144 | 915 / 45 | 947 / 13 | 958 / 2 |
| 40 | 734 / 226 | 866 / 94 | 928 / 32 | 955 / 5 |

Future short signals would be zero-padded deterministically with padding split between beginning and end. Long signals would be truncated deterministically; validation and testing would never use random crops.

## 6. Amplitude Analysis

Original peak values had minimum `0.006134`, maximum `0.999146`, mean `0.171578`, median `0.090469`, 5th percentile `0.017877`, and 95th percentile `0.679030`. Resampled peaks had maximum `1.052423`; 7 recordings exceeded the nominal floating-point range after resampling and will require a reviewed numerical policy if preprocessing is later applied. There were no silent files and 1 original file was potentially clipped at a peak of at least `0.999`.

Decoding PCM values into ordinary floating-point samples is not per-file amplitude normalization. RAVDESS was already peak-normalized in its official preparation, and emotional loudness/intensity can be useful information. The pilot therefore does not recommend additional per-file peak normalization.

## 7. Provisional Recommendation

These settings are provisional, not yet applied, and require review:

- Sample rate: 16,000 Hz using `soxr_hq`.
- Stereo-to-mono: arithmetic mean; all 3 training stereo files had bit-identical channels.
- Silence trimming: `top_db=40`, `frame_length=2048`, `hop_length=512`.
- Fixed duration: 3.5 seconds. It is the shortest candidate after `top_db=40` with truncation at or below 5%: 32 recordings (3.333333%) would be truncated.
- Padding: deterministic zero padding split between beginning and end.
- Truncation: deterministic, with no random crop.
- Per-file amplitude normalization: do not apply.

The trade-off is that 928 recordings would require padding at this setting, while a more aggressive trimming threshold could remove low-energy speech. No recommendation has been applied to raw audio.

## 8. Reproducibility

The pilot JSON is `ml/metadata/ravdess_preprocessing_pilot.json` with SHA-256 `CA37C89DEB292F25AD3ECE8FA134DD7C14C1B9B270412887A096BCF3AEABBA5A`. Both analyzer and validator runs succeeded twice, deterministic regeneration produced the same hash, and no raw or processed audio was committed to Git.

## 9. Next Approval Point

The next step will finalize the configuration and apply it reproducibly to separate processed copies while leaving raw audio unchanged.

## 10. Numerical Safety and Truncation Review

The training-only numerical review is recorded in `ml/metadata/ravdess_numerical_policy_review.json` with SHA-256 `93EB7F4230057C6DB345B183379A25B93C501B010E7324565BE72AA84CFE6865`. It repeated only actors `01-16`; validation and test recordings were excluded.

Seven resampled training signals had finite samples outside `[-1, 1]`. The review compares preserving in-memory floating-point values, hard clipping, and a single training-derived global safety gain. Because no PCM export is required and the overshoots are finite and rare, the provisional recommendation is to preserve float arrays and avoid per-file normalization. This does not apply to a future integer-PCM export, which would require an explicit bounded-output policy.

One source file is labelled potential clipping because its decoded peak is near full scale. Exact PCM full-scale counts are recorded; the review does not claim conclusive clipping from this evidence alone.

For the 32 trimmed signals longer than 56,000 samples, centre crop and maximum-energy-window crop were compared without writing audio. The provisional recommendation is the deterministic maximum-energy window because it selects the highest-energy valid window and resolves ties by choosing the earliest window. Proposed padding remains deterministic zero padding split across both ends, with an odd extra sample at the end. These recommendations have not been applied and no audio was written.
