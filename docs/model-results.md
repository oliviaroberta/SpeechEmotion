# ML Results Summary

## Dataset and Split

RAVDESS speech recordings were split by actor to avoid speaker overlap: actors 01-16 were used for training, actors 17-20 for validation-based model selection, and actors 21-24 were reserved for one final held-out evaluation.

## Preprocessing and Features

Recordings were converted to mono by arithmetic stereo averaging where needed, resampled to 16 kHz with `soxr_hq`, silence-trimmed at `top_db=40`, and padded or maximum-energy cropped to 56,000 samples (3.5 seconds). Log-Mel features used 64 Mel bands and 219 frames. Per-Mel-band normalization statistics were fitted on training data only.

## Model Selection

The following validation-only alternatives were considered: baseline SVM, tuned SVM, original CNN, regularized/augmented CNN, and frozen Wav2Vec2 transfer classifier. The reduced-regularization/no-dropout CNN was selected before the held-out test split was opened.

Selected-model validation performance on actors 17-20 was 55.4167% accuracy, 56.6406% balanced accuracy, and 0.536975 macro F1.

## Final Held-Out Test Result

The frozen selected CNN was evaluated once on actors 21-24. It achieved 46.6667% accuracy, 47.2656% balanced accuracy, 0.451797 macro precision, 0.472656 macro recall, and 0.419553 macro F1. Weighted F1 was 0.421437.

The strongest emotion-level result was surprised (F1 0.800000). Angry and calm were also comparatively reliable (F1 0.586957 and 0.586207). Sad (F1 0.105263) and happy (F1 0.156863) were the weakest. The test result is lower than the validation result, indicating limited generalization to the held-out speakers.

No model retraining, tuning, threshold adjustment, model selection, or alternative-model test evaluation was performed after opening the final test split.
