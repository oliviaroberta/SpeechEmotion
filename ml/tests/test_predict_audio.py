from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from predict_audio import (
    EXPECTED_CLASSES,
    PredictionArtifactError,
    PredictionInputError,
    PredictionIntegrityError,
    _load_model,
    _load_normalization,
    build_prediction_result,
    clear_prediction_resource_cache,
    predict_audio_file,
    prepare_audio_file,
)


class SingleAudioPredictionTests(unittest.TestCase):
    development_audio = Path("ml/data/raw/ravdess/extracted/Actor_01/03-01-05-01-01-01-01.wav")

    def tearDown(self) -> None:
        clear_prediction_resource_cache()

    def test_fixed_label_order_and_result_schema(self) -> None:
        probabilities = np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.72]], dtype=np.float32)
        result = build_prediction_result(probabilities, "1.0.0")
        self.assertEqual(tuple(EXPECTED_CLASSES), ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"))
        self.assertEqual(set(result), {"emotion", "class_index", "confidence", "probabilities", "model"})
        self.assertEqual(result["emotion"], "surprised")
        self.assertEqual(result["class_index"], 7)
        self.assertEqual(result["confidence"], result["probabilities"]["surprised"])
        self.assertEqual(list(result["probabilities"]), list(EXPECTED_CLASSES))

    def test_invalid_probability_output_is_rejected(self) -> None:
        with self.assertRaises(PredictionIntegrityError):
            build_prediction_result(np.full((1, 7), 1 / 7, dtype=np.float32), "1.0.0")
        with self.assertRaises(PredictionIntegrityError):
            build_prediction_result(np.full((1, 8), np.nan, dtype=np.float32), "1.0.0")

    def test_missing_input_and_artifacts_are_reported(self) -> None:
        with self.assertRaises(PredictionInputError):
            predict_audio_file("missing-recording.wav")
        missing = Path("missing-artifact")
        with self.assertRaises(PredictionArtifactError):
            _load_normalization(missing)
        with self.assertRaises(PredictionArtifactError):
            _load_model(missing)

    def test_malformed_audio_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-a-recording.wav"
            path.write_bytes(b"not a WAV file")
            with self.assertRaises(PredictionInputError):
                predict_audio_file(path)

    def test_development_audio_feature_shape_and_deterministic_prediction(self) -> None:
        self.assertTrue(self.development_audio.is_file())
        first_feature, _ = prepare_audio_file(self.development_audio)
        self.assertEqual(first_feature.shape, (1, 64, 219, 1))
        self.assertEqual(first_feature.dtype, np.float32)
        self.assertTrue(first_feature.flags.c_contiguous)
        self.assertTrue(np.isfinite(first_feature).all())
        first = predict_audio_file(self.development_audio)
        second = predict_audio_file(self.development_audio)
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(first["probabilities"].values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
