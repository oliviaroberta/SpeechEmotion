from __future__ import annotations

import unittest

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wav2vec2_transfer import EXPECTED_LABELS, load_transfer_config, pool_hidden_states, validate_waveform


class Wav2Vec2TransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_transfer_config()

    def test_configuration_and_label_order(self) -> None:
        self.assertEqual(tuple(self.config["emotion_labels"]), EXPECTED_LABELS)
        self.assertEqual(self.config["embedding"]["shape"], [1536])

    def test_pooling_shape_values_and_determinism(self) -> None:
        states = np.arange(5 * 768, dtype=np.float32).reshape(5, 768) / 100.0
        first = pool_hidden_states(states, self.config)
        second = pool_hidden_states(states, self.config)
        self.assertEqual(first.shape, (1536,))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(first.flags.c_contiguous)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)
        expected_mean = states.mean(axis=0, dtype=np.float64).astype(np.float32)
        expected_std = states.std(axis=0, dtype=np.float64, ddof=0).astype(np.float32)
        np.testing.assert_array_equal(first[:768], expected_mean)
        np.testing.assert_array_equal(first[768:], expected_std)

    def test_waveform_validation_and_rejections(self) -> None:
        waveform = np.linspace(-1.1, 1.1, 56000, dtype=np.float32)
        checked = validate_waveform(waveform, self.config)
        self.assertEqual(checked.dtype, np.float32)
        np.testing.assert_array_equal(checked, waveform)
        for invalid in (waveform[:-1], np.full(56000, np.nan), np.full((56000, 1), 0.0)):
            with self.assertRaises(ValueError):
                validate_waveform(invalid, self.config)

    def test_classifier_serialization_preserves_label_predictions(self) -> None:
        features = np.vstack((np.zeros((2, 1536)), np.ones((2, 1536)), np.full((2, 1536), 2.0)))
        labels = np.array([0, 0, 1, 1, 2, 2])
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42, solver="lbfgs"))])
        pipeline.fit(features, labels)
        bundle = {"pipeline": pipeline, "label_order": list(EXPECTED_LABELS)}
        serialized = joblib.dumps(bundle) if hasattr(joblib, "dumps") else None
        if serialized is None:
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bundle.joblib"
                joblib.dump(bundle, path)
                restored = joblib.load(path)
        else:
            restored = joblib.loads(serialized)
        np.testing.assert_array_equal(restored["pipeline"].predict(features), pipeline.predict(features))
        self.assertEqual(restored["label_order"], list(EXPECTED_LABELS))


if __name__ == "__main__":
    unittest.main()
