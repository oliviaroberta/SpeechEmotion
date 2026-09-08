from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cnn_model import EXPECTED_CLASSES, EXPECTED_INPUT_SHAPE, SpectrogramMasking, build_cnn_baseline, build_cnn_reduced_regularization, build_cnn_regularized_augmented, load_cnn_config


class CnnBaselineTests(unittest.TestCase):
    def test_configuration_and_shapes(self) -> None:
        config = load_cnn_config()
        self.assertEqual(tuple(config["input_shape"]), EXPECTED_INPUT_SHAPE)
        self.assertEqual(tuple(config["classes"]), EXPECTED_CLASSES)
        model = build_cnn_baseline(config)
        self.assertEqual(model.input_shape, (None, 64, 219, 1))
        self.assertEqual(model.output_shape, (None, 8))
        self.assertGreater(model.count_params(), 0)
        self.assertTrue(any(variable.trainable for variable in model.trainable_variables))

    def test_probabilities_and_deterministic_initial_weights(self) -> None:
        synthetic_input = np.full((2, 64, 219, 1), 0.125, dtype=np.float32)
        first_model = build_cnn_baseline()
        second_model = build_cnn_baseline()
        for first, second in zip(first_model.get_weights(), second_model.get_weights()):
            np.testing.assert_array_equal(first, second)
        probabilities = first_model(synthetic_input, training=False).numpy()
        self.assertEqual(probabilities.shape, (2, 8))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), rtol=0, atol=1e-6)

    def test_synthetic_training_batch(self) -> None:
        generator = np.random.default_rng(42)
        synthetic_input = generator.normal(size=(2, 64, 219, 1)).astype(np.float32)
        labels = np.array([0, 7], dtype=np.int32)
        result = build_cnn_baseline().train_on_batch(synthetic_input, labels, return_dict=True)
        self.assertIn("loss", result)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))

    def test_reduced_regularization_variant(self) -> None:
        model = build_cnn_reduced_regularization()
        self.assertEqual(model.input_shape, (None, 64, 219, 1))
        self.assertEqual(model.output_shape, (None, 8))
        self.assertEqual(model.count_params(), 102344)
        self.assertTrue(all(layer.rate == 0.0 for layer in model.layers if layer.__class__.__name__ == "Dropout"))

    def test_training_only_deterministic_spectrogram_masking(self) -> None:
        inputs = np.ones((1, 64, 219, 1), dtype=np.float32)
        layer = SpectrogramMasking()
        training_first = layer(inputs, training=True).numpy()
        training_second = layer(inputs, training=True).numpy()
        inference = layer(inputs, training=False).numpy()
        self.assertEqual(training_first.shape, inputs.shape)
        self.assertTrue(np.isfinite(training_first).all())
        np.testing.assert_array_equal(training_first, training_second)
        np.testing.assert_array_equal(inference, inputs)
        self.assertLess(np.count_nonzero(training_first), inputs.size)

    def test_augmented_variant_save_and_reload(self) -> None:
        inputs = np.full((1, 64, 219, 1), 0.25, dtype=np.float32)
        model = build_cnn_regularized_augmented()
        self.assertEqual(model.count_params(), 102344)
        before = model(inputs, training=False).numpy()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "temporary.keras"
            model.save(path)
            import tensorflow as tf
            after = tf.keras.models.load_model(path)(inputs, training=False).numpy()
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
