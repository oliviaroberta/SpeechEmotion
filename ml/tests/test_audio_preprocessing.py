from __future__ import annotations

import unittest

import numpy as np

from audio_preprocessing import load_preprocessing_config, preprocess_waveform


class AudioPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_preprocessing_config()

    def waveform(self, length: int, value: float = 0.2) -> np.ndarray:
        return np.full(length, value, dtype=np.float32)

    def test_configuration(self) -> None:
        self.assertEqual(self.config["target_length_samples"], 56000)
        self.assertEqual(self.config["resample_method"], "soxr_hq")

    def test_mono_stereo_and_16khz(self) -> None:
        mono = self.waveform(56000)
        output, audit = preprocess_waveform(mono, 16000, self.config)
        self.assertEqual(audit["source_channels"], 1)
        self.assertEqual(audit["resampled_length"], 56000)
        stereo = np.column_stack((mono, mono * 3))
        output, audit = preprocess_waveform(stereo, 16000, self.config)
        self.assertEqual(audit["source_channels"], 2)
        self.assertAlmostEqual(float(output[28000]), 0.4, places=6)

    def test_padding_and_exact_length(self) -> None:
        output, audit = preprocess_waveform(self.waveform(55999), 16000, self.config)
        self.assertEqual(audit["final_action"], "padded")
        self.assertEqual((audit["padding_before"], audit["padding_after"]), (0, 1))
        output, audit = preprocess_waveform(self.waveform(56000), 16000, self.config)
        self.assertEqual(audit["final_action"], "unchanged")
        self.assertEqual(output.shape, (56000,))

    def test_energy_crop_and_tie(self) -> None:
        long = self.waveform(57000, 0.01); long[1000:57000] = 0.5
        _, audit = preprocess_waveform(long, 16000, self.config)
        self.assertEqual(audit["final_action"], "truncated")
        tie = self.waveform(57000)
        _, audit = preprocess_waveform(tie, 16000, self.config)
        self.assertEqual(audit["crop_start"], 0)

    def test_output_and_overshoot_repeatability(self) -> None:
        source = self.waveform(56000, 1.05)
        first, _ = preprocess_waveform(source, 16000, self.config)
        second, _ = preprocess_waveform(source, 16000, self.config)
        self.assertTrue(first.flags.c_contiguous)
        self.assertEqual(first.dtype, np.float32)
        self.assertGreater(float(first.max()), 1.0)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_rejections(self) -> None:
        for source, rate in ((np.array([], dtype=np.float32), 16000), (np.array([np.nan]), 16000), (np.array([np.inf]), 16000), (self.waveform(10), 0), (np.ones((10, 3)), 16000)):
            with self.assertRaises(ValueError):
                preprocess_waveform(source, rate, self.config)


if __name__ == "__main__":
    unittest.main()
