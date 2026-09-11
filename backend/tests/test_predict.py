from __future__ import annotations

import io
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import DEFAULT_MAX_AUDIO_BYTES, create_app, get_max_audio_bytes, get_prediction_function
from backend.app.prediction_service import (
    PredictionArtifactsUnavailableError,
    PredictionInputUnavailableError,
)


EXPECTED_RESULT = {
    "emotion": "happy",
    "class_index": 2,
    "confidence": 0.72,
    "probabilities": {
        "neutral": 0.01, "calm": 0.02, "happy": 0.72, "sad": 0.03,
        "angry": 0.04, "fearful": 0.05, "disgust": 0.06, "surprised": 0.07,
    },
    "model": {"identifier": "cnn_reduced_regularization_no_dropout", "version": "1.0.0"},
}


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16000)
        recording.writeframes(b"\x00\x00" * 32)
    return buffer.getvalue()


class PredictionEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.application = create_app()
        self.client = TestClient(self.application)
        self.seen_paths: list[Path] = []

        def fake_predictor(path: Path) -> dict[str, object]:
            self.seen_paths.append(path)
            self.assertTrue(path.is_file())
            return EXPECTED_RESULT

        self.application.dependency_overrides[get_prediction_function] = lambda: fake_predictor

    def tearDown(self) -> None:
        self.application.dependency_overrides.clear()
        self.client.close()

    def post_wav(self, content: bytes) -> object:
        return self.client.post("/api/v1/predict", files={"file": ("untrusted-name.wav", content, "audio/wav")})

    def test_successful_prediction_matches_public_schema_and_cleans_up(self) -> None:
        response = self.post_wav(wav_bytes())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), EXPECTED_RESULT)
        self.assertEqual(len(self.seen_paths), 1)
        self.assertFalse(self.seen_paths[0].exists())

    def test_missing_empty_and_unsupported_uploads_are_rejected(self) -> None:
        self.assertEqual(self.client.post("/api/v1/predict").status_code, 400)
        self.assertEqual(self.post_wav(b"").status_code, 400)
        self.assertEqual(self.post_wav(b"not a WAV file").status_code, 415)

    def test_malformed_wav_and_oversized_upload_are_rejected(self) -> None:
        self.assertEqual(DEFAULT_MAX_AUDIO_BYTES, 4 * 1024 * 1024)
        self.assertEqual(get_max_audio_bytes(), 4 * 1024 * 1024)
        self.assertEqual(self.post_wav(b"RIFF\x00\x00\x00\x00WAVEbroken").status_code, 400)
        self.assertEqual(self.post_wav(wav_bytes() + (b"\x00" * DEFAULT_MAX_AUDIO_BYTES)).status_code, 413)
        with patch.dict(os.environ, {"SER_MAX_AUDIO_BYTES": "20"}, clear=False):
            self.assertEqual(self.post_wav(wav_bytes()).status_code, 413)

    def test_predictor_failures_are_safe_and_temporary_files_are_removed(self) -> None:
        def invalid_audio(path: Path) -> dict[str, object]:
            self.seen_paths.append(path)
            raise PredictionInputUnavailableError("internal path must not escape")

        self.application.dependency_overrides[get_prediction_function] = lambda: invalid_audio
        response = self.post_wav(wav_bytes())
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("internal path", response.text)
        self.assertFalse(self.seen_paths[-1].exists())

        self.application.dependency_overrides[get_prediction_function] = lambda: (lambda path: (_ for _ in ()).throw(PredictionArtifactsUnavailableError("missing")))
        self.assertEqual(self.post_wav(wav_bytes()).status_code, 503)

        def unexpected(path: Path) -> dict[str, object]:
            self.seen_paths.append(path)
            raise RuntimeError("unexpected internal failure")

        self.application.dependency_overrides[get_prediction_function] = lambda: unexpected
        response = self.post_wav(wav_bytes())
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("unexpected internal failure", response.text)
        self.assertFalse(self.seen_paths[-1].exists())

    def test_openapi_and_health_endpoint_remain_available(self) -> None:
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/api/v1/predict", schema["paths"])
        self.assertIn("multipart/form-data", schema["paths"]["/api/v1/predict"]["post"]["requestBody"]["content"])
        self.assertEqual(self.client.get("/api/v1/health").json(), {"status": "healthy", "service": "speech-emotion-recognition-api", "version": "0.1.0"})


if __name__ == "__main__":
    unittest.main()
