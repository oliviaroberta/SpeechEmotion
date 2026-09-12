from __future__ import annotations

import importlib
import os
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


class HealthEndpointTests(unittest.TestCase):
    def test_health_response_and_openapi_schema(self) -> None:
        from backend.app.main import API_TITLE, API_VERSION, create_app

        with TestClient(create_app()) as client:
            response = client.get("/api/v1/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "healthy", "service": "speech-emotion-recognition-api", "version": "0.1.0"})
            schema = client.get("/openapi.json")
        self.assertEqual(schema.status_code, 200)
        self.assertEqual(schema.json()["info"], {"title": API_TITLE, "version": API_VERSION})
        self.assertIn("/api/v1/health", schema.json()["paths"])

    def test_cors_uses_configured_allowlist(self) -> None:
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "https://speech-emotion-six.vercel.app, http://localhost:5173"}, clear=False):
            from backend.app import main

            application = main.create_app()
        with TestClient(application) as client:
            response = client.options(
                "/api/v1/predict",
                headers={
                    "Origin": "https://speech-emotion-six.vercel.app",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://speech-emotion-six.vercel.app")
        self.assertIn("POST", response.headers["access-control-allow-methods"])
        self.assertIn("content-type", response.headers["access-control-allow-headers"])

    def test_production_cors_requires_explicit_valid_origins(self) -> None:
        with patch.dict(os.environ, {"SER_ENVIRONMENT": "production"}, clear=False):
            from backend.app import main

            self.assertEqual(main.get_allowed_origins(), [])
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "https://frontend.example.com"}, clear=False):
            self.assertEqual(main.get_allowed_origins(), ["https://frontend.example.com"])
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "*"}, clear=False):
            with self.assertRaises(ValueError):
                main.get_allowed_origins()

    def test_health_import_path_does_not_load_prediction_model(self) -> None:
        sys.modules.pop("predict_audio", None)
        sys.modules.pop("ml.src.predict_audio", None)
        module = importlib.import_module("backend.app.main")
        with TestClient(module.create_app()) as client:
            self.assertEqual(client.get("/api/v1/health").status_code, 200)
        self.assertNotIn("predict_audio", sys.modules)
        self.assertNotIn("ml.src.predict_audio", sys.modules)
        self.assertNotIn("tensorflow", module.__dict__)

    def test_vercel_entrypoint_exports_the_fastapi_application(self) -> None:
        from api.index import app
        from backend.app.main import app as backend_app

        self.assertIs(app, backend_app)


if __name__ == "__main__":
    unittest.main()
