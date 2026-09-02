from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from core.config import get_settings


def test_server_starts_without_reachable_model(monkeypatch):
    """The server must come up even if the configured backend is completely
    unreachable — it should report degraded status via /v1/models instead
    of crashing on startup."""
    get_settings.cache_clear()
    monkeypatch.setenv("FAST_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("FAST_BASE_URL", "http://127.0.0.1:1")  # nothing listens here
    monkeypatch.setenv("BRAIN_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("BRAIN_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("LOG_JSON", "false")

    app = create_app()
    with TestClient(app) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200  # health never fails

        models = client.get("/v1/models")
        assert models.status_code == 200
        data = models.json()
        assert all(m["available"] is False for m in data["models"])

    get_settings.cache_clear()


def test_graceful_shutdown_completes(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("FAST_MODEL_BACKEND", "fake")
    monkeypatch.setenv("BRAIN_MODEL_BACKEND", "fake")
    monkeypatch.setenv("LOG_JSON", "false")

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
    # Exiting the `with` block runs the lifespan shutdown path; reaching
    # here without hanging/raising is the assertion.
    get_settings.cache_clear()
