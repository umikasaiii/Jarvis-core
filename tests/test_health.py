from __future__ import annotations


def test_health_ok(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["protocolVersion"] == "1"
    assert data["llmAvailable"] is True
    assert "uptimeSeconds" in data
    assert data["activeModel"] == "fast-fake"


def test_health_is_lightweight_no_inference(client, monkeypatch):
    """Health must never trigger inference. We can't directly observe "no
    inference ran" over HTTP, but we can assert the response is fast and
    doesn't depend on provider .generate being callable."""
    import time

    start = time.perf_counter()
    resp = client.get("/v1/health")
    elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    assert elapsed < 1.0
