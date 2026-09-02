from __future__ import annotations


def test_models_endpoint_lists_fast_and_brain(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    roles = {m["role"] for m in data["models"]}
    assert roles == {"FAST", "BRAIN"}
    for m in data["models"]:
        assert m["available"] is True
        assert m["backend"] == "fake"
