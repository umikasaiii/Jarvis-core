from __future__ import annotations


def test_capabilities_shape(client):
    resp = client.get("/v1/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["chat"] is True
    assert data["streaming"] is True
    assert data["fastModel"] is True
    assert data["brainModel"] is True
    # Explicitly NOT implemented in this phase.
    for key in ("memory", "rag", "voice", "vision", "contextEngine", "actions"):
        assert data[key] is False
    assert data["protocolVersion"] == "1"
