from __future__ import annotations

import uuid


def _req(**overrides):
    payload = {
        "protocolVersion": "1",
        "text": "hello jarvis",
    }
    payload.update(overrides)
    return payload


def test_chat_basic_ok(client):
    resp = client.post("/v1/chat", json=_req())
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OK"
    assert data["modelUsed"] == "fast-fake"
    assert data["targetUsed"] == "FAST"
    assert "hello jarvis" in data["text"]


def test_chat_explicit_brain_target(client):
    resp = client.post("/v1/chat", json=_req(preferredTarget="BRAIN"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["targetUsed"] == "BRAIN"
    assert data["modelUsed"] == "brain-fake"


def test_chat_complex_request_type_routes_brain(client):
    resp = client.post("/v1/chat", json=_req(requestType="COMPLEX"))
    assert resp.status_code == 200
    assert resp.json()["targetUsed"] == "BRAIN"


def test_ai_request_endpoint(client):
    resp = client.post("/v1/ai/request", json=_req(requestType="COMMAND", text="turn on lights"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OK"
    assert data["targetUsed"] == "FAST"


def test_conversation_persists_across_requests(client):
    conv_id = str(uuid.uuid4())
    r1 = client.post("/v1/chat", json=_req(conversationId=conv_id, text="first message"))
    assert r1.status_code == 200
    r2 = client.post("/v1/chat", json=_req(conversationId=conv_id, text="second message"))
    assert r2.status_code == 200
    # The fake provider echoes back the prompt it received, which for the
    # second turn includes the built-up history.
    assert "first message" in r2.json()["text"]


def test_conversations_are_isolated_over_http(client):
    conv_a = str(uuid.uuid4())
    conv_b = str(uuid.uuid4())
    client.post("/v1/chat", json=_req(conversationId=conv_a, text="secret A"))
    r = client.post("/v1/chat", json=_req(conversationId=conv_b, text="message B"))
    assert "secret A" not in r.json()["text"]
