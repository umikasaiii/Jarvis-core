from __future__ import annotations

import json


def test_stream_endpoint_sse_events(client):
    payload = {"protocolVersion": "1", "text": "stream me a reply"}
    events = []
    with client.stream("POST", "/v1/ai/stream", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        buffer = ""
        for chunk in resp.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                raw, buffer = buffer.split("\n\n", 1)
                if raw.startswith("data: "):
                    events.append(json.loads(raw.removeprefix("data: ")))

    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "done"
    assert any(e["type"] == "token" for e in events)
    assert events[-1]["modelUsed"] == "fast-fake"


def test_stream_endpoint_invalid_protocol_version(client):
    payload = {"protocolVersion": "99", "text": "hello"}
    resp = client.post("/v1/ai/stream", json=payload)
    assert resp.status_code == 400


def test_stream_endpoint_invalid_request_body(client):
    resp = client.post("/v1/ai/stream", json={"protocolVersion": "1"})
    assert resp.status_code == 422
