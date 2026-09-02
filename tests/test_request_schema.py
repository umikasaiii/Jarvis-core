from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas.common import ExecutionTarget, RequestType
from api.schemas.request import JarvisRequest


def test_minimal_valid_request():
    req = JarvisRequest(protocolVersion="1", text="hello")
    assert req.requestType == RequestType.CHAT
    assert req.preferredTarget == ExecutionTarget.AUTO
    assert req.allowFallback is True
    assert req.context is None
    assert req.requestId  # auto-generated


def test_blank_text_rejected():
    with pytest.raises(ValidationError):
        JarvisRequest(protocolVersion="1", text="   ")


def test_missing_text_rejected():
    with pytest.raises(ValidationError):
        JarvisRequest(protocolVersion="1")


def test_invalid_request_type_rejected():
    with pytest.raises(ValidationError):
        JarvisRequest(protocolVersion="1", text="hi", requestType="NOT_A_TYPE")


def test_context_and_metadata_optional_and_freeform():
    req = JarvisRequest(
        protocolVersion="1",
        text="hi",
        context={"battery": 80},
        metadata={"clientVersion": "1.2.3"},
    )
    assert req.context == {"battery": 80}
    assert req.metadata == {"clientVersion": "1.2.3"}


def test_invalid_request_http_422(client):
    resp = client.post("/v1/chat", json={"protocolVersion": "1"})
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_request"


def test_unsupported_protocol_version_http_400(client):
    resp = client.post(
        "/v1/chat", json={"protocolVersion": "99", "text": "hello there"}
    )
    assert resp.status_code == 400
