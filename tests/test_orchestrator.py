from __future__ import annotations

import asyncio

import pytest

from api.schemas.common import ExecutionTarget, FinishReason, RequestType, ResponseStatus, StreamEventType
from api.schemas.request import JarvisRequest
from ai.orchestrator import ProtocolVersionError
from providers.base import LlmProviderError


def _req(**overrides) -> JarvisRequest:
    payload = dict(protocolVersion="1", text="hello there")
    payload.update(overrides)
    return JarvisRequest(**payload)


@pytest.mark.asyncio
async def test_fallback_to_other_target_when_preferred_unavailable(orchestrator_fixture):
    orchestrator, fast, brain, _store = orchestrator_fixture
    brain.healthy = False

    resp = await orchestrator.handle_request(
        _req(preferredTarget="BRAIN", allowFallback=True)
    )
    assert resp.status == ResponseStatus.OK
    assert resp.targetUsed == ExecutionTarget.FAST
    assert resp.warnings


@pytest.mark.asyncio
async def test_no_fallback_returns_error_when_disallowed(orchestrator_fixture):
    orchestrator, fast, brain, _store = orchestrator_fixture
    brain.healthy = False

    resp = await orchestrator.handle_request(
        _req(preferredTarget="BRAIN", allowFallback=False)
    )
    assert resp.status == ResponseStatus.ERROR
    assert resp.finishReason == FinishReason.ERROR


@pytest.mark.asyncio
async def test_both_models_unavailable_returns_error(orchestrator_fixture):
    orchestrator, fast, brain, _store = orchestrator_fixture
    fast.healthy = False
    brain.healthy = False

    resp = await orchestrator.handle_request(_req(preferredTarget="FAST"))
    assert resp.status == ResponseStatus.ERROR


@pytest.mark.asyncio
async def test_request_timeout(orchestrator_fixture):
    orchestrator, fast, brain, _store = orchestrator_fixture
    orchestrator.settings.request_timeout = 0.01
    fast.token_delay_seconds = 1.0

    async def slow_generate(*args, **kwargs):
        await asyncio.sleep(1.0)

    fast.generate = slow_generate  # type: ignore[method-assign]

    resp = await orchestrator.handle_request(_req(preferredTarget="FAST"))
    assert resp.status == ResponseStatus.ERROR
    assert resp.finishReason == FinishReason.TIMEOUT


@pytest.mark.asyncio
async def test_protocol_version_mismatch_raises(orchestrator_fixture):
    orchestrator, fast, brain, _store = orchestrator_fixture
    with pytest.raises(ProtocolVersionError):
        await orchestrator.handle_request(_req(protocolVersion="2"))


@pytest.mark.asyncio
async def test_streaming_emits_start_tokens_done(orchestrator_fixture):
    orchestrator, fast, brain, _store = orchestrator_fixture
    events = [e async for e in orchestrator.handle_stream(_req(text="stream this please"))]
    assert events[0].type == StreamEventType.START
    assert events[-1].type == StreamEventType.DONE
    token_events = [e for e in events if e.type == StreamEventType.TOKEN]
    assert len(token_events) > 0
    assert fast.calls  # provider was actually invoked
    full_text = "".join(e.content for e in token_events)
    assert "stream this please" in full_text


@pytest.mark.asyncio
async def test_streaming_error_event_when_provider_unavailable(orchestrator_fixture):
    orchestrator, fast, brain, _store = orchestrator_fixture
    fast.healthy = False
    brain.healthy = False
    events = [e async for e in orchestrator.handle_stream(_req(preferredTarget="FAST", allowFallback=False))]
    assert events[-1].type == StreamEventType.ERROR


@pytest.mark.asyncio
async def test_system_prompt_override_reaches_provider(orchestrator_fixture):
    """v1.1.0: a client-supplied systemPrompt must reach the provider
    verbatim, instead of the orchestrator's own default - this is the exact
    gap that blocked jarvis-android's Conversational engine (structured
    JSON tool-calling) from ever using JARVIS Core."""
    orchestrator, fast, _brain, _store = orchestrator_fixture

    resp = await orchestrator.handle_request(
        _req(preferredTarget="FAST", systemPrompt="RESPOND ONLY IN JSON: {...}")
    )

    assert resp.status == ResponseStatus.OK
    assert fast.system_prompt_calls[-1] == "RESPOND ONLY IN JSON: {...}"


@pytest.mark.asyncio
async def test_absent_system_prompt_falls_back_to_server_default(orchestrator_fixture):
    """Every existing client that never sends systemPrompt must observe
    byte-for-byte the same behavior as before this field existed."""
    orchestrator, fast, _brain, _store = orchestrator_fixture

    await orchestrator.handle_request(_req(preferredTarget="FAST"))

    assert fast.system_prompt_calls[-1] == "You are JARVIS (test prompt)."


@pytest.mark.asyncio
async def test_blank_system_prompt_also_falls_back_to_server_default(orchestrator_fixture):
    orchestrator, fast, _brain, _store = orchestrator_fixture

    await orchestrator.handle_request(_req(preferredTarget="FAST", systemPrompt="   "))

    assert fast.system_prompt_calls[-1] == "You are JARVIS (test prompt)."


@pytest.mark.asyncio
async def test_system_prompt_override_reaches_provider_when_streaming(orchestrator_fixture):
    orchestrator, fast, _brain, _store = orchestrator_fixture

    events = [
        e async for e in orchestrator.handle_stream(
            _req(preferredTarget="FAST", text="stream this", systemPrompt="STREAM PERSONA")
        )
    ]

    assert events[-1].type == StreamEventType.DONE
    assert fast.system_prompt_calls[-1] == "STREAM PERSONA"


@pytest.mark.asyncio
async def test_streaming_cancellation_stops_cleanly(orchestrator_fixture):
    orchestrator, fast, brain, _store = orchestrator_fixture
    fast.token_delay_seconds = 0.05

    collected = []

    async def consume():
        async for event in orchestrator.handle_stream(_req(text="a long request " * 20)):
            collected.append(event)
            if len(collected) == 2:
                raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await consume()
    # No exception escaped beyond CancelledError, and some events were seen.
    assert len(collected) == 2
