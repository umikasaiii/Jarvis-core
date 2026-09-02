from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from ai.model_manager import ModelManager
from ai.orchestrator import RequestOrchestrator
from ai.queue import InferenceQueue
from ai.router import CoreAiRouter
from app.main import create_app
from conversations.store import InMemoryConversationStore
from core.config import Settings, get_settings
from providers.fake import FakeLlmProvider


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(monkeypatch) -> Settings:
    # Defaults already point at the "fake" backend, so tests never need a
    # real multi-GB model. Individual tests can monkeypatch env vars before
    # calling this fixture-dependent app fixture for different scenarios.
    monkeypatch.setenv("FAST_MODEL_BACKEND", "fake")
    monkeypatch.setenv("BRAIN_MODEL_BACKEND", "fake")
    monkeypatch.setenv("LOG_JSON", "false")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "2")
    monkeypatch.setenv("MAX_QUEUE_SIZE", "4")
    monkeypatch.setenv("REQUEST_TIMEOUT", "5")
    return get_settings()


@pytest.fixture
def app(settings):
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def orchestrator_fixture(settings):
    """A bare orchestrator wired to controllable FakeLlmProvider instances,
    for tests that want direct access to the providers (to flip `healthy`,
    inspect `.calls`, etc.) without going through HTTP."""
    from api.schemas.common import ExecutionTarget

    fast = FakeLlmProvider(name="fast-fake", token_delay_seconds=0.0)
    brain = FakeLlmProvider(name="brain-fake", token_delay_seconds=0.0)

    model_manager = ModelManager(settings)
    model_manager._providers[ExecutionTarget.FAST] = fast
    model_manager._providers[ExecutionTarget.BRAIN] = brain

    router = CoreAiRouter(settings)
    queue = InferenceQueue(max_concurrent=settings.max_concurrent_requests, max_queue_size=settings.max_queue_size)
    store = InMemoryConversationStore(
        max_messages=settings.max_conversation_messages, max_chars=settings.max_context_chars
    )
    orchestrator = RequestOrchestrator(
        settings=settings,
        model_manager=model_manager,
        router=router,
        queue=queue,
        conversation_store=store,
        system_prompt="You are JARVIS (test prompt).",
    )
    return orchestrator, fast, brain, store
