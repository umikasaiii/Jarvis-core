"""§ FASE 2A: per-model Ollama `think` control.

`_build_prompt()` is a pure, synchronous method — most of this file tests it
directly with no network involved. The ModelManager tests below only build
providers (never call `load_model`/`generate`), so they never touch the
network either.
"""
from __future__ import annotations

import pytest

from api.schemas.common import ExecutionTarget
from ai.model_manager import ModelManager, build_provider
from core.config import get_settings
from providers.fake import FakeLlmProvider
from providers.llamacpp import LlamaCppProvider
from providers.ollama import OllamaProvider


# --- OllamaProvider._build_prompt --------------------------------------------------


def test_think_false_is_sent_as_false_top_level():
    provider = OllamaProvider(name="qwen3.5:0.8b", think=False)
    payload = provider._build_prompt("ciao", None)
    assert payload["think"] is False


def test_think_true_is_sent_as_true_top_level():
    provider = OllamaProvider(name="qwen3.5:0.8b", think=True)
    payload = provider._build_prompt("ciao", None)
    assert payload["think"] is True


def test_think_defaults_to_false_when_not_specified():
    provider = OllamaProvider(name="qwen3.5:0.8b")
    payload = provider._build_prompt("ciao", None)
    assert payload["think"] is False


def test_think_is_top_level_not_nested_in_options():
    provider = OllamaProvider(name="qwen3.5:0.8b", think=True)
    payload = provider._build_prompt("ciao", None)
    assert "think" not in payload["options"]
    assert payload["think"] is True


def test_system_prompt_still_goes_into_the_system_field():
    provider = OllamaProvider(name="qwen3.5:0.8b", think=True)
    payload = provider._build_prompt("ciao", "Sei JARVIS.")
    assert payload["system"] == "Sei JARVIS."
    # think must not leak into / replace the system field or vice versa
    assert payload["think"] is True


def test_no_system_prompt_omits_the_system_field():
    provider = OllamaProvider(name="qwen3.5:0.8b", think=False)
    payload = provider._build_prompt("ciao", None)
    assert "system" not in payload


def test_num_ctx_stays_in_options():
    provider = OllamaProvider(name="qwen3.5:0.8b", context_size=4096, think=True)
    payload = provider._build_prompt("ciao", None)
    assert payload["options"] == {"num_ctx": 4096}


def test_build_prompt_shape_matches_the_requested_contract():
    provider = OllamaProvider(name="qwen3.5:0.8b", context_size=4096, think=False)
    payload = provider._build_prompt("ciao", None)
    assert payload == {
        "model": "qwen3.5:0.8b",
        "prompt": "ciao",
        "think": False,
        "options": {"num_ctx": 4096},
    }


# --- generate()/stream() automatically reuse _build_prompt's think ----------------


@pytest.mark.asyncio
async def test_generate_sends_think_via_httpx(monkeypatch):
    provider = OllamaProvider(name="qwen3.5:0.8b", think=True)
    seen_payload = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "ok", "done": True, "eval_count": 1}

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, path, json):
            seen_payload.update(json)
            return FakeResponse()

    monkeypatch.setattr(provider, "_client", lambda: FakeAsyncClient())

    await provider.generate("ciao")

    assert seen_payload["think"] is True


@pytest.mark.asyncio
async def test_load_model_warmup_respects_think_false(monkeypatch):
    """§ richiesta esplicita: il warmup non deve poter attivare il thinking
    di nascosto anche quando il resto della configurazione lo lascia spento."""
    provider = OllamaProvider(name="qwen3.5:0.8b", think=False)
    seen_payload = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, path, json):
            seen_payload.update(json)
            return FakeResponse()

    monkeypatch.setattr(provider, "_client", lambda: FakeAsyncClient())

    await provider.load_model()

    assert seen_payload["think"] is False
    assert provider._loaded is True


@pytest.mark.asyncio
async def test_load_model_warmup_respects_think_true(monkeypatch):
    provider = OllamaProvider(name="qwen3.5:0.8b", think=True)
    seen_payload = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, path, json):
            seen_payload.update(json)
            return FakeResponse()

    monkeypatch.setattr(provider, "_client", lambda: FakeAsyncClient())

    await provider.load_model()

    assert seen_payload["think"] is True


# --- ai.model_manager.build_provider / ModelManager --------------------------------


def test_build_provider_forwards_think_only_to_ollama():
    ollama = build_provider(
        "ollama", "qwen3.5:0.8b", base_url="http://127.0.0.1:11434", context_size=4096, request_timeout=30.0, think=True
    )
    assert isinstance(ollama, OllamaProvider)
    assert ollama.think is True


def test_build_provider_fake_backend_ignores_think_and_still_works():
    fake = build_provider(
        "fake", "fast-fake", base_url="", context_size=4096, request_timeout=30.0, think=True
    )
    assert isinstance(fake, FakeLlmProvider)
    assert not hasattr(fake, "think")


def test_build_provider_llamacpp_backend_ignores_think_and_still_works():
    llama = build_provider(
        "llamacpp", "some-model", base_url="http://127.0.0.1:8080", context_size=4096, request_timeout=30.0, think=True
    )
    assert isinstance(llama, LlamaCppProvider)
    assert not hasattr(llama, "think")


def test_fast_and_brain_receive_independent_think_settings(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("FAST_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("BRAIN_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("FAST_THINK", "true")
    monkeypatch.setenv("BRAIN_THINK", "false")
    settings = get_settings()

    manager = ModelManager(settings)

    fast = manager.get_provider(ExecutionTarget.FAST)
    brain = manager.get_provider(ExecutionTarget.BRAIN)
    assert isinstance(fast, OllamaProvider) and isinstance(brain, OllamaProvider)
    assert fast.think is True
    assert brain.think is False
    get_settings.cache_clear()


def test_fast_and_brain_think_independent_the_other_way_round(monkeypatch):
    """Same assertion, opposite values - proves the two settings are never swapped/shared."""
    get_settings.cache_clear()
    monkeypatch.setenv("FAST_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("BRAIN_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("FAST_THINK", "false")
    monkeypatch.setenv("BRAIN_THINK", "true")
    settings = get_settings()

    manager = ModelManager(settings)

    assert manager.get_provider(ExecutionTarget.FAST).think is False
    assert manager.get_provider(ExecutionTarget.BRAIN).think is True
    get_settings.cache_clear()


def test_default_think_is_false_for_both_when_unset(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("FAST_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("BRAIN_MODEL_BACKEND", "ollama")
    monkeypatch.delenv("FAST_THINK", raising=False)
    monkeypatch.delenv("BRAIN_THINK", raising=False)
    settings = get_settings()

    manager = ModelManager(settings)

    assert manager.get_provider(ExecutionTarget.FAST).think is False
    assert manager.get_provider(ExecutionTarget.BRAIN).think is False
    get_settings.cache_clear()


def test_fake_backends_still_build_and_list_with_no_regression(monkeypatch):
    """§ richiesta esplicita: 'i backend fake continuano a funzionare senza regressioni'."""
    get_settings.cache_clear()
    monkeypatch.setenv("FAST_MODEL_BACKEND", "fake")
    monkeypatch.setenv("BRAIN_MODEL_BACKEND", "fake")
    settings = get_settings()

    manager = ModelManager(settings)

    assert isinstance(manager.get_provider(ExecutionTarget.FAST), FakeLlmProvider)
    assert isinstance(manager.get_provider(ExecutionTarget.BRAIN), FakeLlmProvider)
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_fake_backend_list_models_unaffected_by_think_settings(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("FAST_MODEL_BACKEND", "fake")
    monkeypatch.setenv("BRAIN_MODEL_BACKEND", "fake")
    settings = get_settings()

    manager = ModelManager(settings)
    infos = await manager.list_models()

    assert len(infos) == 2
    assert {info.extra["role"] for info in infos} == {"FAST", "BRAIN"}
    get_settings.cache_clear()
