"""§ FASE 2A: per-model Ollama `think` control.
§ FASE 2A.1: latency-profiling diagnostics (measurement only, no behaviour
change) added below the FASE 2A tests.

`_build_prompt()` is a pure, synchronous method — most of this file tests it
directly with no network involved. The ModelManager tests below only build
providers (never call `load_model`/`generate`), so they never touch the
network either.
"""
from __future__ import annotations

import logging

import pytest

from api.schemas.common import ExecutionTarget
from ai.model_manager import ModelManager, build_provider
from core.config import get_settings
from providers.fake import FakeLlmProvider
from providers.llamacpp import LlamaCppProvider
from providers.ollama import OllamaProvider, ns_to_ms


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

        def json(self):
            return {}

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

        def json(self):
            return {}

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


# --- § FASE 2A.1: ns_to_ms ----------------------------------------------------------


def test_ns_to_ms_converts_nanoseconds_to_milliseconds():
    assert ns_to_ms(1_000_000) == 1.0
    assert ns_to_ms(72_490_000_000) == 72490.0


def test_ns_to_ms_none_in_none_out():
    """A duration Ollama didn't return must never be turned into a fake `0`."""
    assert ns_to_ms(None) is None


def test_ns_to_ms_zero_is_a_real_value_not_none():
    assert ns_to_ms(0) == 0.0


# --- § FASE 2A.1: _log_generation_metrics field extraction --------------------------


_FAKE_OLLAMA_METRICS = {
    "prompt_eval_count": 1800,
    "prompt_eval_duration": 70_000_000_000,
    "eval_count": 63,
    "eval_duration": 2_000_000_000,
    "load_duration": 400_000_000,
    "total_duration": 72_490_000_000,
}


def test_log_generation_metrics_emits_expected_fields(caplog):
    provider = OllamaProvider(name="qwen3.5:0.8b", think=False, role="FAST")

    with caplog.at_level(logging.INFO, logger="jarvis.ollama_provider"):
        provider._log_generation_metrics(_FAKE_OLLAMA_METRICS, prompt="ciao", system_prompt="Sei JARVIS.")

    record = caplog.records[-1]
    assert record.message == "ollama_generation_metrics"
    fields = record.fields

    assert fields["model"] == "qwen3.5:0.8b"
    assert fields["target"] == "FAST"
    assert fields["think"] is False
    assert fields["promptChars"] == len("ciao")
    assert fields["systemPromptChars"] == len("Sei JARVIS.")
    assert fields["totalInputChars"] == len("ciao") + len("Sei JARVIS.")
    assert fields["promptEvalCount"] == 1800
    assert fields["promptEvalDurationMs"] == 70000.0
    assert fields["evalCount"] == 63
    assert fields["evalDurationMs"] == 2000.0
    assert fields["loadDurationMs"] == 400.0
    assert fields["totalDurationMs"] == 72490.0


def test_log_generation_metrics_never_logs_prompt_or_system_prompt_text(caplog):
    provider = OllamaProvider(name="qwen3.5:0.8b")

    with caplog.at_level(logging.INFO, logger="jarvis.ollama_provider"):
        provider._log_generation_metrics(
            _FAKE_OLLAMA_METRICS, prompt="dati personali segreti", system_prompt="prompt segreto"
        )

    fields = caplog.records[-1].fields
    assert "prompt" not in fields
    assert "system_prompt" not in fields
    logged_values = [str(v) for v in fields.values()]
    assert not any("dati personali segreti" in v or "prompt segreto" in v for v in logged_values)


def test_log_generation_metrics_handles_missing_ollama_fields_gracefully():
    """A response body missing some metrics (older Ollama, partial warmup body)
    must never crash the caller - it already succeeded per raise_for_status()."""
    provider = OllamaProvider(name="qwen3.5:0.8b")
    provider._log_generation_metrics({}, prompt="", system_prompt=None)


def test_log_generation_metrics_role_none_when_provider_built_without_role(caplog):
    provider = OllamaProvider(name="qwen3.5:0.8b")

    with caplog.at_level(logging.INFO, logger="jarvis.ollama_provider"):
        provider._log_generation_metrics(_FAKE_OLLAMA_METRICS, prompt="ciao", system_prompt=None)

    assert caplog.records[-1].fields["target"] is None


@pytest.mark.asyncio
async def test_generate_logs_metrics_with_role(monkeypatch, caplog):
    provider = OllamaProvider(name="qwen3.5:0.8b", role="BRAIN")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "ok", "done": True, **_FAKE_OLLAMA_METRICS}

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, path, json):
            return FakeResponse()

    monkeypatch.setattr(provider, "_client", lambda: FakeAsyncClient())

    with caplog.at_level(logging.INFO, logger="jarvis.ollama_provider"):
        await provider.generate("ciao", system_prompt="Sei JARVIS.")

    fields = caplog.records[-1].fields
    assert fields["target"] == "BRAIN"
    assert fields["evalCount"] == 63
    assert fields["promptEvalCount"] == 1800


@pytest.mark.asyncio
async def test_load_model_logs_metrics_when_warmup_body_has_them(monkeypatch, caplog):
    provider = OllamaProvider(name="qwen3.5:0.8b", role="FAST")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return dict(_FAKE_OLLAMA_METRICS)

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, path, json):
            return FakeResponse()

    monkeypatch.setattr(provider, "_client", lambda: FakeAsyncClient())

    with caplog.at_level(logging.INFO, logger="jarvis.ollama_provider"):
        await provider.load_model()

    fields = caplog.records[-1].fields
    assert fields["target"] == "FAST"
    assert fields["loadDurationMs"] == 400.0


@pytest.mark.asyncio
async def test_load_model_warmup_survives_empty_body(monkeypatch):
    """§ richiesta esplicita: un warmup 2xx con body vuoto/non-JSON non deve
    mai far fallire load_model - ha già avuto successo."""
    provider = OllamaProvider(name="qwen3.5:0.8b")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("no body")

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, path, json):
            return FakeResponse()

    monkeypatch.setattr(provider, "_client", lambda: FakeAsyncClient())

    await provider.load_model()

    assert provider._loaded is True


# --- § FASE 2A.1: tokens_generated must reflect eval_count, never prompt_eval_count -


@pytest.mark.asyncio
async def test_tokens_generated_reflects_eval_count_not_prompt_eval_count(monkeypatch):
    """Regression pin: `eval_count` (output tokens) and `prompt_eval_count`
    (input tokens) are deliberately different in this fake response - if
    GenerationResult.tokens_generated ever started reading the wrong field,
    this test would catch it immediately."""
    provider = OllamaProvider(name="qwen3.5:0.8b")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": "ok",
                "done": True,
                "prompt_eval_count": 1800,
                "eval_count": 63,
            }

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, path, json):
            return FakeResponse()

    monkeypatch.setattr(provider, "_client", lambda: FakeAsyncClient())

    result = await provider.generate("ciao")

    assert result.tokens_generated == 63
    assert result.tokens_generated != 1800


# --- § FASE 2A.1: role propagation via ModelManager/build_provider ------------------


def test_build_provider_forwards_role_only_to_ollama():
    ollama = build_provider(
        "ollama", "qwen3.5:0.8b", base_url="http://127.0.0.1:11434", context_size=4096,
        request_timeout=30.0, role="FAST",
    )
    assert isinstance(ollama, OllamaProvider)
    assert ollama.role == "FAST"


def test_build_provider_role_defaults_to_none():
    ollama = build_provider(
        "ollama", "qwen3.5:0.8b", base_url="http://127.0.0.1:11434", context_size=4096,
        request_timeout=30.0,
    )
    assert ollama.role is None


def test_build_provider_fake_and_llamacpp_ignore_role():
    fake = build_provider(
        "fake", "fast-fake", base_url="", context_size=4096, request_timeout=30.0, role="FAST"
    )
    llama = build_provider(
        "llamacpp", "some-model", base_url="http://127.0.0.1:8080", context_size=4096,
        request_timeout=30.0, role="BRAIN",
    )
    assert not hasattr(fake, "role")
    assert not hasattr(llama, "role")


def test_model_manager_gives_fast_and_brain_providers_their_own_role(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("FAST_MODEL_BACKEND", "ollama")
    monkeypatch.setenv("BRAIN_MODEL_BACKEND", "ollama")
    settings = get_settings()

    manager = ModelManager(settings)

    fast = manager.get_provider(ExecutionTarget.FAST)
    brain = manager.get_provider(ExecutionTarget.BRAIN)
    assert fast.role == "FAST"
    assert brain.role == "BRAIN"
    get_settings.cache_clear()
