"""Ollama backend adapter.

Talks to a locally running `ollama serve` over its REST API
(https://github.com/ollama/ollama/blob/main/docs/api.md). Ollama handles
model loading/unloading itself, so `load_model` here just warms the model
with an empty-ish request and `list_models` hits `/api/tags`.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from core.logging import get_logger, log_event
from providers.base import GenerationChunk, GenerationResult, LlmProvider, LlmProviderError, ModelInfo

logger = get_logger("jarvis.ollama_provider")


def ns_to_ms(value: int | float | None) -> float | None:
    """Ollama reports every `*_duration` field in nanoseconds - convert to
    milliseconds for human-readable logs. `None` in, `None` out (a field
    Ollama didn't return is never silently turned into a fake `0`)."""
    if value is None:
        return None
    return value / 1_000_000.0


def _safe_json(resp: httpx.Response) -> dict | None:
    """Best-effort body parse for diagnostics-only reads - a warmup call
    that already succeeded (2xx) must never fail because its body turned
    out to be empty or non-JSON."""
    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError):
        return None


class OllamaProvider(LlmProvider):
    def __init__(
        self,
        name: str,
        base_url: str = "http://127.0.0.1:11434",
        context_size: int = 4096,
        request_timeout: float = 120.0,
        think: bool = False,
        role: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.context_size = context_size
        self.request_timeout = request_timeout
        self.think = think
        # § FASE 2A.1: purely a diagnostics label ("FAST"/"BRAIN") - this
        # provider never routes or behaves differently based on it. `None`
        # when constructed without a role (e.g. a standalone script or a
        # provider built directly in a test) - the metrics log then simply
        # omits `target`, matching "target FAST/BRAIN se disponibile".
        self.role = role
        self._loaded = False

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.request_timeout)

    def _build_prompt(self, prompt: str, system_prompt: str | None) -> dict:
        payload: dict = {
            "model": self.name,
            "prompt": prompt,
            "think": self.think,
            "options": {"num_ctx": self.context_size},
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def _log_generation_metrics(self, data: dict, prompt: str, system_prompt: str | None) -> None:
        """§ FASE 2A.1 - profiling latenza FAST, before any optimization.

        `data` is Ollama's own raw response dict (the non-streamed body, or
        the final `"done": true` line of a stream) - it already carries
        `prompt_eval_count`/`prompt_eval_duration`/`eval_count`/
        `eval_duration`/`load_duration`/`total_duration` (nanoseconds) per
        Ollama's own API; this only reads and relabels them, never invents a
        number Ollama didn't return. Only sizes are logged, never the prompt
        or system prompt text itself (§ "non loggare... dati personali") -
        `core.logging`'s own `_REDACT_KEYS` would strip those two field names
        anyway if they were ever passed raw.
        """
        prompt_chars = len(prompt)
        system_prompt_chars = len(system_prompt) if system_prompt else 0
        log_event(
            logger,
            20,
            "ollama_generation_metrics",
            model=self.name,
            target=self.role,
            think=self.think,
            promptChars=prompt_chars,
            systemPromptChars=system_prompt_chars,
            totalInputChars=prompt_chars + system_prompt_chars,
            promptEvalCount=data.get("prompt_eval_count"),
            promptEvalDurationMs=ns_to_ms(data.get("prompt_eval_duration")),
            evalCount=data.get("eval_count"),
            evalDurationMs=ns_to_ms(data.get("eval_duration")),
            loadDurationMs=ns_to_ms(data.get("load_duration")),
            totalDurationMs=ns_to_ms(data.get("total_duration")),
        )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> GenerationResult:
        payload = self._build_prompt(prompt, system_prompt)
        payload["stream"] = False
        payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        try:
            async with self._client() as client:
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LlmProviderError(f"Ollama request failed: {exc}") from exc
        self._log_generation_metrics(data, prompt, system_prompt)
        return GenerationResult(
            text=data.get("response", ""),
            finish_reason="stop" if data.get("done") else "length",
            tokens_generated=data.get("eval_count", 0),
        )

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[GenerationChunk]:
        payload = self._build_prompt(prompt, system_prompt)
        payload["stream"] = True
        payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with self._client() as client:
                async with client.stream("POST", "/api/generate", json=payload) as resp:
                    resp.raise_for_status()
                    tokens_generated = 0
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        if chunk.get("done"):
                            self._log_generation_metrics(chunk, prompt, system_prompt)
                            yield GenerationChunk(
                                content="",
                                done=True,
                                finish_reason="stop",
                                tokens_generated=chunk.get("eval_count", tokens_generated),
                            )
                            return
                        tokens_generated += 1
                        yield GenerationChunk(
                            content=chunk.get("response", ""),
                            done=False,
                            tokens_generated=tokens_generated,
                        )
        except httpx.HTTPError as exc:
            raise LlmProviderError(f"Ollama streaming request failed: {exc}") from exc

    async def health(self) -> bool:
        try:
            async with self._client() as client:
                resp = await client.get("/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        try:
            async with self._client() as client:
                resp = await client.get("/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError as exc:
            raise LlmProviderError(f"Failed to list Ollama models: {exc}") from exc

    async def load_model(self) -> None:
        try:
            async with self._client() as client:
                payload = self._build_prompt("", None)
                payload["stream"] = False
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
                # § FASE 2A.1: the warmup is itself a real /api/generate call
                # (and, being the FIRST one, the one most likely to carry a
                # non-trivial loadDurationMs) - logging it the same way as a
                # real turn is what lets request 1's ~72s be told apart from
                # "the model itself takes that long to load" vs "generation
                # is slow even once warm". A malformed/empty body here must
                # never fail the warmup itself - it already succeeded per
                # raise_for_status() above.
                data = _safe_json(resp)
        except httpx.HTTPError as exc:
            raise LlmProviderError(f"Failed to warm up Ollama model {self.name}: {exc}") from exc
        if data is not None:
            self._log_generation_metrics(data, "", None)
        self._loaded = True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            backend="ollama",
            context_size=self.context_size,
            loaded=self._loaded,
        )
