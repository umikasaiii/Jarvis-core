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

from providers.base import GenerationChunk, GenerationResult, LlmProvider, LlmProviderError, ModelInfo


class OllamaProvider(LlmProvider):
    def __init__(
        self,
        name: str,
        base_url: str = "http://127.0.0.1:11434",
        context_size: int = 4096,
        request_timeout: float = 120.0,
        think: bool = False,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.context_size = context_size
        self.request_timeout = request_timeout
        self.think = think
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
        except httpx.HTTPError as exc:
            raise LlmProviderError(f"Failed to warm up Ollama model {self.name}: {exc}") from exc
        self._loaded = True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            backend="ollama",
            context_size=self.context_size,
            loaded=self._loaded,
        )
