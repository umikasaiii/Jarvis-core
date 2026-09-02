"""llama.cpp server backend adapter.

Talks to a `llama-server` (a.k.a. `llama.cpp` server) instance over its
native HTTP API (`/completion`, `/health`, `/v1/models`). Model loading is
controlled by how the server process itself was started (`--model
<FAST_MODEL_PATH>` etc.) — this adapter treats the server as already
running with the desired model and only tracks whether it has responded
successfully at least once.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from providers.base import GenerationChunk, GenerationResult, LlmProvider, LlmProviderError, ModelInfo


class LlamaCppProvider(LlmProvider):
    def __init__(
        self,
        name: str,
        base_url: str = "http://127.0.0.1:8080",
        context_size: int = 4096,
        request_timeout: float = 120.0,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.context_size = context_size
        self.request_timeout = request_timeout
        self._loaded = False

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.request_timeout)

    def _build_prompt(self, prompt: str, system_prompt: str | None) -> str:
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> GenerationResult:
        payload = {
            "prompt": self._build_prompt(prompt, system_prompt),
            "temperature": temperature,
            "n_predict": max_tokens or -1,
            "stream": False,
        }
        try:
            async with self._client() as client:
                resp = await client.post("/completion", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LlmProviderError(f"llama.cpp request failed: {exc}") from exc
        return GenerationResult(
            text=data.get("content", ""),
            finish_reason="stop" if data.get("stop") else "length",
            tokens_generated=data.get("tokens_predicted", 0),
        )

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[GenerationChunk]:
        payload = {
            "prompt": self._build_prompt(prompt, system_prompt),
            "temperature": temperature,
            "n_predict": max_tokens or -1,
            "stream": True,
        }
        try:
            async with self._client() as client:
                async with client.stream("POST", "/completion", json=payload) as resp:
                    resp.raise_for_status()
                    tokens_generated = 0
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        chunk = json.loads(line.removeprefix("data: "))
                        tokens_generated += 1
                        if chunk.get("stop"):
                            yield GenerationChunk(
                                content=chunk.get("content", ""),
                                done=True,
                                finish_reason="stop",
                                tokens_generated=chunk.get("tokens_predicted", tokens_generated),
                            )
                            return
                        yield GenerationChunk(
                            content=chunk.get("content", ""),
                            done=False,
                            tokens_generated=tokens_generated,
                        )
        except httpx.HTTPError as exc:
            raise LlmProviderError(f"llama.cpp streaming request failed: {exc}") from exc

    async def health(self) -> bool:
        try:
            async with self._client() as client:
                resp = await client.get("/health")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        try:
            async with self._client() as client:
                resp = await client.get("/v1/models")
                resp.raise_for_status()
                data = resp.json()
                return [m["id"] for m in data.get("data", [])] or [self.name]
        except httpx.HTTPError as exc:
            raise LlmProviderError(f"Failed to list llama.cpp models: {exc}") from exc

    async def load_model(self) -> None:
        healthy = await self.health()
        if not healthy:
            raise LlmProviderError(
                f"llama.cpp server at {self.base_url} is not reachable/ready"
            )
        self._loaded = True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            backend="llamacpp",
            context_size=self.context_size,
            loaded=self._loaded,
        )
