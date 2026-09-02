"""Deterministic in-process provider used by tests and as a zero-dependency
default so the server can start (and be demoed) without any real model
installed.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from providers.base import GenerationChunk, GenerationResult, LlmProvider, LlmProviderError, ModelInfo


class FakeLlmProvider(LlmProvider):
    def __init__(
        self,
        name: str = "fake-model",
        context_size: int = 4096,
        token_delay_seconds: float = 0.0,
        healthy: bool = True,
    ) -> None:
        self.name = name
        self.context_size = context_size
        self.token_delay_seconds = token_delay_seconds
        self.healthy = healthy
        self._loaded = False
        self.calls: list[str] = []

    def _reply_tokens(self, prompt: str) -> list[str]:
        words = f"[{self.name}] echo: {prompt}".split(" ")
        return [w + " " for w in words]

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> GenerationResult:
        if not self.healthy:
            raise LlmProviderError(f"{self.name} backend unavailable")
        self.calls.append(prompt)
        tokens = self._reply_tokens(prompt)
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        text = "".join(tokens)
        return GenerationResult(text=text, finish_reason="stop", tokens_generated=len(tokens))

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[GenerationChunk]:
        if not self.healthy:
            raise LlmProviderError(f"{self.name} backend unavailable")
        self.calls.append(prompt)
        tokens = self._reply_tokens(prompt)
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        generated = 0
        for token in tokens:
            if self.token_delay_seconds:
                await asyncio.sleep(self.token_delay_seconds)
            generated += 1
            yield GenerationChunk(content=token, done=False, tokens_generated=generated)
        yield GenerationChunk(
            content="", done=True, finish_reason="stop", tokens_generated=generated
        )

    async def health(self) -> bool:
        return self.healthy

    async def list_models(self) -> list[str]:
        return [self.name]

    async def load_model(self) -> None:
        if not self.healthy:
            raise LlmProviderError(f"{self.name} failed to load")
        self._loaded = True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            backend="fake",
            context_size=self.context_size,
            loaded=self._loaded,
        )
