"""Backend-independent LLM provider interface.

`CoreAiRouter` and `ModelManager` only ever talk to this interface, never to
a concrete backend. Adding a new backend (OpenAI-compatible server, a
different local runtime, ...) means writing one more subclass here — the
API layer, routing, streaming, and Android contract never change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class GenerationChunk:
    """One piece of streamed output."""

    content: str
    done: bool = False
    finish_reason: str | None = None
    tokens_generated: int | None = None


@dataclass
class GenerationResult:
    """Final, non-streamed result of a generation."""

    text: str
    finish_reason: str = "stop"
    tokens_generated: int = 0


@dataclass
class ModelInfo:
    name: str
    backend: str
    context_size: int
    loaded: bool = False
    extra: dict = field(default_factory=dict)


class LlmProviderError(RuntimeError):
    """Raised when a provider cannot fulfill a request (backend down, model missing, ...)."""


class LlmProvider(ABC):
    """Contract every LLM backend adapter must implement."""

    name: str

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Run a full (non-streaming) generation and return the complete text."""

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[GenerationChunk]:
        """Yield generation chunks as they become available."""

    @abstractmethod
    async def health(self) -> bool:
        """Return True if the backend is reachable and ready to serve requests."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return the model names known to the backend."""

    @abstractmethod
    async def load_model(self) -> None:
        """Ensure the configured model is loaded (no-op for backends that lazy-load)."""

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return static/known metadata about the configured model."""
