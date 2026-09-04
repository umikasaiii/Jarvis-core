"""ModelManager: owns the FAST and BRAIN provider instances.

Responsibilities kept deliberately small for this phase:
  - know which providers exist and whether their model is available;
  - serialize loading so two concurrent requests never trigger a double
    load of the same backend;
  - surface load errors instead of crashing the server;
  - expose metrics/info for /v1/models and /v1/health.

No aggressive preload/unload scheduling is implemented — stability and fast
response beat clever memory management for this phase.
"""
from __future__ import annotations

import asyncio

from api.schemas.common import ExecutionTarget
from core.config import Settings
from core.logging import get_logger, log_event
from providers.base import LlmProvider, LlmProviderError, ModelInfo
from providers.fake import FakeLlmProvider
from providers.llamacpp import LlamaCppProvider
from providers.ollama import OllamaProvider

logger = get_logger("jarvis.model_manager")


def build_provider(
    backend: str,
    name: str,
    *,
    base_url: str,
    context_size: int,
    request_timeout: float,
    think: bool = False,
    role: str | None = None,
) -> LlmProvider:
    if backend == "fake":
        return FakeLlmProvider(name=name, context_size=context_size)
    if backend == "ollama":
        return OllamaProvider(
            name=name,
            base_url=base_url,
            context_size=context_size,
            request_timeout=request_timeout,
            think=think,
            role=role,
        )
    if backend == "llamacpp":
        return LlamaCppProvider(
            name=name, base_url=base_url, context_size=context_size, request_timeout=request_timeout
        )
    raise ValueError(f"Unknown model backend: {backend}")


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[ExecutionTarget, LlmProvider] = {
            ExecutionTarget.FAST: build_provider(
                settings.fast_model_backend,
                settings.fast_model_name,
                base_url=settings.fast_base_url,
                context_size=settings.fast_context_size,
                request_timeout=settings.request_timeout,
                think=settings.fast_think,
                role=ExecutionTarget.FAST.value,
            ),
            ExecutionTarget.BRAIN: build_provider(
                settings.brain_model_backend,
                settings.brain_model_name,
                base_url=settings.brain_base_url,
                context_size=settings.brain_context_size,
                request_timeout=settings.request_timeout,
                think=settings.brain_think,
                role=ExecutionTarget.BRAIN.value,
            ),
        }
        self._load_locks: dict[ExecutionTarget, asyncio.Lock] = {
            target: asyncio.Lock() for target in self._providers
        }
        self._load_errors: dict[ExecutionTarget, str | None] = {
            target: None for target in self._providers
        }
        self._loaded: dict[ExecutionTarget, bool] = {target: False for target in self._providers}

    def get_provider(self, target: ExecutionTarget) -> LlmProvider:
        return self._providers[target]

    async def ensure_loaded(self, target: ExecutionTarget) -> None:
        """Load the model for `target` if not already loaded, without
        allowing two concurrent loads of the same target to race."""
        if self._loaded[target]:
            return
        async with self._load_locks[target]:
            if self._loaded[target]:
                return
            try:
                await self._providers[target].load_model()
                self._loaded[target] = True
                self._load_errors[target] = None
            except LlmProviderError as exc:
                self._load_errors[target] = str(exc)
                log_event(
                    logger,
                    40,
                    "model_load_failed",
                    target=target.value,
                    error=str(exc),
                )
                raise

    async def health(self) -> dict[ExecutionTarget, bool]:
        results = {}
        for target, provider in self._providers.items():
            try:
                results[target] = await provider.health()
            except Exception:  # noqa: BLE001 - health must never raise
                results[target] = False
        return results

    async def any_available(self) -> bool:
        health = await self.health()
        return any(health.values())

    def active_model_name(self, target: ExecutionTarget) -> str:
        return self._providers[target].get_model_info().name

    async def list_models(self) -> list[ModelInfo]:
        infos = []
        health = await self.health()
        for target, provider in self._providers.items():
            info = provider.get_model_info()
            info.loaded = self._loaded[target]
            info.extra = {
                **info.extra,
                "role": target.value,
                "available": health.get(target, False),
                "loadError": self._load_errors[target],
            }
            infos.append(info)
        return infos

    def load_error(self, target: ExecutionTarget) -> str | None:
        return self._load_errors[target]
