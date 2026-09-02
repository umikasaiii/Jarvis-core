from __future__ import annotations

import pytest

from api.schemas.common import ExecutionTarget
from ai.model_manager import ModelManager
from core.config import get_settings
from providers.base import LlmProviderError


@pytest.mark.asyncio
async def test_model_unavailable_raises_on_ensure_loaded(monkeypatch):
    get_settings.cache_clear()
    settings = get_settings()
    manager = ModelManager(settings)
    manager._providers[ExecutionTarget.BRAIN].healthy = False  # type: ignore[attr-defined]

    with pytest.raises(LlmProviderError):
        await manager.ensure_loaded(ExecutionTarget.BRAIN)

    assert manager.load_error(ExecutionTarget.BRAIN) is not None
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_concurrent_ensure_loaded_does_not_double_load(monkeypatch):
    import asyncio

    get_settings.cache_clear()
    settings = get_settings()
    manager = ModelManager(settings)
    provider = manager._providers[ExecutionTarget.FAST]

    load_calls = []
    original_load = provider.load_model

    async def counting_load():
        load_calls.append(1)
        await asyncio.sleep(0.05)
        await original_load()

    provider.load_model = counting_load  # type: ignore[method-assign]

    await asyncio.gather(
        manager.ensure_loaded(ExecutionTarget.FAST),
        manager.ensure_loaded(ExecutionTarget.FAST),
        manager.ensure_loaded(ExecutionTarget.FAST),
    )

    assert len(load_calls) == 1
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_list_models_reports_role_and_availability():
    get_settings.cache_clear()
    settings = get_settings()
    manager = ModelManager(settings)
    infos = await manager.list_models()
    roles = {info.extra["role"] for info in infos}
    assert roles == {"FAST", "BRAIN"}
    assert all(info.extra["available"] for info in infos)
    get_settings.cache_clear()
