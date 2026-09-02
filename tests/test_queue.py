from __future__ import annotations

import asyncio

import pytest

from ai.queue import InferenceQueue, QueueFullError


@pytest.mark.asyncio
async def test_queue_limits_concurrency():
    queue = InferenceQueue(max_concurrent=1, max_queue_size=5)
    order: list[str] = []

    async def worker(name: str):
        async with queue.slot():
            order.append(f"start:{name}")
            await asyncio.sleep(0.05)
            order.append(f"end:{name}")

    await asyncio.gather(worker("a"), worker("b"))
    # With max_concurrent=1, "a" must fully finish before "b" starts.
    assert order == ["start:a", "end:a", "start:b", "end:b"]


@pytest.mark.asyncio
async def test_queue_full_raises():
    queue = InferenceQueue(max_concurrent=1, max_queue_size=1)

    async def hold():
        async with queue.slot():
            await asyncio.sleep(0.2)

    holder_task = asyncio.create_task(hold())
    await asyncio.sleep(0.01)  # let holder acquire the only concurrency slot

    async def waiter():
        async with queue.slot():
            pass

    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0.01)  # waiter now occupies the 1 queue slot

    with pytest.raises(QueueFullError):
        async with queue.slot():
            pass

    await holder_task
    await waiter_task


@pytest.mark.asyncio
async def test_queue_stats():
    queue = InferenceQueue(max_concurrent=2, max_queue_size=5)
    stats = queue.stats()
    assert stats["active"] == 0
    assert stats["maxConcurrent"] == 2
    assert stats["maxQueueSize"] == 5


@pytest.mark.asyncio
async def test_queue_releases_slot_on_cancellation():
    queue = InferenceQueue(max_concurrent=1, max_queue_size=5)

    async def cancellable():
        async with queue.slot():
            await asyncio.sleep(10)

    task = asyncio.create_task(cancellable())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Slot must be free again for a new request.
    async with queue.slot():
        assert queue.active == 1
