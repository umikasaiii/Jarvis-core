"""InferenceQueue: backpressure control for concurrent LLM inference.

Instead of letting an unbounded number of requests kick off inference
concurrently (which would saturate CPU/GPU and tank latency for everyone),
requests acquire a slot from a bounded semaphore. If the queue itself is
already full, callers get a clear "server busy" signal instead of hanging
forever or triggering yet another concurrent inference.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class QueueFullError(RuntimeError):
    """Raised when the inference queue has no room left for a new request."""


class InferenceQueue:
    def __init__(self, max_concurrent: int, max_queue_size: int) -> None:
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._waiting = 0
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    @asynccontextmanager
    async def slot(self):
        async with self._lock:
            if self._waiting >= self.max_queue_size:
                raise QueueFullError(
                    f"Inference queue is full ({self._waiting}/{self.max_queue_size})"
                )
            self._waiting += 1

        acquired = False
        try:
            await self._semaphore.acquire()
            acquired = True
            async with self._lock:
                self._waiting -= 1
                self._active += 1
            try:
                yield
            finally:
                async with self._lock:
                    self._active -= 1
                self._semaphore.release()
        finally:
            if not acquired:
                # Cancelled (e.g. client disconnect) while still queued.
                async with self._lock:
                    self._waiting -= 1

    def stats(self) -> dict:
        return {
            "active": self._active,
            "waiting": self._waiting,
            "maxConcurrent": self.max_concurrent,
            "maxQueueSize": self.max_queue_size,
        }
