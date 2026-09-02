"""Placeholder interface for the future MemoryService. Not wired in yet."""
from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryService(ABC):
    @abstractmethod
    async def remember(self, conversation_id: str, fact: str) -> None: ...

    @abstractmethod
    async def recall(self, conversation_id: str, query: str) -> list[str]: ...
