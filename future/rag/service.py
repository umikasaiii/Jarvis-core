"""Placeholder interface for the future RagService. Not wired in yet."""
from __future__ import annotations

from abc import ABC, abstractmethod


class RagService(ABC):
    @abstractmethod
    async def index_document(self, path: str) -> None: ...

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[str]: ...
