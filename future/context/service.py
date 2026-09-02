"""Placeholder interface for the future ContextEngine. Not wired in yet."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ContextEngine(ABC):
    @abstractmethod
    async def build_context(self, raw_context: dict[str, Any] | None) -> dict[str, Any]: ...
