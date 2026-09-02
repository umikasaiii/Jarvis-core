"""Placeholder interface for the future ActionEngine. Not wired in yet."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ActionEngine(ABC):
    @abstractmethod
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any: ...
