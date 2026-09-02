"""Placeholder interface for the future VisionService. Not wired in yet."""
from __future__ import annotations

from abc import ABC, abstractmethod


class VisionService(ABC):
    @abstractmethod
    async def describe_image(self, image_bytes: bytes, prompt: str | None = None) -> str: ...
