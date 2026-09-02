"""Placeholder interface for the future VoiceService. Not wired in yet."""
from __future__ import annotations

from abc import ABC, abstractmethod


class VoiceService(ABC):
    @abstractmethod
    async def speech_to_text(self, audio_bytes: bytes) -> str: ...

    @abstractmethod
    async def text_to_speech(self, text: str) -> bytes: ...
