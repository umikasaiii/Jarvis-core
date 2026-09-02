"""Lightweight, session-scoped conversation storage.

This is intentionally NOT long-term memory: it exists only to give the LLM
enough recent context within one conversation, trimmed to stay inside the
context window. A future `PersistentMemoryStore` can implement the same
`ConversationStore` interface (e.g. backed by SQLite or a real memory
service) without any change to callers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationStore(ABC):
    @abstractmethod
    async def get_history(self, conversation_id: str) -> list[Message]:
        ...

    @abstractmethod
    async def append(self, conversation_id: str, message: Message) -> None:
        ...

    @abstractmethod
    async def clear(self, conversation_id: str) -> None:
        ...


class InMemoryConversationStore(ConversationStore):
    """Process-local store. Conversations are isolated by `conversation_id`
    and each is capped to the most recent N messages / M characters so the
    prompt sent to the LLM never grows unbounded.
    """

    def __init__(self, max_messages: int = 20, max_chars: int = 12_000) -> None:
        self.max_messages = max_messages
        self.max_chars = max_chars
        self._conversations: dict[str, list[Message]] = {}

    async def get_history(self, conversation_id: str) -> list[Message]:
        return list(self._conversations.get(conversation_id, []))

    async def append(self, conversation_id: str, message: Message) -> None:
        history = self._conversations.setdefault(conversation_id, [])
        history.append(message)
        self._trim(history)

    async def clear(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)

    def _trim(self, history: list[Message]) -> None:
        while len(history) > self.max_messages:
            history.pop(0)
        total_chars = sum(len(m.content) for m in history)
        while total_chars > self.max_chars and len(history) > 1:
            removed = history.pop(0)
            total_chars -= len(removed.content)
