from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from api.schemas.common import ExecutionTarget, RequestType

MAX_TEXT_LENGTH = 20_000


class JarvisRequest(BaseModel):
    """A single request from Android (or any client) to JARVIS Core.

    `context` and `metadata` are intentionally optional and untyped-ish:
    Android is not required to send location, notifications, or any other
    contextual payload. Future phases (Context Engine, Memory) will define
    stricter shapes for `context` without breaking older clients.
    """

    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    protocolVersion: str
    conversationId: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    requestType: RequestType = RequestType.CHAT
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    preferredTarget: ExecutionTarget = ExecutionTarget.AUTO
    allowFallback: bool = True
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v
