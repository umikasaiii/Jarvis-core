from __future__ import annotations

from pydantic import BaseModel

from api.schemas.common import ExecutionTarget, FinishReason, StreamEventType


class StreamEvent(BaseModel):
    type: StreamEventType
    requestId: str
    content: str | None = None
    modelUsed: str | None = None
    targetUsed: ExecutionTarget | None = None
    executionTimeMs: float | None = None
    tokensGenerated: int | None = None
    finishReason: FinishReason | None = None
    error: str | None = None

    def to_sse(self) -> str:
        return f"data: {self.model_dump_json(exclude_none=True)}\n\n"
