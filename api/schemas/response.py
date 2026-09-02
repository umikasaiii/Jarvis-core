from __future__ import annotations

from pydantic import BaseModel

from api.schemas.common import ExecutionTarget, FinishReason, ResponseStatus


class JarvisResponse(BaseModel):
    requestId: str
    status: ResponseStatus
    text: str = ""
    modelUsed: str | None = None
    targetUsed: ExecutionTarget | None = None
    executionTimeMs: float = 0.0
    tokensGenerated: int = 0
    finishReason: FinishReason | None = None
    warnings: list[str] = []
    error: str | None = None
