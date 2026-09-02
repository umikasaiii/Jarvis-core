from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    serverVersion: str
    protocolVersion: str
    uptimeSeconds: float
    llmAvailable: bool
    activeModel: str | None
    device: str
    timestamp: datetime
