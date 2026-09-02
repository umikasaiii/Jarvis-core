from __future__ import annotations

from pydantic import BaseModel

from api.schemas.common import ExecutionTarget


class ModelInfo(BaseModel):
    role: ExecutionTarget
    backend: str
    name: str
    available: bool
    contextSize: int
    loaded: bool = False
    error: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
