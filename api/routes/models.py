from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_model_manager
from api.schemas.common import ExecutionTarget
from api.schemas.models import ModelInfo, ModelsResponse
from ai.model_manager import ModelManager

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
async def list_models(model_manager: ModelManager = Depends(get_model_manager)):
    """Live view of the FAST/BRAIN providers: backend, model name, and
    whether the backend currently responds to a health check."""
    infos = await model_manager.list_models()
    models = [
        ModelInfo(
            role=ExecutionTarget(info.extra["role"]),
            backend=info.backend,
            name=info.name,
            available=info.extra["available"],
            contextSize=info.context_size,
            loaded=info.loaded,
            error=info.extra.get("loadError"),
        )
        for info in infos
    ]
    return ModelsResponse(models=models)
