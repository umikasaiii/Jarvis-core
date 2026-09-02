from __future__ import annotations

import platform
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from api.deps import get_model_manager, get_settings
from api.schemas.common import ExecutionTarget
from api.schemas.health import HealthResponse
from ai.model_manager import ModelManager
from core.config import PROTOCOL_VERSION, SERVER_VERSION, Settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    settings: Settings = Depends(get_settings),
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Extremely lightweight liveness probe. Never runs inference or pings backends.

    For a live per-backend health check use GET /v1/models instead.
    """
    active_model = model_manager.active_model_name(ExecutionTarget.FAST)
    uptime = round(time.monotonic() - request.app.state.start_time, 3)

    return HealthResponse(
        status="online",
        serverVersion=SERVER_VERSION,
        protocolVersion=PROTOCOL_VERSION,
        uptimeSeconds=uptime,
        llmAvailable=True,
        activeModel=active_model,
        device=platform.node() or "unknown",
        timestamp=datetime.now(timezone.utc),
    )
