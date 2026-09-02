from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.deps import get_orchestrator
from api.schemas.request import JarvisRequest
from api.schemas.response import JarvisResponse
from ai.orchestrator import ProtocolVersionError, RequestOrchestrator
from core.logging import get_logger, log_event

router = APIRouter(prefix="/ai", tags=["ai"])
logger = get_logger("jarvis.api.ai")


@router.post("/request", response_model=JarvisResponse)
async def ai_request(
    request: JarvisRequest, orchestrator: RequestOrchestrator = Depends(get_orchestrator)
):
    """Generic, non-streaming AI request. Suitable for COMMAND/TOOL/COMPLEX
    style one-off calls that don't need to grow conversation history."""
    try:
        return await orchestrator.handle_request(request, persist_history=True)
    except ProtocolVersionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/stream")
async def ai_stream(
    request: JarvisRequest,
    http_request: Request,
    orchestrator: RequestOrchestrator = Depends(get_orchestrator),
):
    """Server-Sent Events stream of START / TOKEN / DONE / ERROR events.

    Client disconnects are detected via `http_request.is_disconnected()` so
    a dropped Android connection cancels the underlying generation instead
    of leaving a zombie inference running.
    """
    try:
        orchestrator.validate_protocol_version(request)
    except ProtocolVersionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def event_generator():
        gen = orchestrator.handle_stream(request, persist_history=True)
        try:
            async for event in gen:
                if await http_request.is_disconnected():
                    log_event(
                        logger, 20, "client_disconnected", requestId=request.requestId
                    )
                    break
                yield event.to_sse()
        except asyncio.CancelledError:
            log_event(logger, 20, "stream_cancelled_by_server", requestId=request.requestId)
            raise
        finally:
            await gen.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
