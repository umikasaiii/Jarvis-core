from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_orchestrator
from api.schemas.request import JarvisRequest
from api.schemas.response import JarvisResponse
from ai.orchestrator import ProtocolVersionError, RequestOrchestrator

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=JarvisResponse)
async def chat(request: JarvisRequest, orchestrator: RequestOrchestrator = Depends(get_orchestrator)):
    """Convenience conversational endpoint: routes via CoreAiRouter, waits
    for the full reply, and (when `conversationId` is set) appends both
    turns to the conversation history for follow-up context."""
    try:
        return await orchestrator.handle_request(request, persist_history=True)
    except ProtocolVersionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
