from __future__ import annotations

from fastapi import APIRouter

from api.schemas.capabilities import CapabilitiesResponse
from core.config import PROTOCOL_VERSION

router = APIRouter(tags=["capabilities"])


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities():
    """Declares what this Core build supports so Android can adapt to
    future versions without hardcoding assumptions."""
    return CapabilitiesResponse(
        chat=True,
        streaming=True,
        fastModel=True,
        brainModel=True,
        memory=False,
        rag=False,
        voice=False,
        vision=False,
        contextEngine=False,
        actions=False,
        protocolVersion=PROTOCOL_VERSION,
    )
