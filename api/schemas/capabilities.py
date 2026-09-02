from __future__ import annotations

from pydantic import BaseModel


class CapabilitiesResponse(BaseModel):
    chat: bool = True
    streaming: bool = True
    fastModel: bool = True
    brainModel: bool = True
    memory: bool = False
    rag: bool = False
    voice: bool = False
    vision: bool = False
    contextEngine: bool = False
    actions: bool = False
    protocolVersion: str = "1"
