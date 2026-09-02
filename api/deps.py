"""FastAPI dependency accessors.

Everything the routes need (settings, router, model manager, queue,
orchestrator, start time) lives on `app.state`, built once in the lifespan
handler in `app/main.py`. These small functions just pull typed references
out of `request.app.state` so route signatures stay declarative.
"""
from __future__ import annotations

from fastapi import Request

from ai.model_manager import ModelManager
from ai.orchestrator import RequestOrchestrator
from ai.queue import InferenceQueue
from ai.router import CoreAiRouter
from core.config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


def get_router(request: Request) -> CoreAiRouter:
    return request.app.state.ai_router


def get_queue(request: Request) -> InferenceQueue:
    return request.app.state.queue


def get_orchestrator(request: Request) -> RequestOrchestrator:
    return request.app.state.orchestrator
