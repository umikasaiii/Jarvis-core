"""JARVIS Core application factory.

    Android
      |
      v
    API Layer (api/routes)
      |
      v
    Request Orchestrator (ai/orchestrator.py)
      |
      v
    CoreAiRouter (ai/router.py) -> FastModel / BrainModel (providers/*)
      |
      v
    Response Streaming

Everything under `future/` (memory, RAG, context engine, actions, voice,
vision) is a documented extension point, not wired into the request path
yet — see api/routes/capabilities.py, which currently reports all of them
as `false`.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ai.model_manager import ModelManager
from ai.orchestrator import RequestOrchestrator
from ai.queue import InferenceQueue
from ai.router import CoreAiRouter
from api.routes import ai as ai_routes
from api.routes import capabilities as capabilities_routes
from api.routes import chat as chat_routes
from api.routes import health as health_routes
from api.routes import models as models_routes
from conversations.store import InMemoryConversationStore
from core.config import Settings, get_settings
from core.logging import configure_logging, get_logger, log_event
from core.security import (
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    TokenAuthMiddleware,
    warn_if_unsafe_binding,
)

logger = get_logger("jarvis.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    warn_if_unsafe_binding(settings)

    app.state.settings = settings
    app.state.start_time = time.monotonic()
    app.state.model_manager = ModelManager(settings)
    app.state.ai_router = CoreAiRouter(settings)
    app.state.queue = InferenceQueue(
        max_concurrent=settings.max_concurrent_requests,
        max_queue_size=settings.max_queue_size,
    )
    app.state.conversation_store = InMemoryConversationStore(
        max_messages=settings.max_conversation_messages,
        max_chars=settings.max_context_chars,
    )
    app.state.orchestrator = RequestOrchestrator(
        settings=settings,
        model_manager=app.state.model_manager,
        router=app.state.ai_router,
        queue=app.state.queue,
        conversation_store=app.state.conversation_store,
        system_prompt=settings.load_system_prompt(),
    )

    log_event(
        logger,
        20,
        "server_starting",
        host=settings.server_host,
        port=settings.server_port,
        fastBackend=settings.fast_model_backend,
        brainBackend=settings.brain_model_backend,
    )

    # Startup must succeed even if no model backend is reachable yet: the
    # server should come up and report per-model status via /v1/models
    # rather than crashing because a model failed to load.
    available = await app.state.model_manager.any_available()
    log_event(logger, 20, "server_started", llmAvailable=available)

    yield

    log_event(logger, 20, "server_shutting_down")
    # No persistent connections/background tasks to tear down in this phase
    # beyond what each request's own `async with` blocks already close.
    log_event(logger, 20, "server_shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="JARVIS Core",
        version="0.1.0",
        description="Local orchestration server for the JARVIS assistant.",
        lifespan=lifespan,
    )

    settings = get_settings()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)
    app.add_middleware(TokenAuthMiddleware, api_token=settings.api_token)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_body_bytes)

    app.include_router(health_routes.router, prefix="/v1")
    app.include_router(capabilities_routes.router, prefix="/v1")
    app.include_router(models_routes.router, prefix="/v1")
    app.include_router(chat_routes.router, prefix="/v1")
    app.include_router(ai_routes.router, prefix="/v1")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Never fail silently: invalid requests get a clear 422 with the
        # validation errors instead of Android receiving nothing useful.
        return JSONResponse(status_code=422, content={"error": "invalid_request", "detail": exc.errors()})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        log_event(logger, 40, "unhandled_exception", path=str(request.url.path), error=str(exc))
        return JSONResponse(status_code=500, content={"error": "internal_error"})

    return app


app = create_app()
