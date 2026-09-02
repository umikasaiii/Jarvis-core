"""Security middleware for JARVIS Core.

This is a LAN-first server. HTTP without TLS is acceptable for local
development on a trusted network, but:

  - requests are never trusted blindly: an optional device/API token,
    a body-size cap, and a light in-memory rate limiter are always active;
  - binding to a non-localhost address is logged loudly so nobody mistakes
    a LAN listener for something safe to expose on the public internet;
  - true remote access is out of scope here and will be handled later via
    VPN/TLS, not by this middleware.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from core.config import Settings
from core.logging import get_logger

logger = get_logger("jarvis.security")

# Endpoints that must stay reachable without a token so orchestration tools
# (and Android's "is the Core alive?" probe) never get locked out.
_PUBLIC_PATHS = {"/v1/health", "/docs", "/openapi.json", "/redoc"}


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"error": "request_too_large", "maxBytes": self.max_bytes},
                    )
            except ValueError:
                pass
        return await call_next(request)


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Simple bearer-token check. Disabled (open) when no token is configured."""

    def __init__(self, app, api_token: str | None) -> None:
        super().__init__(app)
        self.api_token = api_token

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.api_token or request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        header = request.headers.get("authorization", "")
        token = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
        if token != self.api_token:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window-ish rate limiter keyed by client IP. Intentionally light."""

    def __init__(self, app, requests_per_minute: int) -> None:
        super().__init__(app)
        self.limit = requests_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self.limit <= 0 or request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._hits[client_ip]
        while window and now - window[0] > 60.0:
            window.popleft()

        if len(window) >= self.limit:
            return JSONResponse(status_code=429, content={"error": "rate_limited"})

        window.append(now)
        return await call_next(request)


def warn_if_unsafe_binding(settings: Settings) -> None:
    if settings.server_host not in ("127.0.0.1", "localhost", "::1"):
        if not settings.allow_remote_connections:
            logger.warning(
                "server_host is not localhost but ALLOW_REMOTE_CONNECTIONS is false; "
                "binding will still occur, review your network exposure",
                extra={"fields": {"server_host": settings.server_host}},
            )
        else:
            logger.warning(
                "JARVIS Core is bound for LAN access over plain HTTP. This is NOT safe "
                "for exposure over the public internet. Use a VPN/TLS for remote access.",
                extra={"fields": {"server_host": settings.server_host}},
            )
