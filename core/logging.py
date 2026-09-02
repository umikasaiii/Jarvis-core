"""Structured logging setup for JARVIS Core.

Logs are emitted as one JSON object per line so they are easy to ship to any
log aggregator later. By design we never log full user text, documents, the
system prompt, or secrets (API tokens) — only metadata useful for
diagnostics and performance tracking (requestId, requestType, target,
model, latency, token counts, success/error).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_REDACT_KEYS = {"text", "prompt", "system_prompt", "token", "api_token", "authorization"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            for key, value in extra.items():
                if key.lower() in _REDACT_KEYS:
                    continue
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
    root.addHandler(handler)

    # Keep third-party loggers reasonably quiet by default.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emit a structured log line with arbitrary metadata fields."""
    logger.log(level, message, extra={"fields": fields})


class Timer:
    """Tiny context manager / manual stopwatch for latency metrics."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def reset(self) -> None:
        self._start = time.perf_counter()
