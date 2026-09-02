"""Shared enums used across request/response schemas.

Names and values are kept in sync with the JARVIS Android contract
(JarvisCoreRequest/JarvisCoreResponse/AiExecutionTarget/AiRequestType). If a
mismatch with the Android app is ever found, it must be documented before
this contract is changed unilaterally.
"""
from __future__ import annotations

from enum import Enum


class RequestType(str, Enum):
    CHAT = "CHAT"
    COMMAND = "COMMAND"
    COMPLEX = "COMPLEX"
    MEMORY = "MEMORY"
    TOOL = "TOOL"
    PROACTIVE = "PROACTIVE"


class ExecutionTarget(str, Enum):
    AUTO = "AUTO"
    FAST = "FAST"
    BRAIN = "BRAIN"


class ResponseStatus(str, Enum):
    OK = "OK"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StreamEventType(str, Enum):
    START = "start"
    TOKEN = "token"
    DONE = "done"
    ERROR = "error"
