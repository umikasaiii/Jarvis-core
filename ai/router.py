"""CoreAiRouter: decides FAST vs BRAIN for a given request.

This phase deliberately uses simple, fully configurable heuristics — no
extra LLM call is used to decide routing. Rules, in priority order:

  1. an explicit `preferredTarget` of FAST or BRAIN always wins;
  2. requestType COMPLEX always goes to BRAIN;
  3. requestType COMMAND (short, direct instructions) goes to FAST;
  4. keyword heuristics (configurable lists) nudge toward FAST or BRAIN;
  5. text length: short text -> FAST, long text -> BRAIN;
  6. default: FAST, since it optimizes for latency.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from api.schemas.common import ExecutionTarget, RequestType
from core.config import Settings


def _contains_keyword(text: str, keyword: str) -> bool:
    """Whole-word/phrase match so short keywords (e.g. "hi", "ok") don't
    false-positive inside unrelated words (e.g. "think", "book")."""
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, text) is not None


@dataclass
class RoutingDecision:
    target: ExecutionTarget
    reason: str


class CoreAiRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def decide(self, *, request_type: RequestType, preferred_target: ExecutionTarget, text: str) -> RoutingDecision:
        if preferred_target == ExecutionTarget.FAST:
            return RoutingDecision(ExecutionTarget.FAST, "explicit_preferred_target")
        if preferred_target == ExecutionTarget.BRAIN:
            return RoutingDecision(ExecutionTarget.BRAIN, "explicit_preferred_target")

        if request_type == RequestType.COMPLEX:
            return RoutingDecision(ExecutionTarget.BRAIN, "request_type_complex")
        if request_type == RequestType.COMMAND:
            return RoutingDecision(ExecutionTarget.FAST, "request_type_command")

        lowered = text.lower()
        for kw in self.settings.router_brain_keyword_list:
            if kw and _contains_keyword(lowered, kw):
                return RoutingDecision(ExecutionTarget.BRAIN, f"brain_keyword:{kw}")
        for kw in self.settings.router_fast_keyword_list:
            if kw and _contains_keyword(lowered, kw):
                return RoutingDecision(ExecutionTarget.FAST, f"fast_keyword:{kw}")

        if len(text) <= self.settings.router_fast_max_chars:
            return RoutingDecision(ExecutionTarget.FAST, "short_text")

        return RoutingDecision(ExecutionTarget.BRAIN, "long_text")
