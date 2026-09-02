from __future__ import annotations

from api.schemas.common import ExecutionTarget, RequestType
from ai.router import CoreAiRouter
from core.config import get_settings


def _router():
    return CoreAiRouter(get_settings())


def test_explicit_fast_wins(monkeypatch):
    monkeypatch.setenv("FAST_MODEL_BACKEND", "fake")
    get_settings.cache_clear()
    router = _router()
    decision = router.decide(
        request_type=RequestType.COMPLEX,
        preferred_target=ExecutionTarget.FAST,
        text="a very long and complex reasoning question " * 10,
    )
    assert decision.target == ExecutionTarget.FAST
    get_settings.cache_clear()


def test_explicit_brain_wins(monkeypatch):
    get_settings.cache_clear()
    router = _router()
    decision = router.decide(
        request_type=RequestType.COMMAND, preferred_target=ExecutionTarget.BRAIN, text="ciao"
    )
    assert decision.target == ExecutionTarget.BRAIN
    get_settings.cache_clear()


def test_complex_request_type_routes_brain():
    get_settings.cache_clear()
    router = _router()
    decision = router.decide(
        request_type=RequestType.COMPLEX, preferred_target=ExecutionTarget.AUTO, text="hi"
    )
    assert decision.target == ExecutionTarget.BRAIN
    get_settings.cache_clear()


def test_command_request_type_routes_fast():
    get_settings.cache_clear()
    router = _router()
    decision = router.decide(
        request_type=RequestType.COMMAND, preferred_target=ExecutionTarget.AUTO, text="turn on the light"
    )
    assert decision.target == ExecutionTarget.FAST
    get_settings.cache_clear()


def test_short_text_defaults_to_fast():
    get_settings.cache_clear()
    router = _router()
    decision = router.decide(
        request_type=RequestType.CHAT, preferred_target=ExecutionTarget.AUTO, text="What time is it?"
    )
    assert decision.target == ExecutionTarget.FAST
    get_settings.cache_clear()


def test_long_text_defaults_to_brain():
    get_settings.cache_clear()
    router = _router()
    long_text = "Please help me think through this problem. " * 20
    decision = router.decide(
        request_type=RequestType.CHAT, preferred_target=ExecutionTarget.AUTO, text=long_text
    )
    assert decision.target == ExecutionTarget.BRAIN
    get_settings.cache_clear()


def test_brain_keyword_overrides_short_text():
    get_settings.cache_clear()
    router = _router()
    decision = router.decide(
        request_type=RequestType.CHAT, preferred_target=ExecutionTarget.AUTO, text="analizza questo"
    )
    assert decision.target == ExecutionTarget.BRAIN
    get_settings.cache_clear()
