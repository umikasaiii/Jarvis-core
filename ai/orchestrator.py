"""RequestOrchestrator: the glue between the API layer and AI Router / LLM Engine.

    Android -> API Layer -> RequestOrchestrator -> CoreAiRouter -> LlmProvider -> streaming

The orchestrator is the one place that knows about conversation history,
routing, the inference queue, and provider errors together, so route
handlers stay thin (validate + call orchestrator + shape HTTP response).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from api.schemas.common import ExecutionTarget, FinishReason, ResponseStatus, StreamEventType
from api.schemas.request import JarvisRequest
from api.schemas.response import JarvisResponse
from api.schemas.stream import StreamEvent
from ai.model_manager import ModelManager
from ai.queue import InferenceQueue, QueueFullError
from ai.router import CoreAiRouter
from conversations.store import ConversationStore, Message
from core.config import Settings, PROTOCOL_VERSION
from core.logging import Timer, get_logger, log_event
from providers.base import LlmProviderError

logger = get_logger("jarvis.orchestrator")


class ProtocolVersionError(ValueError):
    def __init__(self, received: str) -> None:
        super().__init__(
            f"Unsupported protocolVersion '{received}', server expects '{PROTOCOL_VERSION}'"
        )
        self.received = received


class RequestOrchestrator:
    def __init__(
        self,
        settings: Settings,
        model_manager: ModelManager,
        router: CoreAiRouter,
        queue: InferenceQueue,
        conversation_store: ConversationStore,
        system_prompt: str,
    ) -> None:
        self.settings = settings
        self.model_manager = model_manager
        self.router = router
        self.queue = queue
        self.conversation_store = conversation_store
        self.system_prompt = system_prompt

    @staticmethod
    def validate_protocol_version(request: JarvisRequest) -> None:
        if request.protocolVersion != PROTOCOL_VERSION:
            raise ProtocolVersionError(request.protocolVersion)

    async def _build_prompt(self, request: JarvisRequest) -> str:
        if not request.conversationId:
            return request.text
        history = await self.conversation_store.get_history(request.conversationId)
        if not history:
            return request.text
        lines = [f"{m.role}: {m.content}" for m in history]
        lines.append(f"user: {request.text}")
        return "\n".join(lines)

    async def handle_request(
        self, request: JarvisRequest, *, persist_history: bool = True
    ) -> JarvisResponse:
        self.validate_protocol_version(request)
        timer = Timer()

        decision = self.router.decide(
            request_type=request.requestType,
            preferred_target=request.preferredTarget,
            text=request.text,
        )
        target = decision.target
        warnings: list[str] = []

        try:
            provider, target, warnings = await self._resolve_provider(request, decision)
        except LlmProviderError as exc:
            return JarvisResponse(
                requestId=request.requestId,
                status=ResponseStatus.ERROR,
                targetUsed=target,
                executionTimeMs=timer.elapsed_ms(),
                finishReason=FinishReason.ERROR,
                error=str(exc),
            )

        prompt = await self._build_prompt(request)

        try:
            async with self.queue.slot():
                try:
                    result = await asyncio.wait_for(
                        provider.generate(prompt, system_prompt=self._resolve_system_prompt(request)),
                        timeout=self.settings.request_timeout,
                    )
                except asyncio.TimeoutError:
                    return JarvisResponse(
                        requestId=request.requestId,
                        status=ResponseStatus.ERROR,
                        targetUsed=target,
                        modelUsed=self.model_manager.active_model_name(target),
                        executionTimeMs=timer.elapsed_ms(),
                        finishReason=FinishReason.TIMEOUT,
                        error="Request timed out",
                        warnings=warnings,
                    )
        except QueueFullError as exc:
            return JarvisResponse(
                requestId=request.requestId,
                status=ResponseStatus.ERROR,
                targetUsed=target,
                executionTimeMs=timer.elapsed_ms(),
                finishReason=FinishReason.ERROR,
                error=str(exc),
                warnings=warnings,
            )
        except LlmProviderError as exc:
            return JarvisResponse(
                requestId=request.requestId,
                status=ResponseStatus.ERROR,
                targetUsed=target,
                executionTimeMs=timer.elapsed_ms(),
                finishReason=FinishReason.ERROR,
                error=str(exc),
                warnings=warnings,
            )

        if persist_history and request.conversationId:
            await self.conversation_store.append(
                request.conversationId, Message(role="user", content=request.text)
            )
            await self.conversation_store.append(
                request.conversationId, Message(role="assistant", content=result.text)
            )

        elapsed = timer.elapsed_ms()
        log_event(
            logger,
            20,
            "request_completed",
            requestId=request.requestId,
            requestType=request.requestType.value,
            target=target.value,
            model=self.model_manager.active_model_name(target),
            latencyMs=round(elapsed, 2),
            tokens=result.tokens_generated,
            success=True,
        )

        return JarvisResponse(
            requestId=request.requestId,
            status=ResponseStatus.OK,
            text=result.text,
            modelUsed=self.model_manager.active_model_name(target),
            targetUsed=target,
            executionTimeMs=elapsed,
            tokensGenerated=result.tokens_generated,
            finishReason=FinishReason(result.finish_reason)
            if result.finish_reason in FinishReason._value2member_map_
            else FinishReason.STOP,
            warnings=warnings,
        )

    def _resolve_system_prompt(self, request: JarvisRequest) -> str:
        """A per-request `systemPrompt` (v1.1.0, additive) wins over the
        server's own default when present and non-blank; every existing
        client that never sends it observes exactly the prior behavior
        (self.system_prompt), unchanged."""
        override = request.systemPrompt
        if override and override.strip():
            return override
        return self.system_prompt

    async def _resolve_provider(self, request: JarvisRequest, decision):
        target = decision.target
        warnings: list[str] = []
        try:
            await self.model_manager.ensure_loaded(target)
            return self.model_manager.get_provider(target), target, warnings
        except LlmProviderError as exc:
            if not request.allowFallback:
                raise
            fallback = ExecutionTarget.FAST if target == ExecutionTarget.BRAIN else ExecutionTarget.BRAIN
            warnings.append(f"{target.value} unavailable ({exc}); falling back to {fallback.value}")
            await self.model_manager.ensure_loaded(fallback)
            return self.model_manager.get_provider(fallback), fallback, warnings

    async def handle_stream(
        self, request: JarvisRequest, *, persist_history: bool = True
    ) -> AsyncIterator[StreamEvent]:
        self.validate_protocol_version(request)
        timer = Timer()

        decision = self.router.decide(
            request_type=request.requestType,
            preferred_target=request.preferredTarget,
            text=request.text,
        )

        try:
            provider, target, warnings = await self._resolve_provider(request, decision)
        except LlmProviderError as exc:
            yield StreamEvent(
                type=StreamEventType.ERROR, requestId=request.requestId, error=str(exc)
            )
            return

        yield StreamEvent(
            type=StreamEventType.START, requestId=request.requestId, targetUsed=target
        )

        prompt = await self._build_prompt(request)
        full_text_parts: list[str] = []
        tokens_generated = 0
        finish_reason = FinishReason.STOP

        try:
            async with self.queue.slot():
                stream_iter = provider.stream(prompt, system_prompt=self._resolve_system_prompt(request))
                async for chunk in _with_timeout(stream_iter, self.settings.request_timeout):
                    if chunk.content:
                        full_text_parts.append(chunk.content)
                        yield StreamEvent(
                            type=StreamEventType.TOKEN,
                            requestId=request.requestId,
                            content=chunk.content,
                        )
                    if chunk.tokens_generated is not None:
                        tokens_generated = chunk.tokens_generated
                    if chunk.done:
                        if chunk.finish_reason in FinishReason._value2member_map_:
                            finish_reason = FinishReason(chunk.finish_reason)
                        break
        except asyncio.CancelledError:
            log_event(
                logger, 20, "stream_cancelled", requestId=request.requestId, target=target.value
            )
            raise
        except asyncio.TimeoutError:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                requestId=request.requestId,
                error="Request timed out",
            )
            return
        except QueueFullError as exc:
            yield StreamEvent(type=StreamEventType.ERROR, requestId=request.requestId, error=str(exc))
            return
        except LlmProviderError as exc:
            yield StreamEvent(type=StreamEventType.ERROR, requestId=request.requestId, error=str(exc))
            return

        full_text = "".join(full_text_parts)
        if persist_history and request.conversationId:
            await self.conversation_store.append(
                request.conversationId, Message(role="user", content=request.text)
            )
            await self.conversation_store.append(
                request.conversationId, Message(role="assistant", content=full_text)
            )

        elapsed = timer.elapsed_ms()
        log_event(
            logger,
            20,
            "stream_completed",
            requestId=request.requestId,
            requestType=request.requestType.value,
            target=target.value,
            model=self.model_manager.active_model_name(target),
            latencyMs=round(elapsed, 2),
            tokens=tokens_generated,
            success=True,
        )

        yield StreamEvent(
            type=StreamEventType.DONE,
            requestId=request.requestId,
            modelUsed=self.model_manager.active_model_name(target),
            targetUsed=target,
            executionTimeMs=elapsed,
            tokensGenerated=tokens_generated,
            finishReason=finish_reason,
        )


async def _with_timeout(aiter: AsyncIterator, timeout: float) -> AsyncIterator:
    """Apply an overall timeout to consuming an async iterator, so a stuck
    provider can never hold the inference queue slot forever."""
    it = aiter.__aiter__()
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        try:
            item = await asyncio.wait_for(it.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            return
        yield item
