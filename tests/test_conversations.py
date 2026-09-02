from __future__ import annotations

import pytest

from conversations.store import InMemoryConversationStore, Message


@pytest.mark.asyncio
async def test_append_and_get_history():
    store = InMemoryConversationStore(max_messages=20, max_chars=10_000)
    await store.append("conv-1", Message(role="user", content="hello"))
    await store.append("conv-1", Message(role="assistant", content="hi there"))
    history = await store.get_history("conv-1")
    assert [m.content for m in history] == ["hello", "hi there"]


@pytest.mark.asyncio
async def test_conversation_isolation():
    store = InMemoryConversationStore(max_messages=20, max_chars=10_000)
    await store.append("conv-a", Message(role="user", content="from A"))
    await store.append("conv-b", Message(role="user", content="from B"))
    history_a = await store.get_history("conv-a")
    history_b = await store.get_history("conv-b")
    assert [m.content for m in history_a] == ["from A"]
    assert [m.content for m in history_b] == ["from B"]


@pytest.mark.asyncio
async def test_clear_removes_only_target_conversation():
    store = InMemoryConversationStore(max_messages=20, max_chars=10_000)
    await store.append("conv-a", Message(role="user", content="keep me"))
    await store.append("conv-b", Message(role="user", content="clear me"))
    await store.clear("conv-b")
    assert await store.get_history("conv-b") == []
    assert len(await store.get_history("conv-a")) == 1


@pytest.mark.asyncio
async def test_message_count_limit_trims_oldest():
    store = InMemoryConversationStore(max_messages=3, max_chars=10_000)
    for i in range(5):
        await store.append("conv-1", Message(role="user", content=f"msg-{i}"))
    history = await store.get_history("conv-1")
    assert len(history) == 3
    assert [m.content for m in history] == ["msg-2", "msg-3", "msg-4"]


@pytest.mark.asyncio
async def test_char_limit_trims_oldest():
    store = InMemoryConversationStore(max_messages=100, max_chars=25)
    await store.append("conv-1", Message(role="user", content="a" * 10))
    await store.append("conv-1", Message(role="user", content="b" * 10))
    await store.append("conv-1", Message(role="user", content="c" * 10))
    history = await store.get_history("conv-1")
    total_chars = sum(len(m.content) for m in history)
    assert total_chars <= 30  # trimmed to stay near the char budget
    assert history[-1].content == "c" * 10
