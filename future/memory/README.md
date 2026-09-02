# MemoryService (future phase)

Not implemented yet. `future/memory/service.py` sketches the intended
interface only — it is not wired into the request path.

Planned responsibilities:
- long-term facts/preferences about the user, persisted across restarts;
- retrieval of relevant memories to inject into `context` before routing;
- write-back of new memories extracted from conversations.

Will plug in as a `ConversationStore`-compatible `PersistentMemoryStore`
(see `conversations/store.py`) plus a dedicated `MemoryService`, without
changing the `/v1/chat` or `/v1/ai/*` request/response contracts.
