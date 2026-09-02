# RagService (future phase)

Not implemented yet. `future/rag/service.py` sketches the intended
interface only — it is not wired into the request path.

Planned responsibilities:
- indexing local documents into a vector store;
- retrieving relevant chunks for a request and injecting them into the
  prompt built by `RequestOrchestrator._build_prompt`;
- source citation in `JarvisResponse`.
