# ContextEngine (future phase)

Not implemented yet. `future/context/service.py` sketches the intended
interface only — it is not wired into the request path.

Planned responsibilities:
- merging device context (location, notifications, active app, calendar...)
  sent by Android in `JarvisRequest.context` with server-side signals;
- deciding what context is relevant enough to include in a given prompt.
