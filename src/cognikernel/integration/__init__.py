"""Integration layer public API.

Lazy re-export (PEP 562): the convenience names below import `session` on demand,
so importing a lightweight submodule like `cli` — the `python -m cognikernel hook-*`
hot path — does NOT pull the `session` stack. `from cognikernel.integration import
init_project` still works exactly as before. (CK-6a)
"""
from __future__ import annotations

__all__ = ["init_project", "session_end", "get_projection", "render_state"]


def __getattr__(name: str):  # PEP 562
    if name in __all__:
        from cognikernel.integration import session
        return getattr(session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
