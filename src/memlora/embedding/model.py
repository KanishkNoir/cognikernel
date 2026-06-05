"""Local embedding model wrapper — lazy, background-loaded, persistently cached.

Uses fastembed (ONNX runtime, no torch) with a small English model. Two problems
this module exists to prevent:

  1. Re-downloading every session. fastembed's default cache is the OS temp dir,
     which is ephemeral — the ~130MB model would re-fetch (minutes) each run. We
     pin a PERSISTENT cache under the memlora data dir so it downloads once, ever.

  2. Hanging an interactive path on a cold start. The first fetch+load takes
     minutes; the MCP `recall` tool and the per-prompt hook must NOT block on it.
     The model loads in a single background thread; hot-path callers gate on the
     NON-BLOCKING `is_ready()` and fall back to lexical until it's ready, while the
     download proceeds and a later call goes semantic.

Two readiness checks, deliberately distinct:
  - is_ready()      — non-blocking; True only if loaded right now. HOT-PATH gate.
  - is_available()  — loads now if needed (may block once); "is the model usable
                      at all". For test skipif gates, backfill, explicit warmup.

If fastembed is not installed or the model can't load, every entrypoint degrades
to None / not-ready rather than crashing.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_MODEL_TAG = "bge-small-en-v1.5"

# Bump whenever `embedding.input.embedding_input` changes its composition. Stored
# vectors encode the composition that produced them, so a composition change must
# invalidate them — otherwise old vectors silently sit in a different input space
# than new queries. Folding this into the stored `model_version` means the
# existing staleness machinery (load_embeddings filter, backfill LEFT JOIN,
# retrieval WHERE model_version=?) treats a composition bump exactly like a model
# swap: old-version rows stop matching and backfill re-embeds them in place
# (event_id is the PK, so the re-embed REPLACEs — no orphans).
EMBEDDING_INPUT_VERSION = 1
EMBEDDING_MODEL_VERSION = f"{_MODEL_TAG}+in{EMBEDDING_INPUT_VERSION}"
EMBEDDING_DIM = 384

# How long an interactive caller (embed_text -> recall / per-prompt injection)
# waits for the model before giving up. Short on purpose: once the background load
# finishes, ensure_ready returns instantly.
_EMBED_WAIT_S = 2.0

_lock = threading.Lock()
_thread: threading.Thread | None = None
_model_obj: Any | None = None
_load_done = False


def _cache_dir() -> str:
    """Persistent fastembed cache so the model downloads once, not per session.

    Honors MEMLORA_DIR (tests/CI redirect the data dir there); otherwise ~/.memlora.
    """
    base = os.environ.get("MEMLORA_DIR") or str(Path.home() / ".memlora")
    d = Path(base) / "models"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return str(d)


def _load() -> Any | None:
    try:
        # Quiet the Windows "no symlink support" hub warning — harmless, just noise.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=_MODEL_NAME, cache_dir=_cache_dir())
    except Exception:
        return None


def _start_load() -> threading.Thread | None:
    """Spawn the single background loader thread (idempotent, unguarded)."""
    global _thread
    with _lock:
        if _thread is None and not _load_done:

            def _run() -> None:
                global _model_obj, _load_done
                m = _load()
                with _lock:
                    _model_obj = m
                    _load_done = True

            _thread = threading.Thread(target=_run, name="ck-embed-load", daemon=True)
            _thread.start()
        return _thread


def warm() -> None:
    """Background kick for implicit callers (MCP server startup, hooks, recall).

    Never blocks. Suppressed by MEMLORA_DISABLE_AUTO_WARM=1 (set by tests/CI in
    conftest) so the unit suite never triggers a background model download.
    """
    if os.environ.get("MEMLORA_DISABLE_AUTO_WARM"):
        return
    _start_load()


def ensure_ready(timeout: float | None = None) -> bool:
    """Wait up to `timeout` seconds for the model. True if ready.

    An EXPLICIT request — not suppressed by the auto-warm guard — for callers that
    ask to wait: `memlora warm`, is_available(), real-model test gates. `timeout=None`
    blocks until the load finishes (use only off the hot path). The background load
    continues regardless of the timeout, so a later call succeeds once it completes.
    """
    with _lock:
        if _load_done:
            return _model_obj is not None
    _start_load()
    with _lock:
        t = _thread
    if t is not None:
        t.join(timeout)
    with _lock:
        return _load_done and _model_obj is not None


def is_ready() -> bool:
    """True only if the model is loaded and ready RIGHT NOW. Never blocks, never loads.

    The gate for interactive callers (MCP recall, per-prompt injection): if the
    model is still downloading or failed to load, the caller falls back to lexical.
    """
    with _lock:
        return _load_done and _model_obj is not None


def is_available() -> bool:
    """True if the model can be loaded — loading it now if needed (may block on the
    first call while it downloads/initialises). For non-latency-sensitive callers:
    test skipif gates, backfill, explicit warmup. Hot paths must use is_ready().
    """
    return ensure_ready(timeout=None)


def embed_text(text: str):
    """Return an L2-normalized float32 vector for `text`, or None if unavailable.

    Normalizing at the source means cosine similarity is a plain dot product
    downstream. Empty/whitespace text returns None. Under the auto-warm guard
    (tests/CI) it never starts a background download — it embeds only if the model
    is already resident.
    """
    if not text or not text.strip():
        return None
    if os.environ.get("MEMLORA_DISABLE_AUTO_WARM"):
        if not is_ready():
            return None
    elif not ensure_ready(_EMBED_WAIT_S):
        return None
    with _lock:
        model = _model_obj
    if model is None:
        return None
    try:
        import numpy as np

        vec = next(iter(model.embed([text])))
        v = np.asarray(vec, dtype="float32")
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v
    except Exception:
        return None
