"""Local embedding model wrapper — lazy, cached, optional.

Uses fastembed (ONNX runtime, no torch) with a small English model. The model
is loaded once on first use and cached for the process. If fastembed is not
installed or the model can't load, every entrypoint degrades to None so callers
fall back to lexical matching rather than crashing.
"""
from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=1)
def _model() -> Any | None:
    try:
        from fastembed import TextEmbedding
        return TextEmbedding(model_name=_MODEL_NAME)
    except Exception:
        return None


def is_available() -> bool:
    """True when the embedding model is importable + loadable."""
    return _model() is not None


def embed_text(text: str):
    """Return an L2-normalized float32 vector for `text`, or None if unavailable.

    Normalizing at the source means cosine similarity is a plain dot product
    downstream. Empty/whitespace text returns None (nothing to embed).
    """
    if not text or not text.strip():
        return None
    model = _model()
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
