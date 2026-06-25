"""Local semantic embedding layer.

A small, local, offline embedding model (via fastembed/ONNX — no API, no network
at inference once cached) turns event descriptions into vectors. These power
*semantic* candidate retrieval for supersession/dedup — catching decisions that
mean the same thing in different words (semantically near but lexically distinct)
that the lexical Jaccard/Levenshtein path misses.

The model is OPTIONAL: if fastembed is not installed (or fails to load),
`embed_text` returns None and every consumer degrades to the existing lexical
behavior. Embeddings are computed on the write path (session_end), never on the
render hot path.
"""
from __future__ import annotations

from memlora.embedding.backfill import backfill_embeddings
from memlora.embedding.input import embedding_input
from memlora.embedding.model import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_VERSION,
    embed_text,
    ensure_ready,
    is_available,
    is_ready,
    warm,
)
from memlora.embedding.retrieval import find_related, recall
from memlora.embedding.store import (
    cosine_matches,
    load_embeddings,
    upsert_embedding,
)

__all__ = [
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL_VERSION",
    "embed_text",
    "is_available",
    "is_ready",
    "warm",
    "ensure_ready",
    "embedding_input",
    "cosine_matches",
    "load_embeddings",
    "upsert_embedding",
    "recall",
    "find_related",
    "backfill_embeddings",
]
