"""Tests for the embedding store + cosine retrieval (numpy only, no model)."""
from __future__ import annotations

import sqlite3

import pytest

# numpy ships only with the optional `embedding` extra. Skip cleanly (not a
# collection error) when it's absent — e.g. the default lexical-only CI lane.
np = pytest.importorskip("numpy")

from cognikernel.embedding.store import cosine_matches, load_embeddings, upsert_embedding
from cognikernel.storage.migrations import run_migrations


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    run_migrations(c)
    yield c
    c.close()


def _norm(values) -> np.ndarray:
    v = np.asarray(values, dtype="float32")
    return v / np.linalg.norm(v)


class TestEmbeddingStore:
    def test_model_version_encodes_input_version(self) -> None:
        """#3: the stored version folds in the composition (input) version."""
        from cognikernel.embedding.model import EMBEDDING_INPUT_VERSION, EMBEDDING_MODEL_VERSION
        assert f"in{EMBEDDING_INPUT_VERSION}" in EMBEDDING_MODEL_VERSION

    def test_input_version_bump_invalidates_old_vectors(self, conn: sqlite3.Connection) -> None:
        """A composition change (input-version bump) takes old vectors out of the
        current version space — load_embeddings filters them out so backfill
        re-embeds. Old rows remain retrievable only under their own version."""
        from cognikernel.embedding.model import EMBEDDING_MODEL_VERSION
        stale = "bge-small-en-v1.5+in0"  # a prior composition
        upsert_embedding(conn, 1, _norm([1.0, 0.0]), stale)
        assert load_embeddings(conn, [1], EMBEDDING_MODEL_VERSION) == {}
        assert 1 in load_embeddings(conn, [1], stale)

    def test_roundtrip(self, conn: sqlite3.Connection) -> None:
        v = _norm([1.0, 0.0, 0.0, 0.0])
        upsert_embedding(conn, 1, v, "m1")
        got = load_embeddings(conn, [1], "m1")
        assert 1 in got
        assert np.allclose(got[1], v, atol=1e-6)

    def test_replace_on_reupsert(self, conn: sqlite3.Connection) -> None:
        upsert_embedding(conn, 1, _norm([1.0, 0.0]), "m1")
        upsert_embedding(conn, 1, _norm([0.0, 1.0]), "m1")
        got = load_embeddings(conn, [1], "m1")
        assert np.allclose(got[1], _norm([0.0, 1.0]), atol=1e-6)

    def test_model_version_filter(self, conn: sqlite3.Connection) -> None:
        upsert_embedding(conn, 1, _norm([1.0, 0.0]), "m1")
        assert load_embeddings(conn, [1], "m2") == {}
        assert 1 in load_embeddings(conn, [1], "m1")

    def test_none_vector_is_noop(self, conn: sqlite3.Connection) -> None:
        upsert_embedding(conn, 1, None, "m1")
        assert load_embeddings(conn, [1]) == {}

    def test_cosine_matches_threshold(self, conn: sqlite3.Connection) -> None:
        query = _norm([1.0, 0.0])
        candidates = {1: _norm([0.9, 0.1]), 2: _norm([0.0, 1.0])}
        matches = cosine_matches(query, candidates, threshold=0.8)
        assert 1 in matches and matches[1] >= 0.8
        assert 2 not in matches

    def test_cosine_matches_empty(self, conn: sqlite3.Connection) -> None:
        assert cosine_matches(None, {1: _norm([1.0, 0.0])}, 0.5) == {}
        assert cosine_matches(_norm([1.0, 0.0]), {}, 0.5) == {}
