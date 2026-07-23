-- Migration 012: per-event embedding store (local semantic layer).
-- One normalized embedding vector per event, computed at session_end when the
-- embedding feature is enabled. Stored as raw float32 bytes; cosine == dot
-- product because vectors are L2-normalized at write time. model_version lets
-- a model swap invalidate stale vectors without a schema change.

CREATE TABLE IF NOT EXISTS event_embeddings (
    event_id      INTEGER PRIMARY KEY,
    model_version TEXT    NOT NULL,
    dim           INTEGER NOT NULL,
    vector        BLOB    NOT NULL,
    created_at    INTEGER NOT NULL
);
