-- Migration 002: symbol graph tables
-- Stores per-symbol AST nodes and cross-file import edges.
-- Language-agnostic schema; only the extractor layer is language-specific.

CREATE TABLE IF NOT EXISTS symbol_nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    path        TEXT    NOT NULL,
    node_type   TEXT    NOT NULL CHECK (node_type IN ('class', 'function', 'method', 'import')),
    name        TEXT    NOT NULL,
    parent_name TEXT    NOT NULL DEFAULT '',
    signature   TEXT    NOT NULL DEFAULT '',
    return_type TEXT    NOT NULL DEFAULT '',
    fields      TEXT    NOT NULL DEFAULT '',
    updated_at  INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_symbol_nodes_unique
    ON symbol_nodes (project_id, path, node_type, name, parent_name);

CREATE INDEX IF NOT EXISTS idx_symbol_nodes_path
    ON symbol_nodes (project_id, path);

CREATE TABLE IF NOT EXISTS symbol_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    from_path   TEXT    NOT NULL,
    to_path     TEXT    NOT NULL,
    edge_type   TEXT    NOT NULL DEFAULT 'imports'
                        CHECK (edge_type IN ('imports', 'extends', 'implements')),
    is_external INTEGER NOT NULL DEFAULT 0
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_symbol_edges_unique
    ON symbol_edges (project_id, from_path, to_path, edge_type);

CREATE INDEX IF NOT EXISTS idx_symbol_edges_from
    ON symbol_edges (project_id, from_path);
