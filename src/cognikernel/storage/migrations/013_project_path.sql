-- Migration 013: store the project's resolved absolute path in meta.
-- Enables reverse lookup from project_id → path for resource discovery
-- (cognikernel://projects lists all known projects with their human-readable paths).
-- Written by init_project() and session_end(). Existing projects get an empty
-- string until they next run init or a session ends.
INSERT OR IGNORE INTO meta (key, value) VALUES ('project_path', '');
