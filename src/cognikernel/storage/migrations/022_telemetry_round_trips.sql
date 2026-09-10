-- Migration 022: round-trip counters on api_telemetry + usage basis (S4 T-403).
--
-- responses / memory_tool_responses / denied_responses / retried_denials count the
-- API round-trips a session made, and the ones CogniKernel's own tool surface added:
-- responses whose only tool calls were CogniKernel memory tools, reads the PreToolUse
-- gate denied, and retries of those denied calls. Measured on the four-project
-- benchmark, those explained more than all of Relay's cost over an agent with no
-- memory, so they are what G1 is judged on.
--
-- usage_basis records how a row's token columns were counted. Rows ingested before
-- this migration summed usage once per transcript LINE; Claude Code writes one line
-- per content block and repeats the response's usage on each, so those rows are
-- inflated 2-3x. The DEFAULT labels every existing row 'per_line_legacy' so doctor
-- keeps them out of corrected aggregates. Re-ingesting a session
-- (`cognikernel telemetry`) rewrites its row as 'per_response'; a row whose transcript
-- is gone keeps its label rather than a guessed correction.
ALTER TABLE api_telemetry ADD COLUMN responses             INTEGER NOT NULL DEFAULT 0;
ALTER TABLE api_telemetry ADD COLUMN memory_tool_responses INTEGER NOT NULL DEFAULT 0;
ALTER TABLE api_telemetry ADD COLUMN denied_responses      INTEGER NOT NULL DEFAULT 0;
ALTER TABLE api_telemetry ADD COLUMN retried_denials       INTEGER NOT NULL DEFAULT 0;
ALTER TABLE api_telemetry ADD COLUMN usage_basis           TEXT    NOT NULL DEFAULT 'per_line_legacy';
