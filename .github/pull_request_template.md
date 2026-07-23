## What

<!-- What does this change do, and why? -->

## Definition of Done (see CONTRIBUTING.md)

- [ ] `uv run pytest` green (including `tests/reliability/` if the change touches worker, merge, migrations, hooks, or locks)
- [ ] `uv run lint-imports` — all contracts kept and evaluated
- [ ] New failure paths have a test (not just the happy path)
- [ ] New fail-open branches log at `WARNING` and surface in `cognikernel doctor` where applicable
- [ ] New migrations are atomic (no `BEGIN`/`COMMIT`/`PRAGMA`/`VACUUM` inside the script)
