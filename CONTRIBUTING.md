# Contributing — Definition of Done & promotion gate

> Status: engineering discipline reference. Code and CI are the ground truth.

This file exists because of a specific failure the merge-readiness audit surfaced:
for a long time "it works" meant *recall scored well on the benchmark and the
unit tests a developer happened to run locally were green*. Both were true while
the system had untested failure modes (crash-replay drift, an illusory hook
timeout, a non-atomic migration) and a decayed dependency graph (an
extraction⇄delta import cycle) — because **none of those things were measured**,
and the one tool that measured architecture (`import-linter`) had silently died
on a typo. Recall-high + tests-green is necessary but **not** a definition of
done.

## Definition of Done

A change is not "done" until all of the following hold. This is what the CI
promotion gate enforces (`.github/workflows/ci.yml`).

1. **Tests green** — `uv run pytest` passes, including:
   - the unit suite for the changed area,
   - `tests/reliability/` (failure-injection) if the change touches the worker,
     merge, migrations, hooks, or locks,
   - `tests/unit/test_architecture_contracts.py` (the contract meta-guard).
2. **Architecture clean** — `uv run lint-imports` reports all contracts **kept**
   (0 broken) and actually *evaluated* them. A new module must land in the right
   layer; a new cross-layer import must be justified or refactored, never
   baselined to force green.
3. **Failure modes have a test** — if the change adds a path that can fail
   (a crash window, a degraded dependency, a partial write, a timeout, a
   read-only/relocated filesystem), there is a test that exercises that path,
   not just the happy path. Add it to `tests/reliability/` where it fits.
4. **Degradation is legible** — any new fail-open branch (a swallowed exception,
   an empty-result fallback) logs at `WARNING` and, where it represents a
   subsystem that can be unhealthy, is reflected in `cognikernel doctor`
   (`src/cognikernel/integration/health.py`). Silence must not read as success.
5. **Migrations are atomic** — a new `NNN_*.sql` migration must be safe to crash
   mid-script (the runner wraps body + version bump in one transaction; the
   migration must not contain its own `BEGIN`/`COMMIT` or transaction-incompatible
   statements like `PRAGMA`/`VACUUM`).

## The guardrails, and why each exists

| Guardrail | Command | The failure it prevents |
| --- | --- | --- |
| Import-layering | `uv run lint-imports` | Architectural drift / import cycles accumulating unseen. |
| Contract meta-guard | `pytest tests/unit/test_architecture_contracts.py` | A typo'd contract silently disabling **all** enforcement (audit P2). |
| Reliability suite | `pytest tests/reliability/` | Crash/contention/corrupt-input bugs that the happy-path suite structurally cannot reach (audit P1). |
| Diagnostic spine | `cognikernel doctor --strict` | Fail-open degradation looking identical to healthy (audit P3). |

## Pre-flight (local)

Before opening a PR, the quick local equivalent of the gate:

```sh
uv run lint-imports
uv run pytest -q
uv run python -m cognikernel doctor --strict <a-real-project-path>   # optional: subsystem health
```

`doctor --strict` exits non-zero if any subsystem (schema version, FTS5,
embedding model, symbol extraction, worker queue) is degraded — useful before a
benchmark run so a silent gap doesn't get read as a result.

## Beyond the gate: periodic external audit

The gate catches regressions in *known* categories. It does not replace a
periodic adversarial pass with a merge-readiness mandate (not a feature-works
mandate) — that is what found the issues this discipline now guards against.
Builders validating their own features will reliably miss "is the foundation
still sound." Schedule that review; do not assume green CI subsumes it.
