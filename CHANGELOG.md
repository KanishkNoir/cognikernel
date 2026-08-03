# Changelog

All notable changes to CogniKernel are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.1] — unreleased

### Fixed

- **TypeScript/JavaScript symbol extraction was dead for every `pip install`.**
  `tree-sitter-language-pack` changed its binding at 1.13.0 — the native
  `parse_bytes()` / `root_node()` / `kind()` *methods* became py-tree-sitter
  `parse()` / `.root_node` / `.type` *properties*. `pyproject.toml` declared an
  unbounded `>=1.0`, so a fresh install resolved 1.14.0 and every accessor
  raised `AttributeError`. Extraction fails open by design, so the failure was
  silent: `typescript_support_status()` returned `False` and valid TypeScript
  produced **0 nodes, 0 edges**. Our own environment was pinned to 1.9.1 by
  `uv.lock`, which is why CI never saw it.

  Extraction now routes through adapters that accept **both** binding shapes,
  so no one on an older pack is stranded. Verified producing identical output
  on 1.9.1 and 1.14.0.

- **Grounding penalised languages the symbol walk cannot parse.** The path
  inventory is built from a discovery walk that globs only
  `.py/.ts/.tsx/.js/.jsx`, so a real `internal/db/pool.go` was absent for
  reasons unrelated to whether it exists — and was marked `unverified` with its
  weight halved. Every component event in a Go, Rust, Java, or C# project would
  have been pushed off the budget-ranked block. Grounding now applies only to
  file types the inventory actually covers.

- **File paths written as dotfiles, `./`-relative, absolute, or with Windows
  separators were invisible.** The path pattern's lookbehind blocked any path
  preceded by `.`, `/`, or `\`, and its body class carried no backslash, so
  `.claude/settings.json`, `./src/foo.py`, `/srv/app/main.py`, and every
  `C:\...\file.py` produced no match at all. Absolute paths additionally
  required a project root to resolve, which is now threaded through.

- **Catastrophic backtracking in the path pattern.** A run such as
  `a./a./a./…` with no valid extension backtracked exponentially — 3.3 s at 22
  repetitions and climbing, enough to stall extraction on a hostile transcript.
  The pre-existing pattern shared the flaw. Now linear, with regression tests.

- **Descriptions beginning with a file path were corrupted** by sentence-case
  capitalisation rewriting `src/…` to `Src/…`, which then matched nothing in
  the codebase.

- **Vendored and cached files crowded out first-party code in the skeleton.**
  Discovery now respects `.gitignore`, skips `.uv-cache`, `.claude`,
  `.pytest_tmp`, `.ruff_cache`, `target`, `vendor` and similar, and ranks
  candidates so project code wins the file budget rather than walk order
  deciding. Stale out-of-scope nodes are pruned from existing graphs. Measured
  on this repository: the symbol graph went from 4,655 of 4,934 nodes (94%)
  vendored to zero, and the Codebase skeleton section changed from describing
  the `attrs` library to describing the project.

### Added

- **Quality gate on the event write path.** Every extracted event now passes
  through admission control before storage, returning admit / downgrade /
  reject. Harness and compaction boilerplate is rejected outright; statements
  whose subject exists only in unstated context are downgraded so they fall off
  the ranked block while remaining reachable through `recall` and
  `find_related`. Measured across 8,840 real stored statements: 0.42% rejected,
  3.98% downgraded.

- **Path grounding** — referential integrity between stored memory and the
  actual codebase. Unknown paths are downgraded rather than dropped, since new
  files are legitimate; only near-miss truncations of a known path are
  rejected.

- **`quality_telemetry` table** (schema v19) recording per-rule gate activity,
  surfaced by `cognikernel doctor`.

- **Render-time structural invariants** — box-drawing artifacts and cross-type
  duplicate statements are filtered from the injected block. The filter runs
  over the event set before assembly so the render ledger stays accurate.

### Notes

- Multi-language symbol extraction (Go, Rust, Java) is **designed but not
  shipped** in this release. Those files are tracked in the component map but
  produce no symbols.
- Schema migrates from v18 to v19 automatically on first open. Existing stores
  are not rewritten; the fixes prevent new defects rather than repairing old
  ones, and existing low-quality entries decay out.

---

## [0.1.0] — 2026-07

Initial public release.
