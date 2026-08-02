# Injection correctness — what, why, how, impact

**Branch:** injection correctness (cycle 1 of 3; retrieval relevance and
speed/granularity follow)
**Measured:** 2026-08-02, across 163 local project stores / 8,840 active
statements / 10,165 descriptions.
**Reproduce:** `scripts/injection_defect_audit.py`, `scripts/injection_snapshot.py`,
`scripts/path_recall_ab.py`. All three are read-only with respect to memory.

---

## What

CogniKernel's only output surface is the context block it injects at session
start. That block was carrying content that actively misled the agent — and
because the MCP instructions declare it authoritative ("supersedes CLAUDE.md"),
a defect there is worse than no memory at all.

Seven defect classes were named. **Measurement then retracted one of them and
reprioritised the rest**, which is the part worth keeping in the write-up:

| Class | Claimed from the block | What measurement found |
|---|---|---|
| D1 path corruption | headline defect, "a code path strips a leading character" | **retracted** — 17 rows, 2 stores, all on 2026-05-10, pre-rename era. No live generator. |
| D7 subject-less | not prioritised | **the real headline** — 6.5% of statements, 34 stores, rising month over month |
| D3 vendored skeleton | noted | **confirmed and worse** — one store 94% vendored nodes |
| D2 junk-as-constraint | 2.0% | rate inflated by a bad proxy; true rate lower |
| D5 cross-type duplicate | noted | confirmed, 0.9% |
| D4 boilerplate | noted | confirmed, small, 0.1% |
| encoding corruption | suspected | **dropped** — 0 replacement chars in 10,165 descriptions |

## Why

Two root causes, both structural rather than incidental.

**1. Predicates existed but were wired into the wrong paths.**
`extraction/sanitize.py` already shipped `is_context_dependent_fragment()` and
`is_question_description()`. The first ran only in the v1/v2 salience-head
paths; the default `legacy` extractor never called it. The second guarded
`CONSTRAINT_HARD` in windowing and nothing else. The capability was present and
unreachable — which is why D7 kept climbing while the code that could catch it
sat in the repo.

**2. Scan and invalidation rules were incomplete in the same direction.**
`_SKIP_DIRS` omitted `.uv-cache`, `.claude`, and `.pytest_tmp`; collection was
first-glob-wins against a 500-file cap; and `build_symbol_update` emitted
`delete_paths` only for git-*deleted* files, so a path that left scope stayed in
the graph forever. Each is small; together they let a dependency cache occupy
the largest section of the injected block.

## How

One choke point instead of scattered predicates. Every extracted event passes
`quality.gate.admit()` inside `delta.merge.execute_merge`'s candidate loop,
which returns admit / downgrade / reject.

> Finding the right seam took two attempts, and the failure mode is worth
> recording. The gate was first wired into `extraction.pipeline.persist_events`,
> which the design called "the one function every extraction path reaches." It
> has **no production callers**: `session_end`, `process_jobs`, and
> `rebuild_from_raw` all go through `execute_merge`. The gate passed every unit
> test while guarding nothing in production. A grep for callers — not a reading
> of the docstring — is what caught it.

Verdicts are graded by how recoverable the content is:

- **reject** for D2 and D4: box-drawing artifacts and harness boilerplate carry
  no project content. D4 is checked for *every* event type — end-to-end testing
  showed one compaction sentence extracted twice, with the `CONSTRAINT_HARD`
  copy rejected and the `THREAD_OPEN` copy stored, because the check had been
  scoped to statement types.
- **downgrade** for D7 and ungrounded paths: a subject-less statement still
  carries a real fact, so weight collapse removes it from the budget-ranked
  block while leaving it reachable through `recall` and `find_related`.
  Rejecting would delete recoverable memory, and would have contradicted the
  policy `pipeline.py` already states for the same class.

The researched addition is **path grounding** — referential integrity between
stored memory and the actual codebase. The inventory is assembled once per
merge from the symbol store plus a project walk, so each check is a set lookup.
Unknown paths downgrade (new files are legitimate); only near-miss truncations
of a known path are rejected. An **empty** inventory means "cannot verify", not
"nothing is real" — otherwise a brand-new project with no symbol graph would
have every component event it ever captured downgraded.

> The same dead-code trap recurred here and is worth naming, because it is the
> failure mode of this whole branch: adding a `ground` parameter is not the
> same as passing one. After the gate moved to `execute_merge`, grounding was
> still inert because no call site supplied a context. `grep` for the argument,
> not for the parameter.

`cognikernel.quality` is a leaf package with an import-linter contract forbidding
it from reaching into storage, injection, compression, integration, extraction,
or delta. The gate takes its path inventory as an argument rather than reading
the symbol store, which is what keeps that contract true.

## Impact

### Measured — skeleton composition (this repo's own store)

| Metric | Before | After | Δ |
|---|---|---|---|
| Symbol nodes | 4,934 | 2,713 | −2,221 |
| Vendored nodes | 4,655 (94.35%) | 0 (0%) | −4,655 |
| First-party paths described | ~16 | 231 | +215 |
| Block total | 1,798 tok | 1,749 tok | −49 |

The token delta is small and beside the point. **What changed is what the
budget describes.** The Codebase skeleton — the single largest section — led
with `.uv-cache/archive-v0/…/attr/_make.py` and `attr/validators.py`, the
`attrs` library's internals. It now leads with
`src/cognikernel/symbols/extractor.py`. Same tokens, real content.

### Measured — detector precision

The production detectors are deliberately more conservative than the
exploratory proxies that motivated them:

| Class | Exploratory sweep | Production detector | False positives removed |
|---|---|---|---|
| D7 | 575 (6.5%) | 352 (3.98%) | 223 |
| D2 | 64 (2.0%) | 28 (0.86%) | 36 |
| D4 | 9 (0.1%) | 9 (0.10%) | 0 |

The D2 gap is the one that mattered: the proxy flagged every leading "Do", so
well-formed imperative constraints (*"Do not cache in Redis"*) counted as
defects. Those cases are now pinned as regression tests.

### Measured — gate behaviour on 8,840 real statements

| Verdict | Rate | Stop threshold |
|---|---|---|
| reject | 0.42% | 5% |
| downgrade | 3.98% | 15% |

Safe by an order of magnitude. A gate that rejected aggressively would be worse
than the defect it fixes.

### Verified end to end

A synthetic session run through the real capture path (`extract_session` →
`execute_merge`) on a transcript containing compaction boilerplate, a
subject-less statement, and a `.claude/` path mention:

- both copies of the boilerplate rejected (`rejected: 1` → `2` after the D4
  type-independence fix; telemetry `D4: 2`)
- the subject-less statement stored at weight 0.50 and marked
  `quality: context_dependent` — demoted, still recallable
- `.claude/settings.json` captured as a component, which the old pattern could
  not match at all

### Baseline with confidence intervals

`docs/metrics/injection_defect_baseline.json`, 163 stores:

| Class | Rate | 95% Wilson CI | Stores |
|---|---|---|---|
| D7 | 3.98% | [3.59%, 4.41%] | 28 |
| D5 | 0.90% | [0.73%, 1.12%] | 20 |
| D2 | 0.86% | [0.60%, 1.24%] | 12 |
| D4 | 0.10% | [0.05%, 0.19%] | 9 |

### What this does *not* show

**Stored-event defect rates do not improve between before and after, by
design.** The branch is prevent-only: the 352 D7 and 28 D2 statements already
in the stores stay, and decay retires them. Reporting those as a before/after
column would show ~0 change and misrepresent both the method and the result.
The sweep is a *prevalence* baseline — how much defective memory exists — not
the A/B.

The full offline A/B specified in the design is also not available as scoped:
only **22 of 139** stored session ids still have a transcript on disk (15.8%).
That is why the path-recall claim is evidenced by replaying both regexes over
stored `raw_evidence` instead — deterministic, needs no transcript recovery,
and measures precisely what changed.

---

## Reproducing

```bash
python scripts/injection_defect_audit.py     # prevalence baseline + Wilson CIs
python scripts/injection_snapshot.py --label before
python scripts/injection_snapshot.py --diff before after
python scripts/path_recall_ab.py             # old vs new regex over raw_evidence
```

Snapshots write two files: the full one carries verbatim project memory and
stays local (`docs/research/` is gitignored); a metrics-only twin under
`docs/metrics/` carries counts alone and is what these tables cite.
