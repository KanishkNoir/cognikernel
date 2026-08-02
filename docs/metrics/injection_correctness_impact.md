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

### Measured — a ReDoS found by accident, and fixed

The offline A/B would not finish. After ~29 minutes of CPU on ~1 GB of stored
evidence it had produced nothing, against an estimate of ~100 seconds. The
cause was not slowness — it was **catastrophic backtracking in the shipped path
pattern**, and the A/B's refusal to terminate is what surfaced it.

The directory-segment class contained `/` and `\`, which the repeat group's own
terminator also consumes. A run like `a./a./a./…` that never ends in a valid
extension therefore had exponentially many ways to split:

| repetitions | before fix | after fix |
|---|---|---|
| 14 | 0.018 s | 0.0001 s |
| 16 | 0.059 s | 0.0001 s |
| 18 | 0.249 s | 0.0001 s |
| 20 | 0.986 s | 0.0001 s |
| 22 | 3.318 s | 0.0001 s |

Growth is 4× per two extra repetitions — exponential. At n=22 the fix is
~33,000× faster, and it is flat rather than merely faster.

Two things worth stating plainly. First, **the pre-existing pattern had the
same flaw** (0.195 s at n=20); widening it for Windows paths made the blowup
reachable through backslash runs too, so this was inherited and amplified,
not introduced. Second, a transcript containing such a run would have stalled
extraction — this is a robustness defect in shipped code, not merely a slow
research script. The fix removes the separators from the inner class so each
repetition matches exactly one segment with nothing to backtrack over.
Regression tests assert linear time on dot-slash, backslash, and mixed runs.

### Measured — path recall, and what three attempts at measuring it cost

Final result over 1,024 stored evidence blobs across 163 stores:

| | paths extractable |
|---|---|
| old pattern | 2,705 |
| new pattern | 2,774 |
| **gain** | **+69 (+2.6%), 6 stores** |

Recovered by shape: 28 dot-directory, 81 other.

**This number moved twice for methodological reasons, and the progression is
the more useful finding.**

| attempt | input | project_root | result |
|---|---|---|---|
| 1 | raw JSONL | none | **−817 (−6.5%)** — invalid |
| 2 | decoded transcript | none | +27 (+1.0%) — partial |
| 3 | decoded transcript | real, from `meta` | **+69 (+2.6%)** — faithful |

*Attempt 1* fed raw JSONL, whose literal `\n` escapes are backslash+letter —
which the new Windows-separator support reads as directory separators, fusing
prose and paths into strings like `skeleton/n/nsrc/conductor/driver.py`.
Production never sees that: `jsonl_to_transcript` parses each line first. Had
this been reported, it would have shown a regression that does not exist.

*Attempt 2* still withheld `project_root`, so every absolute path canonicalized
to `''` and was dropped. Production supplies a root, so the harness was
measuring a pipeline configuration that never runs.

The 6 paths that genuinely stopped being extracted were checked individually
rather than assumed, and neither category is a regression:

- **URL fragments** — the old pattern minted
  `record/blob/main/locales/en/x.md` out of a GitHub URL. These were never
  project files; dropping them is a precision improvement.
- **Absolute paths** — resolvable only with a root. With one, e.g.
  `/home/me/src/cognitrace/baselines/full_context.py` correctly becomes
  `src/cognitrace/baselines/full_context.py`.

**Read this number honestly**: +2.6% aggregate, 6 of 163 stores. The
qualitative change is larger than the aggregate suggests — `.claude/settings.json`
and `.codex/config.toml` were previously *unmatched entirely*, so they were
silently absent rather than corrupted — but this is a narrow fix, not a
headline one. The skeleton scoping (94% → 0% vendored) is the branch's
substantive win.

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

### Reading merge stats after this change

`execute_merge` now returns a `rejected` count alongside the existing keys, and
`session_end` sets `extracted` from the candidate list *before* the gate runs.
So the identity to expect is:

```
extracted = inserted + updated + rejected
```

Anything that previously assumed `extracted == inserted + updated` will show a
gap once the gate starts firing. The gap is `rejected`, and it is the feature
working — not lost extraction. Verified on the end-to-end run below: 8
extracted = 6 inserted + 1 updated + 1 rejected.

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

**A replay harness must reproduce the pipeline's input, not just its data.**
Both times this one disagreed with the code, the harness was wrong: first by
skipping JSONL decoding, then by withholding `project_root`. The sign of the
result flipped from −6.5% to +2.6% purely on those two corrections. Any future
A/B here should assert it reconstructs the same string extraction actually
receives before it is trusted to contradict a unit test.

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
