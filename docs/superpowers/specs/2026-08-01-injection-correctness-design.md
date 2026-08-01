# Injection correctness — defect taxonomy, root-cause fixes, and admission control

**Status:** design approved; **§0 amended 2026-08-01 after a measurement pass**
(see §0.1) — the amendment corrects, and in one case retracts, claims made from
unmeasured evidence. No re-approval sought: these are factual corrections
discovered while planning, and they narrow scope rather than expand it.
**Date:** 2026-08-01
**Framing:** first cycle of the next-version quality trajectory (injection
correctness → retrieval relevance → speed/granularity, each its own spec).
Prevent-only: no repair of existing user stores; old defects decay out via the
existing decay mechanism.

---

## 0. The weakness

CogniKernel's entire output surface is the injected context block. Live
evidence — the block injected into the very session that produced this spec —
shows it carries defects that authoritatively mislead the agent (the MCP
instructions declare the block "supersedes CLAUDE.md"):

- **Corrupted paths**, each missing exactly its first character:
  `xtraction/jsonl_converter.py`, `rc/memlora/extraction/file_mentions.py`,
  `torage/connection.py`, `claude/settings.json` (from `.claude/…`). The same
  file appears in one store **both** intact (`src/storage/clickhouse_client.py
  · 5x`) and truncated (`rc/storage/clickhouse_client.py · 7x`) — so one
  specific extraction code path strips a leading character, and dedup then
  fails to unify the variants.
- **An ASCII table fragment stored as a HARD constraint** (box-drawing
  characters, no propositional content).
- **Compaction boilerplate stored as a constraint** ("read the full transcript
  at: C:\Users\…jsonl Continue the conversation from").
- **A question classified as a HARD constraint** ("Confirm the env var name…"
  — flagged as such *inside the stored text itself*).
- **Skeleton budget spent on vendored dependencies**: the Codebase skeleton
  section rendered `.uv-cache/archive-v0/…/attr/_make.py` and
  `attr/validators.py` — the `attrs` library's internals — instead of
  CogniKernel's own modules, because `_SKIP_DIRS` in
  `src/cognikernel/symbols/extractor.py:486` omits `.uv-cache`, `.pytest_tmp`,
  and `.claude`, and the 500-file budget fills first-glob-wins.
- **Cross-section duplicates**: the same fact rendered verbatim in "Key
  decisions" and "Do not retry" (content-hash dedup is per-event-type).
- **Mashed fragments**: decisions stitched from unrelated spans with `->`
  splices and orphaned list markers.

## 0.1 Amendment — what the measurement pass actually found

Every defect above was read off **one artifact**: the session-context block
injected into the session that authored this spec. Before planning fixes, a
read-only date-stratified sweep of all 163 local stores (8,822 active
statement rows, 370 component rows, 10,165 descriptions) tested each class.
The result changes the plan materially.

| Class | Measured | Stores | Dating | Verdict |
|---|---|---|---|---|
| D1 path-corruption | 17 / 370 (4.6%) | 2 | **all 2026-05-10, none since** | **LEGACY — retracted as a live bug** |
| D2 junk-as-constraint | 64 / 3,250 (2.0%) | 20 | all months incl. 08 | live, but **detector over-fires** |
| D4 boilerplate-capture | 9 / 8,822 (0.1%) | 9 | through 2026-07 | live, small, cheap |
| D5 cross-type duplicate | 79 / 8,723 (0.9%) | 20 | all months | **live, confirmed** |
| D6 mashed-fragment | 8 / 8,822 (0.1%) | 2 | 2026-07/08 | marginal — see below |
| D7 subject-less | 575 / 8,822 (6.5%) | 34 | **growing: 188→310→66** | **live, dominant** |
| D3 vendored-skeleton | 2 stores affected | 2 | current | **live, severe where it hits** |

**D1 is retracted.** The spec asserted "one specific extraction code path
strips a leading character," present tense. It does not. All 17 occurrences
are in 2 stores on a single day (2026-05-10), during the `memlora` era — the
paths themselves contain `memlora`, the pre-rename package name. A direct
probe of `_FILE_PATTERN` + `canonicalize_path` on 19 realistic path shapes
produced **zero** truncations. There is no live generator. D1 becomes a
regression guard, not a bug hunt.

**But a real, live path bug was found in its place — silent loss, not
corruption.** `_FILE_PATTERN`'s negative lookbehind `(?<![a-zA-Z0-9_./\\])`
combined with a body class `[a-zA-Z0-9_/.-]` containing no backslash means
these produce **no match at all**:

- `.claude/settings.json` and every dotfile/dotdir path
- `./src/foo.py` and `/abs/src/foo.py`
- **every Windows backslash path** (`C:\Users\...\src\storage\connection.py`)

A whole class of file mentions is invisible to component tracking. Fixing the
lookbehind alone is insufficient — the body class and the `/` separator in the
directory-repeat group must widen too, or Windows paths still vanish.

**D2's 2.0% is an artifact of a bad proxy.** Inspection of the hits shows the
detector firing on leading-imperative constraints — *"Do the slow work outside
any transaction"*, *"Do not cache in Redis"* — which are well-formed. It
cannot tell an imperative "Do not…" from an interrogative "Does…". The genuine
hits (*"…should we just bring in Celery + Redis?"* stored as a constraint) are
a small fraction. **Detector precision is therefore itself a deliverable**, not
an assumption; the audit must report per-detector false-positive rate against
hand labels before any rate is quoted.

**D7 is the real headline** and the spec under-weighted it: 6.5%, 34 of 163
stores, rising month over month, with unambiguous examples (*"This only
guarantees the event is captured exactly once."* — what does? *"It must not be
able to take down the pipeline."* — what must not?).

**D6 is marginal but surfaced a distinct real defect:** hits like
`'Src/cognitrace/harness/latency.py …'` are not mashed fragments — they are
`extraction/normalize.py:81` (`s = s[0].upper() + s[1:]`) capitalizing a
description that begins with a file path, corrupting the path's first segment.
Small, live, and a one-line guard.

**Two claims tested and dropped, recorded so they are not re-litigated:**

- *Encoding corruption*: 0 U+FFFD in 10,165 descriptions. The `?`-looking
  glyphs in earlier output were console encoding, not stored data. No such
  defect class exists.
- *D3's mechanism*: confirmed, and worse than described where it hits — one
  store carries **4,655 of 4,934 symbol nodes (94%) from vendored/cache
  paths**, and both affected stores exceed the file cap. So the cap does fill
  with vendored files *and* they crowd the PageRank ranking and 600-token
  `skeleton_budget`. Only 2 of 163 stores are affected — but CogniKernel's own
  store is one of them, which is why it appeared in the authoring session.

The prior statement-quality audit (see
`2026-07-25-memory-statement-generation-design.md` §0) measured 11.9% surface
defects over 6,434 statements across 163 stores, with a plausible true rate of
35–50%. That branch targets statement *generation* and is gated on Phase A0;
this cycle is independent of it and targets *correctness of what is admitted
and rendered*.

Adoption context: 350+ PyPI downloads in week one, 600+ clones and 200+ unique
viewers in two weeks. Users have live stores; injected garbage is now a
user-facing product defect, not an internal curiosity.

---

## 1. Goals and non-goals

**Goal:** every statement, path, and skeleton entry CogniKernel injects must be
well-formed, grounded in the actual codebase, and non-duplicated.

**Success criteria:**

1. **Per-class reduction on the classes measured live in §0.1** — D2, D3, D4,
   D5, D6, D7 — against their committed baseline. Targets are set per class
   *after* detector precision is established (§0.1: D2's proxy over-fires), not
   uniformly: a 90% cut in a detector's hits is meaningless if a third of them
   are false positives. **The ≥90%-for-every-class target from the approved
   draft is withdrawn**; it was written before the classes were measured, and
   for the legacy-only class (D1) reduction against current code is ~0 by
   construction — there is nothing live left to remove.
2. **Zero D1-detectable corrupted paths in any rendered injection block** —
   grounding + near-miss rejection make this a hard invariant, not a
   statistic. (A path that is neither in the inventory nor a near-miss of a
   known path is *unverifiable*, not provably corrupt; it renders flagged
   `unverified`, which is the honest limit of the guarantee.)
3. **No latency regression**: extraction stays LLM-free and single-pass; the
   gate adds <5% extraction wall-time overhead (measured in the benchmark
   harness).

**Non-goals:** statement generation (separate gated branch), retrieval ranking
(next cycle), repair or migration of existing store *content* (prevent-only),
speed optimization beyond not-regressing.

---

## 2. Defect taxonomy and measurement harness

### 2.1 The taxonomy

Seven named defect classes, **defined here but prioritized by §0.1's measured
prevalence**: D7 and D3 first (dominant / severe), then D5, D2, D4, then D1 as
a regression guard. Detector definitions below are the *starting* shapes; §0.1
showed at least D2's needs to be rewritten before its rate means anything.
Each class gets a pure-function detector; the taxonomy
is itself a paper contribution (definitions + anonymized real examples +
baseline prevalence with Wilson CIs), positioned against the gap that the
Mem0/Letta/A-Mem literature benchmarks recall, not stored-memory quality.

| ID | Class | Detector sketch | Observed example |
|---|---|---|---|
| D1 | path-corruption | mentioned path fails grounding against project file inventory; near-miss of a known path (see §4) | `xtraction/jsonl_converter.py` |
| D2 | junk-as-constraint | box-drawing/table chars, >30% non-alphabetic, or interrogative form in `CONSTRAINT_*` | ASCII table as HARD constraint |
| D3 | vendored-skeleton | skeleton entry under a dependency/cache/tool dir | `.uv-cache/…/attr/_make.py` |
| D4 | boilerplate-capture | harness/compaction phrases ("read the full transcript at", "Continue the conversation from"), or CogniKernel's own injection sentinel | constraint quoting compaction summary |
| D5 | cross-type duplicate | same content hash or normalized-text key under two event types | fact rendered in two sections |
| D6 | mashed-fragment | non-contiguous span splice: mid-word joins, `->` splices, orphaned list markers, multiple unrelated sentence heads | the "F6 — …" decision blob |
| D7 | subject-less statement | dangling anaphora / no grammatical subject (reuses Phase A codebook heuristics) | "Fails the build if any file…" |

### 2.2 Where the code lives

- **`src/cognikernel/quality/detectors.py`** (shipped): pure functions,
  stdlib-only, one detector per class, each returning
  `DetectorHit(rule_id, span, note) | None`. Shared by the audit harness and
  the admission gate so measurement and enforcement can never drift apart.
- **`scripts/injection_defect_audit.py`** (research tooling, not shipped —
  same convention as the Phase A0 audit script): runs all detectors over
  every local store, emits per-class × per-event-type prevalence with Wilson
  CIs (reusing the stats code specced for Phase A0), and writes a
  machine-readable **baseline JSON** — the "before" column of the paper's
  impact table and the regression-test fixture.

---

## 3. Root-cause fixes

Fixes are per defect class. The *test* is the contract; exact fix locations
are implementation details found via systematic debugging.

### D1 — path integrity (rescoped by §0.1: guard, not hunt)

There is no live truncation bug. Two things replace the bug hunt:

1. **Regression guard.** A Hypothesis property test asserting that for any
   well-formed path embedded in transcript text, the stored `component_map`
   key equals `canonicalize_path(original)` — no truncation, ever. This is
   expected to pass on first run; its value is preventing regression to the
   May-10 behavior. If it *fails*, a live generator exists after all and the
   original bug hunt resumes.
2. **The live recall bug (§0.1).** Widen `_FILE_PATTERN` to match dotfile
   paths, `./`-relative, absolute, and Windows backslash paths. All three of
   the lookbehind, the body character class, and the directory-separator group
   must change together; `canonicalize_path` then normalizes separators. Also
   guard `normalize.py:81` so first-letter capitalization never rewrites a
   description whose leading token is a path.

### D3 — vendored skeleton (scope the walk)

`symbols/extractor.py:_discover_project_paths` gains, in order:

1. **`.gitignore` respect** — stdlib `fnmatch`-based matcher over the
   project's `.gitignore` (no new dependency); silently skipped when absent.
2. **Expanded `_SKIP_DIRS`** — add `.uv-cache`, `.claude`, `.pytest_tmp`,
   `.ruff_cache`, `.coverage`, `htmlcov`, `.idea`, `.vscode`, `target`,
   `vendor`.
3. **Budget prioritization** — when candidates exceed `_MAX_FILES` (500),
   rank by src-likeness (shallow depth; top-level dirs `src/`, `lib/`, the
   package name; not under a dot-dir) so project code always wins slots.
   Never first-glob-wins.

### D2/D4 — junk and boilerplate (admission hardening)

Candidates for `CONSTRAINT_*` / `DECISION` are rejected when: interrogative
form; >30% non-alphabetic characters (whitespace excluded from the count);
contain box-drawing/table characters;
or match a curated harness-boilerplate phrase list (compaction summaries, hook
chatter). **Self-ingestion guard:** the extractor must never re-ingest text
CogniKernel itself injected, detected via the injection block's own sentinel
header ("## Session context [auto-generated — do not edit]").

### D5 — cross-type duplicates

Content-hash uniqueness checked **across all event types** at write time
(currently per-type). Near-duplicates caught at render time by extending the
existing consolidation pass with a normalized-text key (lowercase, collapse
whitespace, strip punctuation).

### D6/D7 — statement well-formedness

Windowing must not splice non-adjacent sentences into one description:
enforce span contiguity and single-statement extraction (one candidate
sentence + optional adjacent rationale sentence). The Phase A subject-presence
heuristic gates `DECISION`/`CONSTRAINT_*` admission.

---

## 4. Admission gate, grounding, telemetry

One choke point, called from `extraction/pipeline.py` immediately before
storage:

```python
# src/cognikernel/quality/gate.py
def admit(event: Event, ground: GroundingContext) -> Verdict:
    """Verdict = admit | reject(rule_id) | downgrade(rule_id)"""
```

The gate runs the §2 detectors plus **path grounding** — the research-driven
architectural addition. No competing memory system maintains referential
integrity between stored memory and the codebase it describes.

**Grounding semantics:**

- `GroundingContext` is built **once per extraction run** from data already
  loaded: symbol-store paths + `_discover_project_paths` output + git index
  (`git ls-files`, cached). Per-event checks are set lookups — no I/O.
- Path found in inventory → admit as-is.
- Path **not** found → **downgrade**, not drop (new files legitimately appear
  mid-session): weight halved, additive payload key `grounding: "unverified"`.
- Path is a **near-miss** of a known path (its full text equals a known path
  with the first 1–2 characters removed, or edit distance 1 on the first
  segment) → **reject** as D1 corruption.

**Telemetry:** every reject/downgrade increments a per-rule counter in a new
`quality_telemetry` table (additive migration, follows the `api_telemetry`
pattern from Wave 1.3): `(project_id, session_id, rule_id, count, ts)`.
`cognikernel doctor` prints top rejection rules — real-world prevention rates
for the paper's longitudinal claim.

**Render-time invariant check — scope resolved.** The approved draft said the
render check verifies "all paths grounded-or-flagged," which is render-time
*filtering* of stored rows — the option explicitly declined in favour of
prevent-only. Resolved: **the render check enforces structural invariants
only** — no box-drawing characters, no duplicate normalized lines across
sections. It does **not** ground paths, so the 17 legacy May-10 paths continue
to render until decay retires them. That is the accepted cost of prevent-only,
and §0.1 shows the exposure is 2 stores, not a fleet-wide problem. Grounding
lives solely in the admission gate, where it only ever sees new events. A
violation logs to stderr, drops the offending line, and never crashes — an
empty section beats a dead session.

---

## 5. Evidence plan (what / why / how / impact)

- **What/Why — taxonomy section:** seven classes, definitions, anonymized
  real examples, baseline prevalence + Wilson CIs over 163 stores.
- **How — architecture section:** root-cause fixes + admission control with
  grounding, contrasted with filter-only designs.
- **Impact — three tiers:**
  1. **Offline A/B (primary) — rescoped by measurement.** The approved draft
     assumed the source transcripts of all 163 stores could be re-extracted.
     They cannot: only **22 of 139 stored session ids (15.8%)** still have a
     `~/.claude/projects/**/*.jsonl` transcript on disk. Sessions are the unit
     that disappears, so the A/B runs on two corpora instead:
     - **Recovered corpus (n=22 sessions):** true end-to-end old-vs-new
       re-extraction. Small, real, honestly reported as such with CIs.
     - **Fixture corpus (committed):** hand-built transcripts under
       `tests/fixtures/transcripts/` that instantiate each live defect class,
       including the §0.1 path shapes. Deterministic, reproducible by anyone
       cloning the repo — which the recovered corpus is not, since it depends
       on one machine's private transcripts.
     The **stored-event sweep** (163 stores) remains the prevalence baseline
     for the taxonomy; it measures how common each defect is, while the A/B
     measures whether the fix removes it. Keeping those two roles distinct is
     what the approved draft conflated.
  2. **Downstream (Wave 4 tie-in):** run the ready Arm C benchmark on the new
     pipeline; measure token use / constraint adherence. May show no
     significant change — reported either way.
  3. **Longitudinal (slow-burn):** `quality_telemetry` rejection rates from
     real-world use of the released version, reported in a future revision.

---

## 6. Error handling and testing

**Failure posture:** the gate and render checks are advisory layers around a
pipeline that must never block a session. Any exception inside a detector or
grounding → log to stderr, admit the event un-gated, increment a `gate_error`
counter. Extraction exits 0 always (existing contract).

**Testing (TDD throughout):**

- Per-detector unit tests with the §0 observed defects as fixtures — the live
  session-context examples become the test corpus.
- Property tests for path round-tripping (Hypothesis if already a dev
  dependency, otherwise exhaustive parametrized cases).
- Golden-file test for `render_injection` invariants.
- Migration idempotency test for `quality_telemetry`.
- A/B harness smoke test on 2–3 small fixture stores under `tests/fixtures/`.
- **Regression gate in CI:** detectors re-run on fixture stores; the build
  fails if any class's count rises above the committed baseline JSON.

**Compatibility:**

- No changes to existing event fields; `grounding` is an additive payload key.
- One additive migration (`quality_telemetry`), bumping
  `EXPECTED_SCHEMA_VERSION` per the existing `migrations/` pattern.
- Older stores read fine (prevent-only). Same degradation contract as the
  rest of the pipeline: quality-layer failures degrade to un-gated admission,
  never crash.

---

## 7. Decisions recorded

1. **Sequencing:** injection correctness first; retrieval relevance and
   speed/granularity follow as separate spec → plan → implement cycles.
2. **Prevent-only:** no repair or render-time filtering of existing store
   content; old defects decay out. (Considered and declined: repair
   migration; render-time filtering of legacy events.)
3. **Shared detector library:** audit and gate use the same
   `quality/detectors.py` functions so measurement and enforcement cannot
   drift.
4. **Downgrade-not-drop for unknown paths;** reject only on near-miss
   corruption evidence.
5. **Evidence:** all four tracks (offline A/B, taxonomy-as-contribution,
   telemetry, Wave 4 tie-in) plus grounding as the researched
   beyond-current-architecture addition.

## 8. Open questions

**Closed by the §0.1 measurement pass:**

- ~~Exact location of the D1 first-char strip~~ — no live strip exists; D1 is
  legacy-only and becomes a regression guard.
- ~~Whether Hypothesis is a dev dependency~~ — it already is
  (`pyproject.toml` `[dependency-groups] dev`), so property tests are free.
- ~~Near-miss edit-distance threshold~~ — **demoted**. With no live generator
  of truncated paths, near-miss rejection is a defense-in-depth guard, not a
  threshold worth tuning against n=17 in 2 stores. The **downgrade** path for
  unverified paths carries the value and keeps the tuning surface.

**Still open:**

- Per-class reduction targets, set only after each detector's false-positive
  rate is measured against hand labels (§0.1 shows D2's proxy is unfit as-is).
- Whether D6 survives as its own class at 0.1% in 2 stores, or is retired in
  favour of the specific `normalize.py:81` path-capitalization guard it
  actually surfaced.
