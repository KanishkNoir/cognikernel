# Injection correctness — defect taxonomy, root-cause fixes, and admission control

**Status:** design approved, not yet implemented
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

1. Offline A/B re-extraction over the 163 local stores' source transcripts
   shows **≥90% reduction in detector-flagged defects per class** (new
   pipeline vs old, same inputs).
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

Seven named defect classes. Each gets a pure-function detector; the taxonomy
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

### D1 — path corruption (the first-char strip)

Property-based round-trip contract: for any well-formed path embedded in
transcript text, the stored `component_map` key equals
`canonicalize_path(original)`. `utils/paths.py:canonicalize_path` is verified
clean; suspects are windowing span slicing, sanitize's markdown stripping, and
trie matching. The coexistence of intact and truncated variants of the same
file proves a single corrupting code path.

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

**Render-time invariant check (last line of defense):**
`injection/template.py:render_injection` verifies the rendered block: no
box-drawing characters, no duplicate normalized lines across sections, all
paths grounded-or-flagged. A violation logs to stderr, drops the offending
line, and never crashes — an empty section beats a dead session.

---

## 5. Evidence plan (what / why / how / impact)

- **What/Why — taxonomy section:** seven classes, definitions, anonymized
  real examples, baseline prevalence + Wilson CIs over 163 stores.
- **How — architecture section:** root-cause fixes + admission control with
  grounding, contrasted with filter-only designs.
- **Impact — three tiers:**
  1. **Offline A/B (primary, immediate):** re-run old vs new extraction over
     the same source transcripts of all 163 stores; per-class defect
     reduction vs the ≥90% target. Deterministic, exactly reproducible.
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

## 8. Open questions (deferred to implementation)

- Exact location of the D1 first-char strip (found via systematic debugging;
  the round-trip property test is the acceptance contract regardless).
- Whether Hypothesis is added as a dev dependency or property cases are
  hand-parametrized.
- The near-miss edit-distance threshold (start: first-segment distance ≤1;
  tune against the audit corpus, report false-positive rate).
