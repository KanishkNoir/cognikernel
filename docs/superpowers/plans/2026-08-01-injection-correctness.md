# Injection Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop CogniKernel injecting malformed, ungrounded, or duplicated statements into the agent's context by routing every extracted event through one measured admission gate.

**Architecture:** A new leaf package `cognikernel.quality` holds pure-function defect detectors and an admission gate. Extraction calls the gate immediately before persistence; the gate admits, downgrades, or rejects, and records per-rule counters. The same detector functions back the research audit script, so measurement and enforcement cannot drift. A render-time structural check in `injection/template.py` is the last line of defence.

**Tech Stack:** Python ≥3.11, stdlib only for the shipped quality package, SQLite, pytest + Hypothesis (already in the dev group), import-linter for layer contracts.

**Verified before writing:** every regex in Tasks 1, 2, and 4 was executed against the exact assertions this plan makes, so the TDD steps fail for the right reason and pass on a correct implementation. One iteration came out of that: an opener-based interrogative heuristic flagged `"What must be redacted: …"` as a question, so D2 now keys on the `?` terminator alone (see the comment in Task 2). API shapes were checked against the codebase too — `Event` validates `event_type` in `__post_init__`, `get_connection` is a context-manager generator, and `tests/conftest.py` already provides migrated `tmp_db` / `conn` fixtures.

**Four corrections from review, already folded in.** Recording them because each was a plausible-looking plan that would have shipped something wrong:

1. Widening the regex alone does **not** make absolute paths work — `canonicalize_path` returns `''` for any absolute path when `project_root` is `None`, so they matched and then vanished at `file_mentions.py:79`. Task 4 now plumbs `project_root` and asserts end-to-end event production, not just regex matching.
2. D7 **downgrades**, it does not reject (see Deviations §2).
3. Task 9 filters the **event set**, not the rendered string, so the render ledger cannot over-claim.
4. Task 8's tripwire counts the five statement types only, so its rate is comparable to the §0.1 baseline.

> **CORRECTION (found during implementation, 2026-08-02).** This plan repeatedly
> calls `persist_events` "the one function every extraction path reaches." That is
> **false**. `persist_events` has *no production callers at all* — `session_end`,
> `process_jobs`, and `rebuild_from_raw` every one go through
> `delta.merge.execute_merge`. Wiring the gate only into `persist_events` made it
> dead code in production; it passed every unit test while guarding nothing. The
> gate now runs inside `execute_merge`'s candidate loop, which is the real choke
> point, and `execute_merge` gained a `rejected` stats key. Task 8 below is
> retained as written for the record — read it knowing the seam it names is the
> wrong one. The lesson is the cheap one: verify the callers, not the docstring.
>
> The same trap recurred twice more, and all three share one shape — **a
> parameter added is not a parameter passed**:
> 1. the gate wired into a function with no callers;
> 2. `ground` added to `execute_merge` but supplied by no call site, leaving
>    path grounding inert until `build_grounding_context` was wired into all
>    three production merges;
> 3. an empty inventory silently downgrading every component event, since a
>    brand-new project has no symbol graph — now treated as "cannot verify".
>
> **A ReDoS in the Task 4 pattern**, found because the offline A/B would not
> terminate. The directory-segment class contained `/` and `\`, which the repeat
> group's terminator also consumes, so a run like `a./a./a./…` with no valid
> extension backtracked exponentially — 4× per two extra repetitions, 3.3s at
> n=22. The pre-existing pattern had the same flaw (0.195s at n=20); widening it
> for Windows paths made it reachable through backslash runs as well. Removing
> the separators from the inner class makes each repetition match exactly one
> segment: 0.0001s, flat. Regression tests assert linear time. A transcript
> containing such a run would have stalled extraction, so this was a shipped
> robustness defect, not a slow script.
>
> A second gap surfaced only under end-to-end testing: D4 boilerplate was scoped
> to statement types, so one compaction sentence extracted twice had its
> `CONSTRAINT_HARD` copy rejected while the `THREAD_OPEN` copy was stored. D4 is
> now type-independent — harness chatter is never a project fact regardless of
> classification. D7 stays statement-scoped, since a `THREAD_OPEN` legitimately
> refers to the current work item.

## Why one gate (the finding that shaped this plan)

`extraction/sanitize.py` already ships `is_context_dependent_fragment()` and `is_question_description()`. They are wired in only partially:

| Predicate | Wired into | Not wired into |
|---|---|---|
| `is_context_dependent_fragment` | `pipeline.py:356` (`_extract_via_head`), `pipeline.py:415` (`_filter_and_retype_with_head`) — the **v1/v2 head paths only** | the **`legacy` path, which is the default** (`config.extractor = "legacy"`) |
| `is_question_description` | `windowing.py:233,251` — `CONSTRAINT_HARD` only | pattern events, co-capture events, all other types |

That is why the measured D7 rate is 6.5% across 34 stores and rising: the detector exists, the default extraction path never calls it. **Do not fix this by adding more call sites.** The whole point of the gate is that every path funnels through one choke point.

## Global Constraints

- Python `>=3.11`. Package is `cognikernel`, src layout (`src/cognikernel/...`).
- **`cognikernel.quality` must import only the stdlib and `cognikernel.model`.** It is a leaf. Enforced by an import-linter contract added in Task 1. Never import `storage`, `injection`, `compression`, `integration`, or `delta` from it.
- **Fail-open everywhere.** Any exception inside a detector, the gate, or the render check must be caught, logged to stderr, and treated as "admit". Extraction must never block or crash a session. This is an existing contract of the codebase.
- **Additive schema only.** One migration `019_quality_telemetry.sql`; bump `EXPECTED_SCHEMA_VERSION` in `src/cognikernel/config.py` from `18` to `19`. Migration files must contain no `BEGIN`/`COMMIT`/`PRAGMA`/`VACUUM` (the runner wraps them in a transaction — see `storage/migrations.py:116`).
- **No changes to existing event fields.** `grounding` is a new additive key inside `Event.payload`.
- **Prevent-only.** Never modify, delete, or filter already-stored events. The render check enforces structural invariants only; it does **not** ground paths.
- Test style follows the codebase: `tests/unit/<package>/test_<module>.py`, class-grouped tests, `-> None` return annotations, direct imports.
- Run tests with `.venv/Scripts/python.exe -m pytest` on Windows.

## Measured baseline (from spec §0.1 — targets are set against these)

| Class | Rate | Stores | Status |
|---|---|---|---|
| D7 subject-less | 575/8,822 (6.5%) | 34 | live, dominant, growing |
| D2 junk-as-constraint | 64/3,250 (2.0%) | 20 | live; **proxy over-fires, must be rebuilt** |
| D5 cross-type duplicate | 79/8,723 (0.9%) | 20 | live, confirmed |
| D4 boilerplate | 9/8,822 (0.1%) | 9 | live, small |
| D6 path-capitalization | 8/8,822 (0.1%) | 2 | marginal; real cause is `normalize.py:81` |
| D3 vendored skeleton | 2 stores; one at 4,655/4,934 nodes (94%) | 2 | live, severe where it hits |
| D1 path truncation | 17/370, all 2026-05-10 | 2 | **legacy — regression guard only** |

## File structure

| File | Responsibility |
|---|---|
| `src/cognikernel/quality/__init__.py` | Package marker; re-exports `DetectorHit`, `run_detectors`, `admit`, `Verdict` |
| `src/cognikernel/quality/detectors.py` | Pure defect detectors, one per class. No I/O. |
| `src/cognikernel/quality/gate.py` | `admit()`, `Verdict`, `GroundingContext`. Pure; caller supplies the path inventory. |
| `src/cognikernel/storage/migrations/019_quality_telemetry.sql` | `quality_telemetry` table |
| `src/cognikernel/storage/quality_telemetry.py` | Insert/read counters |
| `scripts/injection_defect_audit.py` | Research tooling: 163-store sweep → baseline JSON + Wilson CIs |
| `scripts/injection_ab.py` | Research tooling: old-vs-new A/B over recovered + fixture corpora |
| `tests/fixtures/transcripts/defects/*.txt` | Committed fixture corpus, one file per live defect class |

---

### Task 1: Quality package skeleton, `DetectorHit`, and the D7 detector

D7 is the dominant live class, so it goes first and establishes the detector contract every later task follows.

**Files:**
- Create: `src/cognikernel/quality/__init__.py`
- Create: `src/cognikernel/quality/detectors.py`
- Modify: `pyproject.toml` (add import-linter contract)
- Test: `tests/unit/quality/__init__.py`, `tests/unit/quality/test_detectors.py`

**Interfaces:**
- Consumes: nothing (leaf package).
- Produces:
  - `DetectorHit(rule_id: str, note: str)` — frozen dataclass.
  - `detect_subject_less(text: str) -> DetectorHit | None`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/quality/__init__.py` (empty file), then `tests/unit/quality/test_detectors.py`:

```python
"""Tests for the shared defect detectors (spec §2)."""
from cognikernel.quality.detectors import DetectorHit, detect_subject_less


class TestDetectSubjectLess:
    """D7 — statements whose subject only exists in unstated context.

    Positive cases are verbatim from the 163-store sweep (spec §0.1).
    """

    def test_flags_bare_pronoun_with_adverb(self) -> None:
        hit = detect_subject_less("This only guarantees the event is captured exactly once.")
        assert hit is not None
        assert hit.rule_id == "D7"

    def test_flags_bare_pronoun_with_modal(self) -> None:
        assert detect_subject_less("It must not be able to take down the pipeline.") is not None

    def test_flags_demonstrative_with_verb(self) -> None:
        assert detect_subject_less("This also connects back to the erasure angle.") is not None

    def test_flags_conjunction_opener(self) -> None:
        assert detect_subject_less("And make the raw payload hard to log by accident.") is not None

    def test_allows_statement_with_explicit_subject(self) -> None:
        assert detect_subject_less("The dispatcher must not take down the pipeline.") is None

    def test_allows_demonstrative_with_noun_head(self) -> None:
        # "This migration" names its referent — recoverable, not subject-less.
        assert detect_subject_less("This migration is idempotent.") is None

    def test_allows_imperative_constraint(self) -> None:
        assert detect_subject_less("Do not cache in Redis.") is None

    def test_allows_empty(self) -> None:
        assert detect_subject_less("") is None

    def test_hit_carries_note(self) -> None:
        hit = detect_subject_less("It must not be able to take down the pipeline.")
        assert isinstance(hit, DetectorHit)
        assert hit.note
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/quality/test_detectors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cognikernel.quality'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cognikernel/quality/detectors.py`:

```python
"""Pure defect detectors shared by the admission gate and the research audit.

One function per defect class from the spec taxonomy. Every detector is a pure
function of text (plus, where needed, the event type) — no I/O, no database, no
filesystem — so the audit script and the live gate cannot drift apart.

Detectors return None for "clean" and a DetectorHit for "flagged". They never
raise: a detector that cannot decide returns None, because the gate's failure
posture is to admit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorHit:
    """A single defect finding. `rule_id` is the taxonomy id (D1..D7)."""
    rule_id: str
    note: str


# ── D7: subject-less statements ──────────────────────────────────────────────
#
# Tightened relative to the exploratory sweep regex, which flagged every leading
# "Do..." and so mislabelled well-formed imperative constraints. The signal we
# want is a *bare* pronoun subject — a pronoun followed directly by a verbal or
# adverbial, with no noun head to name the referent. "This migration is ..."
# keeps its referent and is therefore clean; "This only guarantees ..." does not.

_BARE_PRONOUN_SUBJECT = re.compile(
    r"^(?:this|that|these|those|it|they|he|she)\s+"
    r"(?:is|are|was|were|will|would|should|shall|must|can|could|may|might|"
    r"has|have|had|does|do|did|only|also|just|now|then|still|already|"
    r"means|makes|gives|guarantees|connects|requires|needs|lets|allows|"
    r"avoids|breaks|keeps|works|happens|applies|matters)\b",
    re.IGNORECASE,
)

_CONJUNCTION_OPENER = re.compile(
    r"^(?:and|but|so|then|also|however|therefore|thus|moreover|furthermore|"
    r"besides|otherwise|instead|meanwhile)\b[\s,]",
    re.IGNORECASE,
)


def detect_subject_less(text: str) -> DetectorHit | None:
    """D7 — the statement's subject exists only in unstated context."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    if _BARE_PRONOUN_SUBJECT.match(stripped):
        return DetectorHit("D7", "bare pronoun subject with no antecedent")
    if _CONJUNCTION_OPENER.match(stripped):
        return DetectorHit("D7", "discourse-connective opener")
    return None
```

Create `src/cognikernel/quality/__init__.py`:

```python
"""Statement-quality detection and admission control.

This package is a LEAF: it imports only the stdlib and cognikernel.model.
Never import storage, injection, compression, integration, or delta here —
an import-linter contract enforces it (pyproject.toml).
"""
from cognikernel.quality.detectors import DetectorHit, detect_subject_less

__all__ = ["DetectorHit", "detect_subject_less"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/quality/test_detectors.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Add and verify the import-linter contract**

Append to `pyproject.toml`, after the "Delta forbidden upstream imports" contract:

```toml
# Quality is a LEAF: detectors and the admission gate are pure functions over
# text and an injected path inventory. Keeping it dependency-free is what lets
# extraction call it without violating the layered pipeline above.
[[tool.importlinter.contracts]]
name = "Quality is a leaf"
type = "forbidden"
source_modules = ["cognikernel.quality"]
forbidden_modules = [
    "cognikernel.injection",
    "cognikernel.compression",
    "cognikernel.integration",
    "cognikernel.extraction",
    "cognikernel.delta",
    "cognikernel.storage",
]
```

Run: `./.venv/Scripts/lint-imports.exe`
Expected: all contracts KEPT.

- [ ] **Step 6: Commit**

```bash
git add src/cognikernel/quality tests/unit/quality pyproject.toml
git commit -m "feat(quality): detector contract and D7 subject-less detector"
```

---

### Task 2: D2 (junk-as-constraint) and D4 (boilerplate) detectors

The spec's §0.1 measurement showed the exploratory D2 proxy firing on well-formed imperatives. This task builds the real one and **pins the false positives as tests** so the mistake cannot recur.

**Files:**
- Modify: `src/cognikernel/quality/detectors.py`
- Modify: `src/cognikernel/quality/__init__.py`
- Test: `tests/unit/quality/test_detectors.py`

**Interfaces:**
- Consumes: `DetectorHit` from Task 1.
- Produces:
  - `detect_junk_constraint(text: str, event_type: str) -> DetectorHit | None`
  - `detect_boilerplate(text: str) -> DetectorHit | None`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/quality/test_detectors.py`:

```python
from cognikernel.quality.detectors import detect_boilerplate, detect_junk_constraint


class TestDetectJunkConstraint:
    """D2 — non-propositional content stored as a constraint."""

    def test_flags_box_drawing_table(self) -> None:
        text = "┌─────────┬────────┐ │ Layer │ Choice │"
        hit = detect_junk_constraint(text, "CONSTRAINT_HARD")
        assert hit is not None
        assert hit.rule_id == "D2"

    def test_flags_interrogative_stored_as_constraint(self) -> None:
        text = "For the queue between worker and dispatcher, should we just bring in Celery?"
        assert detect_junk_constraint(text, "CONSTRAINT_HARD") is not None

    def test_flags_mostly_non_alphabetic(self) -> None:
        assert detect_junk_constraint("=== 12 | 34 || 56 -- 78 ===", "CONSTRAINT_SOFT") is not None

    # ── regression pins: these are WELL-FORMED and were false positives in the
    # exploratory sweep (spec §0.1). They must never be flagged again.

    def test_allows_leading_imperative_do(self) -> None:
        text = "Do the slow work outside any transaction (LLM call, HTTP dispatch)."
        assert detect_junk_constraint(text, "CONSTRAINT_HARD") is None

    def test_allows_leading_imperative_do_not(self) -> None:
        text = "Do not cache in Redis — that adds a network hop and defeats the point."
        assert detect_junk_constraint(text, "CONSTRAINT_HARD") is None

    def test_allows_colon_list_constraint(self) -> None:
        text = "What must be redacted: message content, system prompt text, tool definitions."
        assert detect_junk_constraint(text, "CONSTRAINT_HARD") is None

    def test_ignores_non_constraint_types(self) -> None:
        # D2 is scoped to constraints; a question captured as a DECISION is out of scope here.
        assert detect_junk_constraint("Should we use Celery?", "DECISION") is None


class TestDetectBoilerplate:
    """D4 — harness/compaction chatter captured as project memory."""

    def test_flags_compaction_resume_instruction(self) -> None:
        text = "Pick up the last task as if the break never happened."
        hit = detect_boilerplate(text)
        assert hit is not None
        assert hit.rule_id == "D4"

    def test_flags_transcript_pointer(self) -> None:
        assert detect_boilerplate("read the full transcript at: C:/Users/x/a.jsonl") is not None

    def test_flags_cognikernel_own_injection_header(self) -> None:
        assert detect_boilerplate("## Session context [auto-generated — do not edit]") is not None

    def test_allows_ordinary_statement(self) -> None:
        assert detect_boilerplate("The worker retries twice before dead-lettering.") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/quality/test_detectors.py -v`
Expected: FAIL — `ImportError: cannot import name 'detect_junk_constraint'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/cognikernel/quality/detectors.py`:

```python
# ── D2: junk stored as a constraint ──────────────────────────────────────────

_BOX_DRAWING = re.compile(r"[\u2500-\u257f\u2580-\u259f]")

# A question is identified ONLY by a trailing '?'. Opener-based detection was
# tried and rejected: every candidate opener also begins well-formed
# declaratives in this corpus.
#   "Do not cache in Redis."                     imperative, not interrogative
#   "What must be redacted: message content, …"  label-list, not interrogative
# Both were flagged by an opener heuristic during verification. Requiring the
# '?' costs recall on unpunctuated questions and is the deliberate
# precision-first trade — it also matches the precedent already set by
# sanitize.is_question_description, which requires a question terminator and
# then excludes declaratives.
_CONSTRAINT_TYPES = frozenset({"CONSTRAINT_HARD", "CONSTRAINT_SOFT"})

_NON_ALPHA_MAX = 0.30


def _non_alpha_ratio(text: str) -> float:
    """Share of non-whitespace characters that are not letters."""
    body = [c for c in text if not c.isspace()]
    if not body:
        return 1.0
    return sum(1 for c in body if not c.isalpha()) / len(body)


def detect_junk_constraint(text: str, event_type: str) -> DetectorHit | None:
    """D2 — a constraint slot holding something that is not a proposition."""
    stripped = (text or "").strip()
    if not stripped or event_type not in _CONSTRAINT_TYPES:
        return None
    if _BOX_DRAWING.search(stripped):
        return DetectorHit("D2", "box-drawing/table artifact")
    if stripped.endswith("?"):
        return DetectorHit("D2", "interrogative stored as a constraint")
    if _non_alpha_ratio(stripped) > _NON_ALPHA_MAX:
        return DetectorHit("D2", "predominantly non-alphabetic")
    return None


# ── D4: harness boilerplate captured as memory ───────────────────────────────
#
# Phrases emitted by the agent harness (compaction summaries, resume banners)
# and by CogniKernel's own injected block. None of these are project facts.

_BOILERPLATE = re.compile(
    r"read the full transcript at|continue the conversation from|"
    r"resume directly|do not acknowledge the summary|"
    r"do not recap what was happening|pick up the last task|"
    r"session context \[auto-generated|as if the break never happened",
    re.IGNORECASE,
)


def detect_boilerplate(text: str) -> DetectorHit | None:
    """D4 — harness/compaction chatter, or CogniKernel's own injected block."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    if _BOILERPLATE.search(stripped):
        return DetectorHit("D4", "harness or compaction boilerplate")
    return None
```

Update `src/cognikernel/quality/__init__.py`:

```python
"""Statement-quality detection and admission control.

This package is a LEAF: it imports only the stdlib and cognikernel.model.
Never import storage, injection, compression, integration, or delta here —
an import-linter contract enforces it (pyproject.toml).
"""
from cognikernel.quality.detectors import (
    DetectorHit,
    detect_boilerplate,
    detect_junk_constraint,
    detect_subject_less,
)

__all__ = [
    "DetectorHit",
    "detect_boilerplate",
    "detect_junk_constraint",
    "detect_subject_less",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/quality/test_detectors.py -v`
Expected: PASS (20 tests)

- [ ] **Step 5: Commit**

```bash
git add src/cognikernel/quality tests/unit/quality
git commit -m "feat(quality): D2 junk-constraint and D4 boilerplate detectors

D2 deliberately does not treat a leading 'Do' as interrogative: the
exploratory sweep did, and flagged well-formed imperative constraints
('Do not cache in Redis') as defects. Those cases are pinned as tests."
```

---

### Task 3: D5 cross-type duplicate detection and the D6 path-capitalization guard

D5 needs a normalized key that a *set* of events is checked against, so its shape differs from the single-text detectors. D6's real cause is `normalize.py:81` capitalizing descriptions that begin with a file path.

**Files:**
- Modify: `src/cognikernel/quality/detectors.py`
- Modify: `src/cognikernel/quality/__init__.py`
- Modify: `src/cognikernel/extraction/normalize.py:81`
- Test: `tests/unit/quality/test_detectors.py`, `tests/unit/extraction/test_normalize.py`

**Interfaces:**
- Consumes: `DetectorHit`.
- Produces: `normalized_key(text: str) -> str` — used by the gate and the render check for near-duplicate matching.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/quality/test_detectors.py`:

```python
from cognikernel.quality.detectors import normalized_key


class TestNormalizedKey:
    """D5 — the key that makes 'same fact, different type' detectable."""

    def test_collapses_case_and_punctuation(self) -> None:
        a = normalized_key("Record Celery as an explicitly abandoned approach!")
        b = normalized_key("record celery as an explicitly abandoned approach")
        assert a == b

    def test_collapses_whitespace(self) -> None:
        assert normalized_key("a   b\n c") == normalized_key("a b c")

    def test_distinguishes_different_statements(self) -> None:
        assert normalized_key("use postgres") != normalized_key("use redis")

    def test_empty_is_empty(self) -> None:
        assert normalized_key("   ") == ""
```

Create `tests/unit/extraction/test_normalize.py` if absent, else append:

```python
"""Regression tests for description normalization."""
from cognikernel.extraction.normalize import normalize_description


class TestPathCapitalizationGuard:
    """D6 — first-letter capitalization must not rewrite a leading file path.

    Observed in the store sweep as 'Src/cognitrace/harness/latency.py ...'.
    """

    def test_does_not_capitalize_leading_path(self) -> None:
        out = normalize_description("src/cognitrace/harness/latency.py added timing")
        assert out.startswith("src/")

    def test_does_not_capitalize_dotted_path(self) -> None:
        out = normalize_description(".claude/settings.json registers the hook")
        assert out.startswith(".claude/")

    def test_still_capitalizes_ordinary_prose(self) -> None:
        assert normalize_description("the worker retries twice").startswith("The")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/quality/test_detectors.py tests/unit/extraction/test_normalize.py -v`
Expected: FAIL — `ImportError: cannot import name 'normalized_key'`, and the path-capitalization tests fail with `'Src/cognitrace/...'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/cognikernel/quality/detectors.py`:

```python
# ── D5: cross-type duplicates ────────────────────────────────────────────────
#
# Content-hash dedup is per (event_type, description), so the SAME fact stored
# under two types survives as two rows and renders in two sections. This key is
# type-independent: equal keys mean "same statement", whatever the type.

_NON_KEY_CHARS = re.compile(r"[^a-z0-9 ]")
_WHITESPACE = re.compile(r"\s+")


def normalized_key(text: str) -> str:
    """Type-independent identity key for a statement.

    Lowercase, strip punctuation, collapse whitespace. Equal keys across two
    different event types is exactly the D5 defect.
    """
    lowered = (text or "").lower()
    stripped = _NON_KEY_CHARS.sub(" ", lowered)
    return _WHITESPACE.sub(" ", stripped).strip()
```

Add `normalized_key` to the `__init__.py` import list and `__all__`.

Now read `src/cognikernel/extraction/normalize.py` around line 81. The existing line is:

```python
        s = s[0].upper() + s[1:]
```

Replace it with:

```python
        # Never recapitalize a description that opens with a file path: doing so
        # rewrites the path's first segment ('src/...' -> 'Src/...'), which then
        # fails to match anything in the codebase. Observed in 2 stores.
        if not _OPENS_WITH_PATH.match(s):
            s = s[0].upper() + s[1:]
```

And add near the other module-level patterns in `normalize.py`:

```python
# A leading token that looks like a path (has a '/' before any space, or starts
# with a dot-directory). Capitalizing these corrupts them.
_OPENS_WITH_PATH = re.compile(r"^(?:\.[a-zA-Z0-9_]|[a-zA-Z0-9_.-]+/)")
```

Confirm `import re` is already present at the top of `normalize.py`; add it if not.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/quality/test_detectors.py tests/unit/extraction/test_normalize.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite for regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/unit -q`
Expected: PASS. `normalize_description` is on the hot path for every event, so a break here shows up broadly. If existing tests assert a capitalized path, they encoded the bug — update them and say so in the commit.

- [ ] **Step 6: Commit**

```bash
git add src/cognikernel/quality src/cognikernel/extraction/normalize.py tests/unit
git commit -m "feat(quality): normalized_key for cross-type dedup; stop capitalizing leading paths"
```

---

### Task 4: Path recall fix and the D1 regression guard

Two things: widen `_FILE_PATTERN` so dotfile, relative, absolute, and Windows paths are visible at all; and pin with Hypothesis that extraction never truncates a path (the May-10 behaviour must not return).

**Files:**
- Modify: `src/cognikernel/extraction/file_mentions.py:20-27`
- Test: `tests/unit/extraction/test_file_mentions.py`

**Interfaces:**
- Consumes: `cognikernel.utils.paths.canonicalize_path` (already exists, verified correct).
- Produces: no new public API — `extract_file_mention_events` gains recall.

- [ ] **Step 1: Write the failing test**

Create or append `tests/unit/extraction/test_file_mentions.py`:

```python
"""Path recall and non-truncation contracts for file-mention extraction."""
from hypothesis import given, strategies as st

from cognikernel.extraction.file_mentions import _FILE_PATTERN
from cognikernel.utils.paths import canonicalize_path


def _match(text: str) -> list[str]:
    return [m.group(0) for m in _FILE_PATTERN.finditer(text)]


class TestPathRecall:
    """Shapes that produced NO match before the widening (spec §0.1)."""

    def test_matches_dotdir_path(self) -> None:
        assert _match("edited .claude/settings.json today")

    def test_matches_dot_relative_path(self) -> None:
        assert _match("edited ./src/storage/connection.py today")

    def test_matches_absolute_posix_path(self) -> None:
        assert _match("edited /srv/app/src/storage/connection.py today")

    def test_matches_windows_backslash_path(self) -> None:
        assert _match(r"edited C:\Users\Admin\src\storage\connection.py today")

    def test_windows_path_canonicalizes_to_forward_slashes(self) -> None:
        hits = _match(r"edited src\storage\connection.py today")
        assert hits
        assert canonicalize_path(hits[0]) == "src/storage/connection.py"

    def test_still_matches_plain_relative_path(self) -> None:
        assert _match("edited src/storage/connection.py today") == [
            "src/storage/connection.py"
        ]


class TestEndToEndEventProduction:
    """Matching the regex is not enough — an event must actually come out.

    canonicalize_path returns '' for ANY absolute path when project_root is
    None (paths.py rule 7), and file_mentions drops empty results. So without
    the project_root plumbing below, absolute paths match the pattern and then
    vanish silently — a widened regex alone does not fix them.
    """

    def _paths(self, text: str, project_root: str | None = None) -> list[str]:
        from cognikernel.extraction.file_mentions import extract_file_mention_events
        from cognikernel.extraction.tokenize import tokenize

        sentences = tokenize(f"Assistant: {text}")
        events = extract_file_mention_events(
            sentences, "p", "s", project_root=project_root
        )
        return [e.payload["path"] for e in events]

    def test_relative_path_produces_event(self) -> None:
        assert "src/storage/connection.py" in self._paths(
            "I edited src/storage/connection.py today"
        )

    def test_dotdir_path_produces_event(self) -> None:
        assert ".claude/settings.json" in self._paths(
            "I edited .claude/settings.json today"
        )

    def test_windows_relative_path_produces_event(self) -> None:
        assert "src/storage/connection.py" in self._paths(
            r"I edited src\storage\connection.py today"
        )

    def test_absolute_path_needs_project_root(self) -> None:
        text = r"I edited C:\proj\src\storage\connection.py today"
        assert self._paths(text) == []          # no root -> correctly unresolvable
        assert "src/storage/connection.py" in self._paths(text, project_root=r"C:\proj")

    def test_path_outside_project_root_is_dropped(self) -> None:
        # Not a project component; dropping it is correct, not a bug.
        assert self._paths(
            r"I edited C:\elsewhere\other.py today", project_root=r"C:\proj"
        ) == []


class TestNoTruncation:
    """D1 regression guard. Expected to pass on first run — its job is to fail
    loudly if the 2026-05-10 first-character-strip behaviour ever returns."""

    @given(
        segments=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8),
            min_size=1,
            max_size=4,
        ),
        stem=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
    )
    def test_extracted_path_is_never_a_prefix_truncation(
        self, segments: list[str], stem: str
    ) -> None:
        path = "/".join(segments) + "/" + stem + ".py"
        hits = _match(f"edited {path} today")
        assert hits, f"path vanished entirely: {path!r}"
        assert canonicalize_path(hits[0]) == path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/extraction/test_file_mentions.py -v`
Expected: the four `TestPathRecall` shape tests FAIL (empty match list); `TestEndToEndEventProduction` fails on the `project_root` keyword not existing. `TestNoTruncation` should already PASS — that is the point of a regression guard. If it fails, stop and report: a live truncation generator exists after all.

- [ ] **Step 3: Write minimal implementation**

Replace `_FILE_PATTERN` in `src/cognikernel/extraction/file_mentions.py` (lines 20-27):

```python
# Path shapes this must match — all four were invisible before (spec §0.1):
#   src/storage/connection.py     plain relative
#   ./src/storage/connection.py   dot-relative
#   .claude/settings.json         dot-directory
#   /srv/app/main.py              absolute POSIX
#   C:\Users\x\src\main.py        absolute Windows, backslash-separated
#
# Three things had to change together. Widening only the lookbehind leaves
# Windows paths invisible, because the body class carried no backslash and the
# directory-repeat group required a literal '/' terminator.
#   1. lookbehind: no longer blocks on '.', '/' or '\' — those START a path
#      rather than continue a word. It still blocks mid-identifier matches.
#   2. separator: [/\\] everywhere a separator can appear.
#   3. body class: includes '\' so backslash paths hold together.
# canonicalize_path() then folds separators to '/' downstream.
_FILE_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9_])"
    r"(?:[a-zA-Z]:[/\\])?"                       # optional Windows drive
    r"(?:[/\\]|\./|\.(?=[a-zA-Z0-9_]))?"         # optional leading / ./ or dot-dir
    r"(?:[a-zA-Z0-9_.][a-zA-Z0-9_.\\/-]*[/\\])*"  # directory segments
    r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*\."
    r"(?:py|ts|tsx|js|jsx|mjs|json|yaml|yml|sql|md|toml|env|cfg|ini|go|rs|java|cs)"
    r"(?![a-zA-Z0-9_])",
    re.ASCII,
)
```

Then plumb `project_root` so absolute paths can resolve. In `file_mentions.py`, add the keyword and pass it through:

```python
def extract_file_mention_events(
    sentences: list[Sentence],
    project_id: str,
    session_id: str,
    project_root: str | None = None,
) -> list[Event]:
```

and at the canonicalization call (currently line 78):

```python
            path = canonicalize_path(match.group(0), project_root)
```

In `extraction/pipeline.py`, add an optional field to `SessionMetadata` (last position, so existing positional construction is unaffected):

```python
@dataclass
class SessionMetadata:
    project_id: str
    session_id: str
    started_at: int   # Unix milliseconds
    ended_at: int     # Unix milliseconds
    # Absolute path of the project checkout. Optional because not every caller
    # knows it; when None, absolute paths in the transcript stay unresolvable
    # and are dropped, which is the pre-existing behaviour.
    project_root: str | None = None
```

and forward it at both `extract_file_mention_events(...)` call sites in `_extract_session_impl` (the broad-mode path and the main path):

```python
        mention_events = extract_file_mention_events(
            sentences, session_meta.project_id, session_meta.session_id,
            project_root=session_meta.project_root,
        )
```

Finally, set it where the project path is known. In `src/cognikernel/integration/session.py`, find where `SessionMetadata(...)` is constructed and add `project_root=<the project path variable already in scope>`. If no such variable exists there, leave it unset and note it — the other path shapes still work, and absolute-path resolution simply stays off for that caller.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/extraction/test_file_mentions.py -v`
Expected: PASS, including the Hypothesis property and all of `TestEndToEndEventProduction`.

- [ ] **Step 5: Check for over-matching regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/unit -q`
Expected: PASS. A widened pattern risks matching prose like "version 3.11" or module references like `cognikernel.config.py`. If any test shows a new spurious match, tighten the pattern rather than editing the assertion.

- [ ] **Step 6: Commit**

```bash
git add src/cognikernel/extraction/file_mentions.py src/cognikernel/extraction/pipeline.py src/cognikernel/integration/session.py tests/unit/extraction/test_file_mentions.py
git commit -m "fix(extraction): make dotfile, relative and Windows paths visible

The lookbehind blocked any path preceded by '.', '/' or '\\', and the body
class carried no backslash, so .claude/*, ./src/* and every Windows path
produced no match at all.

Widening the regex alone was not enough for absolute paths: canonicalize_path
returns '' for any absolute path when project_root is None, so they matched
and then vanished. SessionMetadata now carries an optional project_root that
file_mentions threads into canonicalization. Adds a Hypothesis guard against
the legacy first-character-strip behaviour."
```

---

### Task 5: D3 — scope the symbol-graph walk

One store carries 4,655 of 4,934 symbol nodes (94%) from vendored paths. Three changes: respect `.gitignore`, extend the skip list, and rank candidates so project code wins the 500-file budget.

**Files:**
- Modify: `src/cognikernel/symbols/extractor.py:486-508`
- Test: `tests/unit/symbols/test_discovery.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_discover_project_paths(project_root: Path) -> dict[str, str]` — unchanged signature, changed selection.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/symbols/test_discovery.py`:

```python
"""Scoping rules for symbol-graph file discovery (spec D3)."""
from pathlib import Path

from cognikernel.symbols.extractor import _discover_project_paths


def _touch(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x = 1\n", encoding="utf-8")


class TestSkipDirs:
    def test_skips_uv_cache(self, tmp_path: Path) -> None:
        _touch(tmp_path, "src/app.py")
        _touch(tmp_path, ".uv-cache/archive-v0/attr/_make.py")
        assert set(_discover_project_paths(tmp_path)) == {"src/app.py"}

    def test_skips_claude_and_pytest_tmp(self, tmp_path: Path) -> None:
        _touch(tmp_path, "src/app.py")
        _touch(tmp_path, ".claude/hooks/x.py")
        _touch(tmp_path, ".pytest_tmp/basetemp/proj/main.py")
        assert set(_discover_project_paths(tmp_path)) == {"src/app.py"}


class TestGitignore:
    def test_respects_gitignore_entry(self, tmp_path: Path) -> None:
        _touch(tmp_path, "src/app.py")
        _touch(tmp_path, "generated/pb2.py")
        (tmp_path / ".gitignore").write_text("generated/\n", encoding="utf-8")
        assert set(_discover_project_paths(tmp_path)) == {"src/app.py"}

    def test_absent_gitignore_is_not_an_error(self, tmp_path: Path) -> None:
        _touch(tmp_path, "src/app.py")
        assert set(_discover_project_paths(tmp_path)) == {"src/app.py"}


class TestBudgetPriority:
    def test_shallow_src_files_win_the_budget(self, tmp_path: Path) -> None:
        # 600 deep vendored-ish files vs one shallow src file; src must survive.
        for i in range(600):
            _touch(tmp_path, f"third_party/pkg{i}/deep/nested/mod.py")
        _touch(tmp_path, "src/app.py")
        found = _discover_project_paths(tmp_path)
        assert "src/app.py" in found
        assert len(found) <= 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/symbols/test_discovery.py -v`
Expected: FAIL — `.uv-cache`, `.claude`, `.pytest_tmp`, and `.gitignore` entries are all collected; the budget test fails because collection is first-glob-wins.

- [ ] **Step 3: Write minimal implementation**

Replace lines 486-508 of `src/cognikernel/symbols/extractor.py`:

```python
_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build", ".eggs",
    # Added after a store sweep found one project with 94% of its symbol
    # nodes coming from vendored/cache trees (spec D3).
    ".uv-cache", ".claude", ".pytest_tmp", ".ruff_cache", "htmlcov",
    ".idea", ".vscode", "target", "vendor", ".next", ".nuxt",
})
_MAX_FILES = 500

# Top-level directories that mark first-party source. Used only for ranking.
_SRC_HINTS = frozenset({"src", "lib", "app", "pkg", "internal", "cmd"})


def _load_gitignore_globs(project_root: Path) -> list[str]:
    """Return fnmatch-able patterns from .gitignore. Missing file → [].

    Deliberately not a full gitignore implementation (no negation, no
    ancestor files): this is a noise filter, and over-matching would hide
    real source. Unsupported lines are skipped.
    """
    path = project_root / ".gitignore"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, ValueError):
        return []
    globs: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        line = line.rstrip("/").lstrip("/")
        if line:
            globs.append(line)
    return globs


def _is_gitignored(rel_path: str, globs: list[str]) -> bool:
    """True if any gitignore pattern matches the path or one of its segments."""
    import fnmatch

    segments = rel_path.split("/")
    for pattern in globs:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        if any(fnmatch.fnmatch(seg, pattern) for seg in segments):
            return True
    return False


def _src_rank(rel_path: str) -> tuple[int, int]:
    """Sort key — lower is kept first when the budget binds.

    Ranks by (not-first-party, depth) so a shallow src/ file always outranks a
    deep third-party one. Previously collection was first-glob-wins, so walk
    order decided what made the budget.
    """
    segments = rel_path.split("/")
    top = segments[0] if segments else ""
    first_party = 0 if (top in _SRC_HINTS or len(segments) == 1) else 1
    return (first_party, len(segments))


def _discover_project_paths(project_root: Path) -> dict[str, str]:
    """Walk project for supported source files, skip noise dirs, rank by
    src-likeness, and keep at most _MAX_FILES. Returns {rel_path: abs_path}."""
    globs = _load_gitignore_globs(project_root)
    candidates: list[tuple[tuple[int, int], str, str]] = []
    patterns = ("*.py", "*.ts", "*.tsx", "*.js", "*.jsx")

    for pattern in patterns:
        for abs_p in project_root.rglob(pattern):
            if any(part in _SKIP_DIRS for part in abs_p.parts):
                continue
            try:
                rel = str(abs_p.relative_to(project_root)).replace("\\", "/")
            except ValueError:
                continue
            if globs and _is_gitignored(rel, globs):
                continue
            candidates.append((_src_rank(rel), rel, str(abs_p)))

    candidates.sort(key=lambda c: (c[0], c[1]))
    return {rel: abs_p for _, rel, abs_p in candidates[:_MAX_FILES]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/symbols/test_discovery.py -v`
Expected: PASS

- [ ] **Step 5: Verify against the real repo**

Run:

```bash
.venv/Scripts/python.exe -c "from pathlib import Path; from cognikernel.symbols.extractor import _discover_project_paths; p=_discover_project_paths(Path('.')); print(len(p)); print([k for k in p if '.uv-cache' in k or '.pytest_tmp' in k][:5])"
```

Expected: a count well under 500 and an empty list of vendored paths. Before this change the same call collected `.uv-cache` entries.

- [ ] **Step 6: Commit**

```bash
git add src/cognikernel/symbols/extractor.py tests/unit/symbols/test_discovery.py
git commit -m "fix(symbols): scope discovery to first-party source

Respects .gitignore, extends the skip list (.uv-cache, .claude, .pytest_tmp
and friends), and ranks candidates so project code wins the 500-file budget
instead of walk order deciding. One store had 94% vendored symbol nodes."
```

---

### Task 6: The admission gate

The single choke point. Pure: the caller supplies the path inventory, which keeps `quality` a leaf.

**Files:**
- Create: `src/cognikernel/quality/gate.py`
- Modify: `src/cognikernel/quality/__init__.py`
- Test: `tests/unit/quality/test_gate.py`

**Interfaces:**
- Consumes: all detectors from Tasks 1-3, `Event` from `cognikernel.model`.
- Produces:
  - `GroundingContext(known_paths: frozenset[str])` with `.is_known(path)` and `.is_near_miss(path)`
  - `Verdict(action: str, rule_id: str | None, note: str)` where `action ∈ {"admit", "downgrade", "reject"}`
  - `admit(event: Event, ground: GroundingContext | None = None) -> Verdict`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/quality/test_gate.py`:

```python
"""Admission gate behaviour (spec §4)."""
from cognikernel.model import Event
from cognikernel.quality.gate import GroundingContext, Verdict, admit


def _event(event_type: str, description: str, **payload) -> Event:
    return Event(
        project_id="p",
        session_id="s",
        event_type=event_type,
        payload={"description": description, **payload},
        content_hash="h",
        weight=1.0,
    )


class TestStatementRules:
    def test_admits_well_formed_decision(self) -> None:
        v = admit(_event("DECISION", "The dispatcher retries twice before dead-lettering."))
        assert v.action == "admit"

    def test_downgrades_subject_less_statement(self) -> None:
        # DOWNGRADE, not reject: the harm is that it renders in the block, and
        # weight collapse already prevents that. Rejecting would also remove it
        # from recall/find_related, which is strictly more destructive. This
        # matches the policy pipeline.py already states for context-dependent
        # fragments ("We DEMOTE (not drop)").
        v = admit(_event("DECISION", "It must not be able to take down the pipeline."))
        assert v.action == "downgrade"
        assert v.rule_id == "D7"

    def test_subject_less_downgrade_is_not_applied_twice(self) -> None:
        # v1/v2 head paths already demote fragments pre-hash (_FRAG_DEMOTE).
        # The gate must not stack a second multiplier on the same event.
        e = _event("DECISION", "It must not take down the pipeline.")
        e.payload["provenance"] = "salience_v2+frag"
        assert admit(e).action == "admit"

    def test_rejects_boilerplate(self) -> None:
        v = admit(_event("CONSTRAINT_HARD", "Pick up the last task as if the break never happened."))
        assert v.action == "reject"
        assert v.rule_id == "D4"

    def test_rejects_interrogative_constraint(self) -> None:
        v = admit(_event("CONSTRAINT_HARD", "Should we just bring in Celery?"))
        assert v.action == "reject"
        assert v.rule_id == "D2"

    def test_admits_imperative_constraint(self) -> None:
        v = admit(_event("CONSTRAINT_HARD", "Do not cache in Redis."))
        assert v.action == "admit"


class TestGrounding:
    def test_admits_known_path(self) -> None:
        g = GroundingContext(frozenset({"src/storage/connection.py"}))
        v = admit(_event("COMPONENT_STATUS", "x", path="src/storage/connection.py"), g)
        assert v.action == "admit"

    def test_downgrades_unknown_path(self) -> None:
        # A genuinely new file must survive — downgraded, never dropped.
        g = GroundingContext(frozenset({"src/storage/connection.py"}))
        v = admit(_event("COMPONENT_STATUS", "x", path="src/brand/new_file.py"), g)
        assert v.action == "downgrade"

    def test_rejects_near_miss_truncation(self) -> None:
        g = GroundingContext(frozenset({"src/storage/connection.py"}))
        v = admit(_event("COMPONENT_STATUS", "x", path="rc/storage/connection.py"), g)
        assert v.action == "reject"
        assert v.rule_id == "D1"

    def test_no_grounding_context_admits(self) -> None:
        v = admit(_event("COMPONENT_STATUS", "x", path="anything/at/all.py"), None)
        assert v.action == "admit"


class TestFailOpen:
    def test_detector_exception_admits(self, monkeypatch) -> None:
        import cognikernel.quality.gate as gate_mod

        def boom(*_a, **_k):
            raise RuntimeError("detector exploded")

        monkeypatch.setattr(gate_mod, "detect_subject_less", boom)
        v = admit(_event("DECISION", "It must not take down the pipeline."))
        assert v.action == "admit"
        assert v.rule_id == "gate_error"

    def test_verdict_is_a_verdict(self) -> None:
        assert isinstance(admit(_event("DECISION", "The queue is durable.")), Verdict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/quality/test_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cognikernel.quality.gate'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cognikernel/quality/gate.py`:

```python
"""Admission control for extracted events — the single choke point.

Every extracted event passes through admit() immediately before persistence.
The gate returns one of three verdicts:

  admit      store as-is
  downgrade  store with halved weight and payload['grounding'] = 'unverified'
  reject     do not store

FAILURE POSTURE: the gate never blocks a session. Any exception inside a
detector produces an 'admit' verdict tagged rule_id='gate_error', which the
caller counts. Losing a defect is acceptable; losing a session is not.

PURITY: this module takes the path inventory as an argument rather than
reading the symbol store, which keeps cognikernel.quality a leaf package
(see the import-linter contract in pyproject.toml).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cognikernel.model import Event
from cognikernel.quality.detectors import (
    detect_boilerplate,
    detect_junk_constraint,
    detect_subject_less,
)

_log = logging.getLogger("cognikernel.quality")

# Types whose payload carries a file path worth grounding.
_PATH_TYPES = frozenset({"COMPONENT_STATUS"})

# Types that assert something and must therefore be well-formed statements.
_STATEMENT_TYPES = frozenset({
    "DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
    "APPROACH_ABANDONED", "APPROACH_ABANDONED_DO_NOT_RETRY",
})

_DOWNGRADE_FACTOR = 0.5


@dataclass(frozen=True)
class Verdict:
    action: str                     # "admit" | "downgrade" | "reject"
    rule_id: str | None = None
    note: str = ""


@dataclass(frozen=True)
class GroundingContext:
    """Referential integrity between stored memory and the real codebase.

    `known_paths` is built once per extraction run by the caller from the
    symbol store, the project walk, and the git index — so a per-event check
    is a set lookup with no I/O.
    """
    known_paths: frozenset[str] = field(default_factory=frozenset)

    def is_known(self, path: str) -> bool:
        return path in self.known_paths

    def is_near_miss(self, path: str) -> bool:
        """True when `path` is a known path with leading characters removed.

        This is the 2026-05-10 corruption signature. There is no live generator
        of it, so this is defence in depth, not a tuned classifier.
        """
        if not path or self.is_known(path):
            return False
        for known in self.known_paths:
            if len(known) > len(path) and known.endswith(path):
                if len(known) - len(path) <= 2:
                    return True
        return False


def admit(event: Event, ground: GroundingContext | None = None) -> Verdict:
    """Decide whether an extracted event may be stored. Never raises."""
    try:
        return _admit_inner(event, ground)
    except Exception as exc:                      # fail-open by contract
        _log.warning("quality gate error — admitting un-gated: %s", exc)
        return Verdict("admit", "gate_error", str(exc))


def _admit_inner(event: Event, ground: GroundingContext | None) -> Verdict:
    payload = event.payload or {}
    description = payload.get("description", "") or ""

    if event.event_type in _STATEMENT_TYPES:
        # REJECT for D2/D4: box-drawing artifacts and harness boilerplate carry
        # no recoverable project content, so keeping them helps nothing.
        for hit in (
            detect_boilerplate(description),
            detect_junk_constraint(description, event.event_type),
        ):
            if hit is not None:
                return Verdict("reject", hit.rule_id, hit.note)

        # DOWNGRADE for D7: a subject-less statement still carries a real fact,
        # just one the reader cannot resolve. The harm is that it occupies the
        # budget-ranked block; weight collapse fixes exactly that while leaving
        # it reachable via recall/find_related. Rejecting would delete recoverable
        # memory. Skipped when the v1/v2 head path already demoted this event
        # (provenance carries '+frag'), so the multiplier is never applied twice.
        if "+frag" not in (payload.get("provenance") or ""):
            hit = detect_subject_less(description)
            if hit is not None:
                return Verdict("downgrade", hit.rule_id, hit.note)

    if event.event_type in _PATH_TYPES and ground is not None:
        path = payload.get("path", "") or ""
        if path:
            if ground.is_near_miss(path):
                return Verdict("reject", "D1", "path is a truncation of a known path")
            if not ground.is_known(path):
                return Verdict("downgrade", "D1", "path not found in codebase inventory")

    return Verdict("admit")


def apply_verdict(event: Event, verdict: Verdict) -> Event:
    """Mutate `event` per a downgrade verdict and return it.

    Only 'downgrade' changes the event; 'admit' and 'reject' leave it alone
    (the caller drops rejects). The marker key differs by rule so the two
    downgrade reasons stay distinguishable in the store:
      D1 -> payload['grounding'] = 'unverified'   (path not in the codebase)
      D7 -> payload['quality']   = 'context_dependent'
    """
    if verdict.action != "downgrade":
        return event
    event.weight = (event.weight or 1.0) * _DOWNGRADE_FACTOR
    if verdict.rule_id == "D1":
        event.payload["grounding"] = "unverified"
    else:
        event.payload["quality"] = "context_dependent"
    return event
```

Add `admit`, `apply_verdict`, `GroundingContext`, `Verdict` to `__init__.py` imports and `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/quality/test_gate.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Verify the leaf contract still holds**

Run: `./.venv/Scripts/lint-imports.exe`
Expected: all contracts KEPT.

- [ ] **Step 6: Commit**

```bash
git add src/cognikernel/quality tests/unit/quality/test_gate.py
git commit -m "feat(quality): admission gate with path grounding

One choke point instead of predicates wired into some extraction paths and
not others.

Verdicts are graded by how recoverable the content is. D2 and D4 reject —
box-drawing artifacts and harness boilerplate carry nothing worth keeping.
D7 downgrades, matching the policy pipeline.py already states for
context-dependent fragments: weight collapse keeps them out of the
budget-ranked block while leaving them reachable via recall. Unknown paths
downgrade too, so genuinely new files survive; only near-miss truncations
of a known path are rejected."
```

---

### Task 7: `quality_telemetry` table and migration

**Files:**
- Create: `src/cognikernel/storage/migrations/019_quality_telemetry.sql`
- Create: `src/cognikernel/storage/quality_telemetry.py`
- Modify: `src/cognikernel/config.py:11`
- Test: `tests/unit/storage/test_quality_telemetry.py`

**Interfaces:**
- Consumes: `Verdict` (Task 6).
- Produces:
  - `record_verdict(conn, project_id: str, session_id: str, rule_id: str) -> None`
  - `get_rule_counts(conn, project_id: str, limit: int = 10) -> list[dict]`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/storage/test_quality_telemetry.py`:

**Use the existing `conn` fixture from `tests/conftest.py`** — it yields an open, migrated connection. Do not define a local one; `get_connection` is a context-manager generator (`with get_connection(path) as c:`), not a plain factory, so a hand-rolled fixture that calls `.close()` on it will fail.

```python
"""Per-rule gate counters."""
import sqlite3

from cognikernel.storage.quality_telemetry import get_rule_counts, record_verdict


class TestRecordVerdict:
    def test_records_a_rule_hit(self, conn: sqlite3.Connection) -> None:
        record_verdict(conn, "p", "s", "D7")
        counts = get_rule_counts(conn, "p")
        assert counts[0]["rule_id"] == "D7"
        assert counts[0]["total"] == 1

    def test_accumulates_repeat_hits(self, conn: sqlite3.Connection) -> None:
        for _ in range(3):
            record_verdict(conn, "p", "s", "D7")
        assert get_rule_counts(conn, "p")[0]["total"] == 3

    def test_orders_by_frequency(self, conn: sqlite3.Connection) -> None:
        record_verdict(conn, "p", "s", "D2")
        for _ in range(5):
            record_verdict(conn, "p", "s", "D7")
        assert get_rule_counts(conn, "p")[0]["rule_id"] == "D7"

    def test_scopes_to_project(self, conn: sqlite3.Connection) -> None:
        record_verdict(conn, "p1", "s", "D7")
        assert get_rule_counts(conn, "p2") == []


class TestMigrationIdempotency:
    def test_running_migrations_twice_is_safe(self, conn: sqlite3.Connection) -> None:
        from cognikernel.storage.migrations import run_migrations

        # The `conn` fixture already ran migrations once; a second run must
        # be a no-op rather than an error.
        run_migrations(conn)
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        assert int(row[0]) >= 19
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/storage/test_quality_telemetry.py -v`
Expected: FAIL — no module `cognikernel.storage.quality_telemetry`

- [ ] **Step 3: Write minimal implementation**

Create `src/cognikernel/storage/migrations/019_quality_telemetry.sql` (no `BEGIN`/`COMMIT`/`PRAGMA` — the runner wraps this):

```sql
-- Per-rule admission-gate counters. One row per (project, session, rule);
-- repeat hits increment `count` so the table stays small.
CREATE TABLE IF NOT EXISTS quality_telemetry (
    project_id  TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    rule_id     TEXT    NOT NULL,
    count       INTEGER NOT NULL DEFAULT 0,
    updated_at  INTEGER NOT NULL,
    PRIMARY KEY (project_id, session_id, rule_id)
);

CREATE INDEX IF NOT EXISTS idx_quality_telemetry_project
    ON quality_telemetry (project_id, rule_id);
```

Create `src/cognikernel/storage/quality_telemetry.py`:

```python
"""Admission-gate rule counters.

Feeds `cognikernel doctor` and the longitudinal impact claim: how often each
defect rule fires in real use. Never raises — telemetry must not be able to
break extraction.
"""
from __future__ import annotations

import logging
import sqlite3
import time

_log = logging.getLogger("cognikernel.quality")


def record_verdict(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    rule_id: str,
) -> None:
    """Increment the counter for one rule firing. Best-effort."""
    if not rule_id:
        return
    try:
        conn.execute(
            """
            INSERT INTO quality_telemetry
                (project_id, session_id, rule_id, count, updated_at)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT (project_id, session_id, rule_id) DO UPDATE SET
                count      = count + 1,
                updated_at = excluded.updated_at
            """,
            (project_id, session_id, rule_id, int(time.time() * 1000)),
        )
        conn.commit()
    except Exception as exc:
        _log.warning("quality telemetry write failed: %s", exc)


def get_rule_counts(
    conn: sqlite3.Connection,
    project_id: str,
    limit: int = 10,
) -> list[dict]:
    """Return [{rule_id, total}] for a project, most frequent first."""
    try:
        rows = conn.execute(
            """
            SELECT rule_id, SUM(count) AS total
            FROM quality_telemetry
            WHERE project_id = ?
            GROUP BY rule_id
            ORDER BY total DESC, rule_id ASC
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()
        return [{"rule_id": r[0], "total": r[1]} for r in rows]
    except Exception as exc:
        _log.warning("quality telemetry read failed: %s", exc)
        return []
```

Change `src/cognikernel/config.py:11`:

```python
EXPECTED_SCHEMA_VERSION: int = 19
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/storage/test_quality_telemetry.py -v`
Expected: PASS

- [ ] **Step 5: Surface the counters in `doctor` (spec §4)**

Counters nobody reads are not telemetry. In `src/cognikernel/integration/cli.py`, find `_cmd_doctor` and the existing cache-stats section it prints (added in Wave 1.3), then add an adjacent section following the same output style:

```python
    # Quality gate — which defect rules are firing in real use. Fail-soft: a
    # doctor section must never be the thing that breaks doctor.
    try:
        from cognikernel.storage.quality_telemetry import get_rule_counts

        rule_counts = get_rule_counts(conn, project_id, limit=5)
        if rule_counts:
            print("\nQuality gate — top rejection rules:")
            for entry in rule_counts:
                print(f"  {entry['rule_id']}: {entry['total']}")
        else:
            print("\nQuality gate: no rejections recorded yet.")
    except Exception as exc:
        print(f"\nQuality gate: unavailable ({exc})")
```

Match the surrounding code's variable names for the connection and project id — they may differ from `conn` / `project_id`.

Verify by hand:

```bash
.venv/Scripts/python.exe -m cognikernel doctor .
```

Expected: a "Quality gate" section appears. On a store with no gate activity it reads "no rejections recorded yet".

- [ ] **Step 6: Verify existing stores still open**

Run: `.venv/Scripts/python.exe -m pytest tests/unit tests/integration -q`
Expected: PASS. The migration is additive, so v18 stores upgrade in place on next open.

- [ ] **Step 7: Commit**

```bash
git add src/cognikernel/storage/migrations/019_quality_telemetry.sql src/cognikernel/storage/quality_telemetry.py src/cognikernel/config.py src/cognikernel/integration/cli.py tests/unit/storage/test_quality_telemetry.py
git commit -m "feat(storage): quality_telemetry table, per-rule counters, doctor section (schema v19)"
```

---

### Task 8: Wire the gate into the pipeline

Where prevention actually happens. `persist_events` is the one function every extraction path reaches.

**Files:**
- Modify: `src/cognikernel/extraction/pipeline.py:432-459` (`persist_events`)
- Test: `tests/unit/extraction/test_pipeline_gate.py`

**Interfaces:**
- Consumes: `admit`, `apply_verdict`, `GroundingContext`, `record_verdict`.
- Produces: `persist_events(events, conn, session_meta=None, ground=None) -> list[int]` — `ground` is a new optional keyword; existing callers are unaffected.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/extraction/test_pipeline_gate.py`:

Uses the shared `conn` fixture from `tests/conftest.py` (see Task 7).

```python
"""The gate is wired into the one function every extraction path reaches."""
import sqlite3

from cognikernel.extraction.pipeline import SessionMetadata, persist_events
from cognikernel.model import Event
from cognikernel.quality.gate import GroundingContext
from cognikernel.storage.quality_telemetry import get_rule_counts


def _event(event_type: str, description: str, **payload) -> Event:
    return Event(
        project_id="p",
        session_id="s",
        event_type=event_type,
        payload={"description": description, **payload},
        content_hash=description[:32] or "h",
        weight=1.0,
    )


_META = SessionMetadata(project_id="p", session_id="s", started_at=0, ended_at=0)


class TestGateWiring:
    def test_rejected_event_is_not_stored(self, conn: sqlite3.Connection) -> None:
        ids = persist_events(
            [_event("CONSTRAINT_HARD", "Pick up the last task as if the break never happened.")],
            conn, _META,
        )
        assert ids == []
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0

    def test_clean_event_is_stored(self, conn: sqlite3.Connection) -> None:
        ids = persist_events(
            [_event("DECISION", "The dispatcher retries twice.")], conn, _META
        )
        assert len(ids) == 1

    def test_rejection_is_counted(self, conn: sqlite3.Connection) -> None:
        persist_events(
            [_event("CONSTRAINT_HARD", "Pick up the last task as if the break never happened.")],
            conn, _META,
        )
        counts = get_rule_counts(conn, "p")
        assert any(c["rule_id"] == "D4" for c in counts)

    def test_subject_less_event_is_stored_but_demoted(self, conn: sqlite3.Connection) -> None:
        # D7 downgrades rather than rejects: still recallable, but weight
        # collapsed so it falls off the budget-ranked block.
        ids = persist_events(
            [_event("DECISION", "It must not take down the pipeline.")], conn, _META
        )
        assert len(ids) == 1
        row = conn.execute("SELECT payload, weight FROM events").fetchone()
        assert "context_dependent" in row[0]
        assert row[1] < 1.0

    def test_unknown_path_is_downgraded_not_dropped(self, conn: sqlite3.Connection) -> None:
        ground = GroundingContext(frozenset({"src/known.py"}))
        ids = persist_events(
            [_event("COMPONENT_STATUS", "x", path="src/brand_new.py")],
            conn, _META, ground=ground,
        )
        assert len(ids) == 1
        row = conn.execute("SELECT payload, weight FROM events").fetchone()
        assert "unverified" in row[0]
        assert row[1] < 1.0

    def test_legacy_call_without_ground_still_works(self, conn: sqlite3.Connection) -> None:
        # Backwards compatibility: existing call sites pass no `ground`.
        ids = persist_events([_event("DECISION", "The queue is durable.")], conn, _META)
        assert len(ids) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/extraction/test_pipeline_gate.py -v`
Expected: FAIL — the defective event is stored (`ids` has one entry) because no gate runs yet.

- [ ] **Step 3: Write minimal implementation**

Replace `persist_events` in `src/cognikernel/extraction/pipeline.py`:

```python
def persist_events(
    events: list[Event],
    conn: sqlite3.Connection,
    session_meta: SessionMetadata | None = None,
    ground=None,
) -> list[int]:
    """Write extracted events to storage. Returns row IDs of inserted/updated rows.

    Every event passes the quality gate first (spec §4). This is the single
    choke point: the sanitize predicates it supersedes were wired into the v1/v2
    head paths only, so the default `legacy` extractor never ran them.

    `ground` is an optional GroundingContext for path referential integrity;
    when None, path checks are skipped and behaviour is unchanged.
    """
    from cognikernel.quality.gate import admit, apply_verdict
    from cognikernel.storage.quality_telemetry import record_verdict

    ids: list[int] = []
    for event in events:
        verdict = admit(event, ground)
        if verdict.rule_id:
            record_verdict(conn, event.project_id, event.session_id, verdict.rule_id)
        if verdict.action == "reject":
            _log.debug(
                "event rejected by quality gate",
                extra={"rule_id": verdict.rule_id, "note": verdict.note},
            )
            continue
        apply_verdict(event, verdict)
        try:
            ids.append(insert_event(conn, event))
        except Exception as exc:
            _log.error(
                "event persist failed",
                extra={"content_hash": event.content_hash, "error": str(exc)},
            )
            if session_meta is not None:
                try:
                    insert_extraction_failure(
                        conn,
                        project_id=event.project_id,
                        session_id=event.session_id,
                        stage="pipeline.persist",
                        error_message=str(exc),
                        raw_input_path="",
                    )
                except Exception:
                    pass
    return ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/extraction/test_pipeline_gate.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and measure the rejection rate**

Run: `.venv/Scripts/python.exe -m pytest tests/unit tests/integration -q`
Expected: PASS. Some existing tests may assert that a defective fixture event round-trips; if one does, it encoded the old behaviour — update it and note it in the commit.

Then confirm the gate's verdict mix on real stored data. The denominator is the **five statement types only**, matching how the §0.1 baseline was computed — counting `COMPONENT_STATUS` and `THREAD_*` rows would dilute the rate and hide over-firing:

```bash
.venv/Scripts/python.exe -c "
import sqlite3, json
from pathlib import Path
from cognikernel.quality.gate import admit
from cognikernel.model import Event
T={'DECISION','CONSTRAINT_HARD','CONSTRAINT_SOFT','APPROACH_ABANDONED','APPROACH_ABANDONED_DO_NOT_RETRY'}
n=rej=dn=0
for db in sorted((Path.home()/'.cognikernel'/'projects').glob('*.db')):
    try: c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
    except Exception: continue
    for et,pl in c.execute('SELECT event_type,payload FROM events WHERE archived=0'):
        if et not in T: continue
        try: p=json.loads(pl)
        except Exception: continue
        n+=1
        a=admit(Event(project_id='p',session_id='s',event_type=et,payload=p,content_hash='h',weight=1.0)).action
        if a=='reject': rej+=1
        elif a=='downgrade': dn+=1
    c.close()
print(f'statements={n}  reject={rej} ({100*rej/max(n,1):.1f}%)  downgrade={dn} ({100*dn/max(n,1):.1f}%)')
"
```

Expected, against the §0.1 baseline: **reject ≈ 0.1-2%** (D2 + D4 only, both small) and **downgrade ≈ 6-7%** (D7). **Stop and tighten if reject exceeds 5% or downgrade exceeds 15%** — that means a detector is over-firing and would suppress real memory. A reject rate near zero is fine and expected; D2/D4 are genuinely rare.

- [ ] **Step 6: Commit**

```bash
git add src/cognikernel/extraction/pipeline.py tests/unit/extraction/test_pipeline_gate.py
git commit -m "feat(extraction): route every event through the quality gate

persist_events is the one function all extraction paths reach, which is why
the gate lives there rather than in the per-path predicates it supersedes."
```

---

### Task 9: Structural invariants at event level (not on the rendered string)

Last line of defence. Two constraints shape where this runs:

1. **Structural only** — no path grounding, because that would be render-time filtering of stored rows, which prevent-only rules out (spec §4).
2. **It must filter events, not lines of the finished block.** `render_injection(ctx, survivors_out=out)` populates `out["events"]` with the set that actually rendered, and `render_with_budget_enforcement_ex` returns it for `storage/render_ledger.py` (which feeds the PreToolUse channel). Dropping lines from the assembled string *after* that set is computed would make the ledger claim an event rendered when its line was removed. So the filter runs over the event lists **before** sections are assembled, and the survivors set is correct by construction.

**Files:**
- Modify: `src/cognikernel/injection/template.py` (`render_injection`)
- Test: `tests/unit/injection/test_render_invariants.py`

**Interfaces:**
- Consumes: `normalized_key`, `BOX_DRAWING_RE` (Tasks 2-3), `Event`.
- Produces: `filter_structural_defects(events: list[Event]) -> list[Event]`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/injection/test_render_invariants.py`:

```python
"""Structural invariants over the event set before rendering (spec §4)."""
from cognikernel.injection.template import filter_structural_defects
from cognikernel.model import Event


def _event(event_type: str, description: str, chash: str) -> Event:
    return Event(
        project_id="p",
        session_id="s",
        event_type=event_type,
        payload={"description": description},
        content_hash=chash,
        weight=1.0,
    )


class TestStructuralInvariants:
    def test_drops_box_drawing_event(self) -> None:
        events = [
            _event("DECISION", "Use SQLite for local state.", "a"),
            _event("DECISION", "┌────┬────┐ │ Layer │ Choice │", "b"),
        ]
        out = filter_structural_defects(events)
        assert len(out) == 1
        assert out[0].content_hash == "a"

    def test_drops_cross_type_duplicate_keeping_first(self) -> None:
        events = [
            _event("APPROACH_ABANDONED", "Record Celery as abandoned.", "a"),
            _event("CONSTRAINT_HARD", "record celery as abandoned!", "b"),
        ]
        out = filter_structural_defects(events)
        assert len(out) == 1
        assert out[0].content_hash == "a"

    def test_keeps_distinct_statements(self) -> None:
        events = [
            _event("DECISION", "Use Postgres.", "a"),
            _event("DECISION", "Use Redis for the cache.", "b"),
        ]
        assert len(filter_structural_defects(events)) == 2

    def test_does_not_ground_paths(self) -> None:
        # Prevent-only: a legacy truncated path still renders. Grounding is the
        # admission gate's job and must never run here.
        e = _event("COMPONENT_STATUS", "rc/storage/connection.py modified", "a")
        e.payload["path"] = "rc/storage/connection.py"
        assert filter_structural_defects([e]) == [e]

    def test_empty_input_is_empty_output(self) -> None:
        assert filter_structural_defects([]) == []

    def test_never_raises_on_malformed_event(self) -> None:
        e = _event("DECISION", "fine", "a")
        e.payload = {}          # no description at all
        assert filter_structural_defects([e]) == [e]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/injection/test_render_invariants.py -v`
Expected: FAIL — `ImportError: cannot import name 'filter_structural_defects'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/cognikernel/injection/template.py`:

```python
def filter_structural_defects(events: list[Event]) -> list[Event]:
    """Drop malformed and cross-type-duplicate events before rendering.

    Runs over the EVENT SET, not the assembled string, so that survivors_out
    (and therefore the render ledger) reflects what actually rendered. Dropping
    lines from the finished block would desynchronize them.

    STRUCTURAL ONLY. This deliberately does NOT check path grounding: doing so
    would filter already-stored rows at render time, which the design rules out
    in favour of prevent-only (spec §4). Grounding belongs to the admission
    gate, which only ever sees new events.

    Never raises — on any error the input list is returned unchanged, because a
    slightly malformed block beats no context at all.
    """
    if not events:
        return events
    try:
        from cognikernel.quality.detectors import BOX_DRAWING_RE, normalized_key

        seen: set[str] = set()
        kept: list[Event] = []
        for event in events:
            description = (event.payload or {}).get("description", "") or ""
            if BOX_DRAWING_RE.search(description):
                _log.debug("render invariant: dropped box-drawing event")
                continue
            key = normalized_key(description)
            if key:
                if key in seen:
                    _log.debug("render invariant: dropped cross-type duplicate")
                    continue
                seen.add(key)
            kept.append(event)
        return kept
    except Exception as exc:
        _log.warning("render invariant check failed — passing events through: %s", exc)
        return events
```

Add a public alias to `detectors.py` (below the `_BOX_DRAWING` definition) so the template does not import a private name across modules:

```python
# Public alias — the render-time invariant check in injection/template.py
# reuses this rather than defining a second box-drawing pattern.
BOX_DRAWING_RE = _BOX_DRAWING
```

Add it to `__init__.py`'s imports and `__all__` too.

Then apply the filter **at the top of `render_injection`**, before any section is built and before `survivors_out` is populated. Near the start of the function body, filter each event list the renderer consumes:

```python
    ctx = copy.copy(ctx)
    ctx.decisions = filter_structural_defects(ctx.decisions)
    ctx.components = filter_structural_defects(ctx.components)
    ctx.hard_constraints = filter_structural_defects(ctx.hard_constraints)
    ctx.graveyard = filter_structural_defects(ctx.graveyard)
```

Match the actual attribute names on `InjectionContext` — read the dataclass first and filter every event-list field it exposes except `active_threads`, which is protected from dropping elsewhere in this module and should stay that way.

Do **not** touch line 138 (`return "\n\n".join(s for s in sections if s)`); the string stays untouched by design.

Ensure `template.py` has `import copy`, `import logging`, and a module-level `_log = logging.getLogger("cognikernel.injection")`; `copy` and `_log` are already used by `render_with_budget_enforcement_ex`, so most likely only the check is needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/injection/test_render_invariants.py -v`
Expected: PASS

- [ ] **Step 5: Check golden-file render tests**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/injection -q`
Expected: PASS. If a golden fixture contained a duplicate line, the invariant now removes it — that is the intended behaviour; update the golden file and note it.

- [ ] **Step 6: Commit**

```bash
git add src/cognikernel/injection/template.py src/cognikernel/quality/detectors.py tests/unit/injection/test_render_invariants.py
git commit -m "feat(injection): structural invariants on the rendered block

Structural only by design: path grounding here would be render-time filtering
of stored rows, which prevent-only explicitly declined."
```

---

### Task 10: Audit harness, fixture corpus, and the CI regression gate

Turns the exploratory scratchpad sweep into reproducible research tooling and locks the gains in.

**Files:**
- Create: `scripts/injection_defect_audit.py`
- Create: `tests/fixtures/transcripts/defects/*.txt` (5 files)
- Create: `tests/unit/quality/test_baseline_regression.py`
- Create: `docs/metrics/injection_defect_baseline.json`
- Test: as above

**Interfaces:**
- Consumes: all detectors.
- Produces: `sweep_store(db_path: Path) -> tuple[Counter, Counter]` (hits, denominators), `wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]`, and the committed baseline JSON.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/quality/test_baseline_regression.py`:

```python
"""CI gate: detector hit counts on the committed fixture corpus may not rise."""
import json
from pathlib import Path

import pytest

from cognikernel.quality.detectors import (
    detect_boilerplate,
    detect_junk_constraint,
    detect_subject_less,
)

_FIXTURES = Path(__file__).parents[2] / "fixtures" / "transcripts" / "defects"
_BASELINE = Path(__file__).parents[3] / "docs" / "research" / "injection_defect_baseline.json"


class TestWilsonInterval:
    def test_interval_brackets_the_point_estimate(self) -> None:
        from scripts.injection_defect_audit import wilson_interval

        lo, hi = wilson_interval(30, 100)
        assert lo < 0.30 < hi

    def test_zero_hits_has_zero_lower_bound(self) -> None:
        from scripts.injection_defect_audit import wilson_interval

        lo, _ = wilson_interval(0, 100)
        assert lo == pytest.approx(0.0, abs=1e-9)

    def test_empty_denominator_is_zero_zero(self) -> None:
        from scripts.injection_defect_audit import wilson_interval

        assert wilson_interval(0, 0) == (0.0, 0.0)


class TestFixtureCorpus:
    def test_every_defect_fixture_is_detected(self) -> None:
        """Each fixture file instantiates its class; the detector must catch it."""
        cases = {
            "d2_junk_constraint.txt": lambda t: detect_junk_constraint(t, "CONSTRAINT_HARD"),
            "d4_boilerplate.txt": detect_boilerplate,
            "d7_subject_less.txt": detect_subject_less,
        }
        for name, detector in cases.items():
            text = (_FIXTURES / name).read_text(encoding="utf-8").strip()
            assert detector(text) is not None, f"{name} not detected"

    def test_clean_fixture_is_not_flagged(self) -> None:
        text = (_FIXTURES / "clean.txt").read_text(encoding="utf-8").strip()
        assert detect_subject_less(text) is None
        assert detect_boilerplate(text) is None
        assert detect_junk_constraint(text, "CONSTRAINT_HARD") is None


class TestBaselineGate:
    def test_baseline_file_exists_and_is_valid(self) -> None:
        data = json.loads(_BASELINE.read_text(encoding="utf-8"))
        assert "classes" in data
        assert "generated_at" in data
```

- [ ] **Step 2: Create the fixture corpus**

Create these files under `tests/fixtures/transcripts/defects/`. Text is drawn from the real store sweep so the fixtures represent observed defects, not invented ones.

`d2_junk_constraint.txt`:
```
For the queue between worker and dispatcher, should we just bring in Celery + Redis?
```

`d4_boilerplate.txt`:
```
Pick up the last task as if the break never happened.
```

`d7_subject_less.txt`:
```
It must not be able to take down the pipeline.
```

`clean.txt`:
```
Do not cache in Redis, because that adds a network hop and defeats the point.
```

`d1_legacy_path.txt` (documentation of the retired class; not asserted against a live detector):
```
rc/storage/connection.py modified
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/quality/test_baseline_regression.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.injection_defect_audit'`

- [ ] **Step 4: Write the audit script**

Create `scripts/__init__.py` if absent (empty file, so the tests can import the module).

Create `scripts/injection_defect_audit.py`:

```python
"""Read-only defect sweep over every local CogniKernel store.

Research tooling — NOT part of the shipped package. Produces the prevalence
baseline that the paper's "before" column and the CI regression gate both use.

Usage:
    python scripts/injection_defect_audit.py [--out docs/metrics/injection_defect_baseline.json]

Never writes to a store: every connection is opened read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from cognikernel.quality.detectors import (
    detect_boilerplate,
    detect_junk_constraint,
    detect_subject_less,
    normalized_key,
)

STATEMENT_TYPES = (
    "DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
    "APPROACH_ABANDONED", "APPROACH_ABANDONED_DO_NOT_RETRY",
)


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because defect rates are small and
    the normal interval misbehaves (and can go negative) near zero.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def sweep_store(db_path: Path) -> tuple[Counter, Counter]:
    """Return (hits_by_rule, denominators_by_rule) for one store. Read-only."""
    hits: Counter = Counter()
    denom: Counter = Counter()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT event_type, payload FROM events WHERE archived = 0"
        ).fetchall()
    except Exception:
        return hits, denom

    keys_by_norm: dict[str, set[str]] = defaultdict(set)
    for event_type, raw in rows:
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        description = (payload.get("description") or "").strip()
        if event_type not in STATEMENT_TYPES or not description:
            continue

        denom["D7"] += 1
        if detect_subject_less(description):
            hits["D7"] += 1

        denom["D4"] += 1
        if detect_boilerplate(description):
            hits["D4"] += 1

        if event_type.startswith("CONSTRAINT"):
            denom["D2"] += 1
            if detect_junk_constraint(description, event_type):
                hits["D2"] += 1

        key = normalized_key(description)
        if key:
            keys_by_norm[key].add(event_type)

    for key, types in keys_by_norm.items():
        denom["D5"] += 1
        if len(types) > 1:
            hits["D5"] += 1

    conn.close()
    return hits, denom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects-dir",
        default=str(Path.home() / ".cognikernel" / "projects"),
    )
    parser.add_argument("--out", default="docs/metrics/injection_defect_baseline.json")
    args = parser.parse_args()

    stores = sorted(Path(args.projects_dir).glob("*.db"))
    total_hits: Counter = Counter()
    total_denom: Counter = Counter()
    stores_affected: dict[str, set] = defaultdict(set)

    for db in stores:
        hits, denom = sweep_store(db)
        total_hits.update(hits)
        total_denom.update(denom)
        for rule in hits:
            stores_affected[rule].add(db.name)

    classes = {}
    for rule in sorted(set(total_denom) | set(total_hits)):
        n = total_denom.get(rule, 0)
        h = total_hits.get(rule, 0)
        lo, hi = wilson_interval(h, n)
        classes[rule] = {
            "hits": h,
            "denominator": n,
            "rate": (h / n) if n else 0.0,
            "wilson_95": [lo, hi],
            "stores_affected": len(stores_affected.get(rule, ())),
        }

    report = {
        "generated_at": int(time.time()),
        "stores_scanned": len(stores),
        "classes": classes,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"stores scanned: {len(stores)}")
    print(f"{'rule':6} {'hits':>7} {'denom':>8} {'rate':>8}  95% CI")
    print("-" * 56)
    for rule, c in classes.items():
        lo, hi = c["wilson_95"]
        print(f"{rule:6} {c['hits']:>7} {c['denominator']:>8} "
              f"{100*c['rate']:>7.2f}%  [{100*lo:.2f}%, {100*hi:.2f}%]")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Generate the baseline and run the tests**

Run:
```bash
.venv/Scripts/python.exe scripts/injection_defect_audit.py
.venv/Scripts/python.exe -m pytest tests/unit/quality/test_baseline_regression.py -v
```
Expected: the script prints a per-rule table with Wilson CIs and writes `docs/metrics/injection_defect_baseline.json`; tests PASS.

**Note:** this baseline is measured with the *new* detectors against *existing stored* events, so it is the honest "before" for the paper — it says how much defective memory is currently in the stores, which is exactly the prevalence claim. It is not the A/B; that compares old vs new extraction over the recovered and fixture corpora (spec §5 tier 1) and is out of scope for this plan.

- [ ] **Step 6: Run the whole suite and commit**

Run: `.venv/Scripts/python.exe -m pytest -q && ./.venv/Scripts/lint-imports.exe`
Expected: PASS, all import contracts KEPT.

```bash
git add scripts/injection_defect_audit.py scripts/__init__.py tests/fixtures/transcripts/defects tests/unit/quality/test_baseline_regression.py docs/metrics/injection_defect_baseline.json
git commit -m "feat(research): defect audit harness, fixture corpus, and baseline

Sweeps every local store read-only, reports per-rule prevalence with Wilson
intervals, and commits the baseline JSON that the CI regression gate and the
paper's prevalence table both read."
```

---

## Verification checklist (run before declaring the branch done)

- [ ] `.venv/Scripts/python.exe -m pytest -q` — full suite green
- [ ] `./.venv/Scripts/lint-imports.exe` — all contracts KEPT, including "Quality is a leaf"
- [ ] Gate rejection rate on real stores is single-digit percent (Task 8, Step 5)
- [ ] `_discover_project_paths(Path('.'))` returns no vendored paths (Task 5, Step 5)
- [ ] `docs/metrics/injection_defect_baseline.json` committed
- [ ] A v18 store opens and upgrades to v19 without error

## Deviations from the spec (decided while planning — read before implementing)

Five items diverge from the approved spec — four narrowings and one addition. The reasoning belongs with the plan rather than in a commit message nobody re-reads.

**1. D5 is deduplicated at render time, not by write-time cross-type suppression.** Spec §3 says "content-hash uniqueness checked across all event types at write time." Not done, because the gate sees one event at a time and has no principled way to choose *which* of two same-text events survives: suppressing the second arrival means a `CONSTRAINT_HARD` can lose to a `DECISION` purely on extraction order, which trades a visible duplicate for a silent loss of the more authoritative row. Task 9 removes the user-visible symptom — the same fact appearing in two sections — with no row deleted from the store. Note it filters the *event set* before section assembly rather than the rendered string, so `survivors_out` and the render ledger stay accurate; a string-level filter would have made the ledger claim an event rendered when its line had been removed. If D5 stays material after these fixes, revisit with an explicit type-priority rule rather than first-writer-wins.

**2. D7 downgrades instead of rejecting.** The spec's §3 wording ("the subject-presence heuristic gates admission") reads as rejection. Changed to downgrade after finding that `pipeline.py` already states the opposite policy for the same class of statement — *"We DEMOTE (not drop): weight collapses so they fall off the budget-ranked block while staying in the store"* — with `_FRAG_DEMOTE = 0.4`. The harm D7 causes is occupying the budget-ranked block, and weight collapse fixes exactly that; rejecting would additionally remove the statement from `recall` and `find_related`, which is strictly more destructive for a memory system and would contradict a reasoned decision already in the file. D2 and D4 still reject, because box-drawing artifacts and harness boilerplate carry no recoverable content. The gate skips its downgrade when the v1/v2 head path already demoted the event (`provenance` contains `+frag`) so the multiplier is never applied twice.

**3. The CI regression gate asserts detection, not counts.** Task 10 specified "the build fails if any class's count rises above the committed baseline." It cannot: the 163 stores are private, so CI only ever sees the five committed fixture files, and a count gate over five files is noise. `test_baseline_regression.py` instead asserts that every defect fixture is still detected, that the clean fixture is still not flagged, and that the baseline JSON's Wilson intervals are internally consistent (each class's rate lies inside its own interval). Count-based regression detection stays a local operation via `scripts/injection_defect_audit.py`.

**4. Out-of-scope symbol pruning was ADDED (not in the approved plan).** Without it the D3 fix does nothing for anyone who already has a store: `build_symbol_update` emits `delete_paths` only for git-*deleted* files, so nodes for paths that leave scope persist forever and keep consuming the skeleton budget. `prune_out_of_scope_symbols` completes that invalidation rule. This is consistent with prevent-only because the symbol graph is a **derived cache** rebuilt by scanning disk — pruned paths reappear if they re-enter scope, and no decision, constraint, or abandonment is touched. It keys on the *exclusion predicate* (skip-dir component or gitignore match), never on set-difference against a discovery run: discovery caps at `_MAX_FILES = 500`, so a project with more in-scope files than the cap has legitimate modules absent from any single run, and "delete what wasn't discovered" would permanently purge them. `test_does_not_prune_by_set_difference` pins that.

**5. Windowing span-contiguity enforcement (spec §3, D6/D7) is not implemented.** The spec proposed changing `windowing.py` so descriptions cannot splice non-adjacent sentences. §0.1 then measured D6 at 0.1% in 2 stores and found its actual cause was `normalize.py:81` capitalization, which Task 3 fixes directly. Changing windowing's span logic is high-blast-radius surgery on the main extraction path to chase a 0.1% class whose diagnosis has already moved. D7 — the class that actually matters at 6.5% — is addressed at the gate, which catches subject-less statements regardless of how windowing assembled them. Spec §8 already flags "does D6 survive as a class" as open; this plan answers "not yet."

## Deliberately out of scope

- Statement *generation* — separate gated branch (`2026-07-25-memory-statement-generation-design.md`).
- Repair of existing stores — prevent-only by decision; legacy defects decay out.
- The offline A/B (spec §5 tier 1) — needs the recovered-corpus tooling; plan it separately once these fixes land, since the A/B measures *these* changes.
- Retrieval relevance and speed/granularity — the next two cycles.
