"""Admission control for extracted events — the single choke point.

Every extracted event passes through admit() immediately before persistence.
The gate returns one of three verdicts:

  admit      store as-is
  downgrade  store with halved weight and a payload marker
  reject     do not store

WHY ONE CHOKE POINT. extraction/sanitize.py already ships
is_context_dependent_fragment() and is_question_description(), but they are
wired in only partially: the fragment predicate runs in the v1/v2 salience-head
paths (pipeline.py) and NOT in the default `legacy` extractor, and the question
predicate guards CONSTRAINT_HARD in windowing.py and nothing else. That is why
subject-less statements measured 6.5% of all statements across 34 of 163
stores. The fix is not more call sites — it is one function every extraction
path reaches.

VERDICTS ARE GRADED BY RECOVERABILITY. D2 and D4 reject: box-drawing artifacts
and harness boilerplate carry no project content, so keeping them helps nobody.
D7 downgrades: a subject-less statement still carries a real fact, just one the
reader cannot resolve, and its harm is occupying the budget-ranked block —
which weight collapse fixes while leaving it reachable through recall and
find_related. This mirrors the policy pipeline.py already states for the same
class of statement ("We DEMOTE (not drop)").

FAILURE POSTURE: the gate never blocks a session. Any exception inside a
detector produces an 'admit' verdict tagged rule_id='gate_error', which the
caller counts. Losing a defect is acceptable; losing a session is not.

PURITY: this module takes the path inventory as an argument rather than
reading the symbol store, which keeps cognikernel.quality a leaf package
(see the "Quality is a leaf" contract in pyproject.toml).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cognikernel.model import Event
from cognikernel.quality.detectors import (
    detect_bare_instruction_thread,
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

# Where a D9-demoted thread lands. `assistant_decided` is the next tier down
# from `user_stated` in extraction.authority's precedence table, and is a
# legitimate, non-junk tier — the event stays a real, retrievable thread, it
# just no longer competes with what the user actually said is outstanding.
# A literal, not an import: quality is a leaf package (see detectors.py).
_DEMOTED_THREAD_AUTHORITY = "assistant_decided"

# Maximum number of leading characters that may be missing for a path to count
# as a truncation of a known path rather than an unrelated new file.
_NEAR_MISS_MAX_MISSING = 2


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

    `verifiable_suffixes` bounds what the inventory can speak to. The discovery
    walk globs only the languages we can parse, so a real `internal/db/pool.go`
    is absent from `known_paths` for a reason that has nothing to do with
    whether it exists. Grounding it would mark every component event in a
    Go/Rust/Java project 'unverified' and halve its weight, pushing the whole
    project off the budget-ranked block. An empty set means "verify
    everything", preserving the behaviour of callers that supply no suffixes.
    """
    known_paths: frozenset[str] = field(default_factory=frozenset)
    verifiable_suffixes: frozenset[str] = field(default_factory=frozenset)

    def is_known(self, path: str) -> bool:
        return path in self.known_paths

    def can_verify(self, path: str) -> bool:
        """False when the inventory has no authority over this path's language.

        'Cannot verify' is not 'does not exist' — the same distinction the gate
        already draws for an empty inventory.
        """
        if not self.verifiable_suffixes:
            return True
        dot = path.rfind(".")
        return dot != -1 and path[dot:].lower() in self.verifiable_suffixes

    def is_near_miss(self, path: str) -> bool:
        """True when `path` is a known path with leading characters removed.

        This is the 2026-05-10 corruption signature. A store sweep found no
        live generator of it, so this is defence in depth rather than a tuned
        classifier — which is why the threshold is a flat 1-2 characters and
        not an edit distance worth calibrating.
        """
        if not path or self.is_known(path):
            return False
        for known in self.known_paths:
            missing = len(known) - len(path)
            if 0 < missing <= _NEAR_MISS_MAX_MISSING and known.endswith(path):
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

    # D4 is TYPE-INDEPENDENT. Harness and compaction chatter is never a project
    # fact, whatever type the classifier assigned it. Found end-to-end: one
    # compaction sentence was extracted twice, and scoping D4 to statement types
    # rejected the CONSTRAINT_HARD copy while the THREAD_OPEN copy was stored.
    hit = detect_boilerplate(description)
    if hit is not None:
        return Verdict("reject", hit.rule_id, hit.note)

    if event.event_type in _STATEMENT_TYPES:
        # Reject: nothing recoverable in a constraint slot holding a non-proposition.
        hit = detect_junk_constraint(description, event.event_type)
        if hit is not None:
            return Verdict("reject", hit.rule_id, hit.note)

        # Downgrade: real fact, unresolvable referent. Skipped when a head path
        # already demoted this event (provenance carries '+frag') so the
        # multiplier is never applied twice.
        if "+frag" not in (payload.get("provenance") or ""):
            hit = detect_subject_less(description)
            if hit is not None:
                return Verdict("downgrade", hit.rule_id, hit.note)

    # D9 (T-202a / #20 Defect A): an ordinary user instruction typed THREAD_OPEN
    # arrives at the TOP authority tier purely because a user said it, and then
    # outranks a genuinely-queued thread. Downgrade, never reject: the marker
    # vocabulary this rests on is a heuristic, and a demoted event stays fully
    # reachable through recall while a dropped one is gone for good.
    hit = detect_bare_instruction_thread(
        description, event.event_type, payload.get("authority", "") or ""
    )
    if hit is not None:
        return Verdict("downgrade", hit.rule_id, hit.note)

    # An EMPTY inventory means "cannot verify", not "nothing is real". A brand-new
    # project has no symbol graph yet, and grounding against an empty set would
    # downgrade every component event it ever captured.
    if event.event_type in _PATH_TYPES and ground is not None and ground.known_paths:
        path = payload.get("path", "") or ""
        if path and ground.can_verify(path):
            if ground.is_near_miss(path):
                return Verdict("reject", "D1", "path is a truncation of a known path")
            if not ground.is_known(path):
                return Verdict("downgrade", "D1", "path not found in codebase inventory")

    return Verdict("admit")


def apply_verdict(event: Event, verdict: Verdict) -> Event:
    """Mutate `event` per a downgrade verdict and return it.

    Only 'downgrade' changes the event; 'admit' and 'reject' leave it alone
    (the caller drops rejects). The marker key differs by rule so the
    downgrade reasons stay distinguishable in the store:
      D1 -> payload['grounding'] = 'unverified'   (path not in the codebase)
      D7 -> payload['quality']   = 'context_dependent'
      D9 -> payload['authority'] demoted          (instruction, not a thread)

    D9 is the one rule whose point is the AUTHORITY, not the weight. Thread
    selection ranks by (authority_priority, -weight), so an ordinary
    instruction sharing the top `user_stated` tier beats a genuine queued
    thread on weight alone — halving its weight would only have made that a
    closer race, not stopped it. Demoting the tier stops it outright: a
    genuine `user_stated` thread now wins regardless of weight. `source_role`
    is left untouched, so the record of who actually said it survives — this
    changes precedence, not provenance.
    """
    if verdict.action != "downgrade":
        return event
    event.weight = (event.weight or 1.0) * _DOWNGRADE_FACTOR
    if verdict.rule_id == "D1":
        event.payload["grounding"] = "unverified"
    elif verdict.rule_id == "D9":
        event.payload["authority"] = _DEMOTED_THREAD_AUTHORITY
        event.payload["quality"] = "instruction_not_thread"
    else:
        event.payload["quality"] = "context_dependent"
    return event
