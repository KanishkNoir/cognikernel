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
