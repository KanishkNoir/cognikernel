"""Decision keys (J2.2) — the normalized topic axis a choice-family event is about.

ONE derivation replaces the three scattered subject implementations
(extraction/authority.normalize_subject, delta/supersede.derive_subject,
extraction/triple typed triples) as the *key* authority. The projection groups
same-key DECISION/CONSTRAINT_HARD/CONSTRAINT_SOFT events and renders the
latest highest-authority value as canonical (latest-wins READ).

Honesty rule: a WRONG key is worse than NO key — an over-general key silently
demotes an unrelated fact at read time. Every candidate source here is
deterministic and structural; there is no head-noun guessing. '' (no key)
means the event participates exactly as before: pairwise supersession +
weight ranking. Nothing regresses on keyless events.

Candidate ladder (first non-empty wins):
  1. payload['subject']         — pattern extraction already names the topic.
  2. payload['triple']          — ¬: the prohibited thing IS the topic;
                                  →/∅: the subject side.
  3. utils.subject.derive_subject — decision-verb topic regexes ("switch X for Y").
  4. label prefix               — "Retry: 2 attempts…" names its own axis; the
                                  label-value register carries the key in the
                                  label (same register the F8 backstop mints).

Lives in utils (a dependency-free base) so the delta merge can mint keys without
importing the extraction stage — the extraction<->delta layering cycle the
architecture audit surfaced. extraction.decision_key re-exports it.
"""
from __future__ import annotations

import re

from cognikernel.utils.subject import STOPWORDS, derive_subject

CHOICE_FAMILY: frozenset[str] = frozenset({
    "CONSTRAINT_HARD",
    "CONSTRAINT_SOFT",
    "DECISION",
})

_KEY_MAX_TOKENS = 4

# Label prefix: a short leading label terminated by a colon ("Retry:",
# "Cost unit:", "Key format —"). Mirrors tokenize.is_label_value_line's
# register without importing it (that predicate also checks the VALUE side;
# here only the label names the axis). ≤4 words, no sentence verbs.
_LABEL_PREFIX_RE = re.compile(r"^([A-Za-z_][\w .()/+-]{0,40}?)\s*[:—]\s+\S")
_LABEL_MAX_WORDS = 4
# Clause-starters that make a "Label:" actually a sentence lead-in, not an axis
# ("Note:", "Example:", "Important:"). Mirrors tokenize._LABEL_STOP_FIRST.
_LABEL_STOPWORDS = frozenset({
    "note", "example", "warning", "important", "remember", "see", "however",
    "result", "summary", "update", "edit", "fix", "todo", "caveat", "answer",
    "question", "step", "output", "input", "error", "before", "after", "now",
    # Measured junk labels (gamma): "New constraint: …" → 'new', "Three
    # changes: …" → 'change three'. Sentence lead-ins, not topic axes.
    "new", "old", "change", "changes", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten", "first", "second", "third",
})


def derive_decision_key(payload: dict, event_type: str) -> str:
    """Normalized topic key for a choice-family event; '' when none derivable."""
    if event_type not in CHOICE_FAMILY:
        return ""

    cand = (payload.get("subject") or "").strip()

    if not cand:
        triple = payload.get("triple")
        if isinstance(triple, dict):
            if triple.get("operator") == "¬":
                cand = (triple.get("object") or triple.get("subject") or "").strip()
            else:
                cand = (triple.get("subject") or "").strip()

    description = payload.get("description", "") or ""
    if not cand:
        cand = derive_subject(description)

    if not cand:
        cand = _label_prefix(description)

    return normalize_key(cand)


def normalize_key(text: str) -> str:
    """Canonical key form: lowercase, content tokens only, sorted, capped.

    Sorting makes 'default alias' ≡ 'alias default'. Singularization is the
    conservative consonant+s rule only — naive trailing-s stripping would
    corrupt exactly the identifiers keys exist for ('alias' → 'alia').
    """
    if not text:
        return ""
    s = re.sub(r"[^\w\s]", " ", text.lower())
    toks = []
    for t in s.split():
        if len(t) <= 2 or t in STOPWORDS:
            continue
        if (
            len(t) > 4
            and t.endswith("s")
            and not t.endswith(("ss", "us", "is", "as", "os"))
        ):
            t = t[:-1]
        toks.append(t)
    return " ".join(sorted(set(toks))[:_KEY_MAX_TOKENS])


def _label_prefix(description: str) -> str:
    m = _LABEL_PREFIX_RE.match(description.strip())
    if not m:
        return ""
    label = m.group(1).strip()
    words = label.split()
    if not words or len(words) > _LABEL_MAX_WORDS:
        return ""
    if words[0].lower().strip(".,") in _LABEL_STOPWORDS:
        return ""
    return label


def backfill_keys(conn, project_id: str) -> int:
    """Lazily derive keys for pre-016 rows (decision_key IS NULL). Idempotent.

    Writes '' (not NULL) when derivation finds nothing, so a project is
    rescanned at most once. O(active choice-family events); called from
    rebuild_projection so existing DBs heal on first render with zero user
    action.
    """
    import json

    rows = conn.execute(
        "SELECT id, event_type, payload FROM events "
        "WHERE project_id = ? AND decision_key IS NULL",
        (project_id,),
    ).fetchall()
    if not rows:
        return 0
    updates = []
    for r in rows:
        try:
            payload = json.loads(r["payload"])
        except Exception:
            payload = {}
        updates.append((derive_decision_key(payload, r["event_type"]), r["id"]))
    conn.executemany("UPDATE events SET decision_key = ? WHERE id = ?", updates)
    conn.commit()
    return len(updates)
