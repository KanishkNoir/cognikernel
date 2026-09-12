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


# ── D2: junk stored as a constraint ──────────────────────────────────────────

_BOX_DRAWING = re.compile(r"[─-╿▀-▟]")

# Public alias — the render-time invariant check in injection/template.py
# reuses this rather than defining a second box-drawing pattern.
BOX_DRAWING_RE = _BOX_DRAWING

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
# Phrases emitted by the agent harness (compaction summaries, resume banners),
# by CogniKernel's own injected block, or by the agent narrating a check of
# CogniKernel's own state (e.g. echoing CLAUDE.md/trust-header "greenfield"
# guidance back as if it were a project fact). None of these are project facts.

_BOILERPLATE = re.compile(
    r"read the full transcript at|continue the conversation from|"
    r"resume directly|do not acknowledge the summary|"
    r"do not recap what was happening|pick up the last task|"
    r"session context \[auto-generated|as if the break never happened|"
    r"no prior decisions stored|confirmed greenfield",
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


# ── R1: memory-meta narration ────────────────────────────────────────────────
#
# The assistant narrating CogniKernel's OWN memory ("the session context
# flagged…", "CogniKernel's Stop hook will persist it") instead of stating a
# project fact. Extraction demotes these (pipeline._META_DEMOTE). The list lives
# here, in the leaf, so the ranking can read the same predicate.
#
# Precision first. Measured 2026-09-12 on all 162 claims the earlier list had
# tagged in the local stores: 30 were real project facts, from three classes —
# a project that discusses CogniKernel as a design subject ("vendor
# CogniKernel's event-sourced store"), "from memory" as an ordinary phrase
# ("re-emit the stored JSON as chunks from memory"), and a project's own
# "graveyard" of rejected ideas. Each alternative below is a strict narrowing of
# the one it replaced, checked over all 12,149 stored events: nothing matches now
# that did not match before. It still matches 6 real facts that quote memory
# ("that's the hard constraint (CONSTRAINT_HARD: …)"), and no longer matches 27
# narration claims, only 2 of which were weight-ranked. A miss leaves a claim
# undemoted; a false match demotes a real fact, which is the costlier error.
MEMORY_META_RE = re.compile(
    r"\b(?:session[- ]context|injection block|injected (?:session )?context|"
    # CogniKernel acting as the tool, not as a subject of the project's own design.
    r"cognikernel(?:'s|’s)?\s+(?:stop hook|(?:userpromptsubmit |skeleton[- ]gate )?hooks?|mcp|"
    r"skeleton(?:[- ]gate| tool)?|session[- ]?(?:context|state)|memory(?!\s+store)|recall|tools?|"
    r"says|(?:has |had )?flagged|agrees|captures|will (?:record|persist|extract|pick)|scaffold\w*|"
    r"config\w*|(?:stored )?(?:project )?state|read-gate|strict mode|is blocking|doesn't expose)|"
    r"stop hook|pending confirmation|memory confirms|recorded in memory|"
    # "from memory" only as framing or retrieval, never the ordinary phrase.
    r"from memory\s*:|(?:pull\w*|recall\w*|retriev\w*|check\w*)\b[^.]{0,60}\bfrom memory|"
    # The graveyard as CogniKernel's section, not a project's word for rejected ideas.
    r"(?:record\w*|extract\w*|persist\w*)\b[^.]{0,40}\bgraveyard|as a graveyard entry|"
    r"the recall (?:surfaces|surfaced|returns|returned|results|mentions|shows|tool)|"
    # Compaction-summary instructions leaking into transcripts.
    r"resume directly|do not acknowledge the summary|do not recap what was happening|"
    r"continue the conversation from where it left off|"
    # Event-type tokens narrated in prose; underscore forms only, so "hard constraint" stays.
    r"approach_abandoned\w*|constraint_hard|constraint_soft|thread_open|component_status|"
    # Supersession governance narration, not the superseded fact itself.
    r"(?:now|explicitly) superseded|rejection is superseded|superseded abandoned|"
    # Memory-reference framing around a fact whose canonical capture exists separately.
    r"memory (?:shows|says)|recorded decision|decision to record|decision log|"
    r"locked in the project memory|prior decision being overridden|entry recording|"
    # The MCP server instructions themselves.
    r"call recall|missing from the block)(?!\w)",
    re.IGNORECASE,
)


def is_memory_meta(text: str) -> bool:
    """R1 — the text narrates CogniKernel's own memory, not the project."""
    return bool(MEMORY_META_RE.search(text or ""))


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


# ── D9: an ordinary instruction typed as a user-stated thread ────────────────
#
# T-202a (#20 Defect A). A THREAD_OPEN answers "what is queued / what were we
# in the middle of". An ordinary imperative from the user — "add the Pydantic
# response schema for a task", "write the cache lookup" — is not that: it is
# work for the current turn, done by the end of it. But it arrives carrying
# `user_stated`, the TOP authority tier, because authority is assigned from
# the speaker's role alone and never looks at what kind of statement it is.
#
# Measured consequence (research/fixes/thread_precision.md): in the real
# Taskflow store, "Add the Pydantic response schema for a task." (user_stated,
# w=0.88) outranked the genuinely-queued JWT thread (user_stated, w=0.81) —
# same tier, so weight alone decided — and the graded probe recorded the wrong
# thread as the answer.
#
# The discriminator is NOT "is this imperative mood": genuine deferrals are
# routinely imperative too ("come back to the retry logic next session"). It is
# whether the statement refers to future, queued, outstanding or in-progress
# work at all. These markers were validated against all 131 real THREAD_OPEN
# events from the four benchmark stores: both genuine Taskflow threads keep
# their tier, and all nine non-threads (bare imperatives, spec/label lines,
# condition statements) are caught. The first draft of this list missed plain
# "need to" and would have demoted the real gold thread — hence the deliberate
# inclusion of obligation phrasing below.
_DEFERRAL_MARKERS = (
    # explicit future time / session reference
    "next session", "next time", "later", "tomorrow", "next week",
    # queued, pending or outstanding state
    "queued", "queue", "pending", "on hold", "parked", "park ",
    "backlog", "not yet", "unfinished", "incomplete", "remaining",
    "left to do", "still need", "still needs", "still to",
    # obligation phrasing — the commonest way a real deferred thread is stated
    # ("we need to implement JWT auth end-to-end"), and the shape the gold
    # Taskflow thread used.
    "need to", "needs to", "have to", "has to", "must still",
    # explicit resumption
    "come back to", "pick up", "picks up", "picking up", "resume",
    "continue with", "continuing with", "next step", "next up",
    "follow up", "follow-up",
    # in-progress state
    "in progress", "working on", "work item", "work thread", "open thread",
    "active thread",
    # stated intent to do
    "will implement", "will add", "will need", "will write", "going to",
    "todo", "to do", "flagging", "flagged as",
)
_DEFERRAL_RE = re.compile(
    "|".join(re.escape(m) for m in _DEFERRAL_MARKERS), re.IGNORECASE
)

# String literals rather than imports: `cognikernel.quality` is a leaf package
# (see the "Quality is a leaf" contract in pyproject.toml), so it cannot import
# the event-type or authority constants from extraction/storage. gate.py
# already hardcodes type names the same way.
_THREAD_OPEN = "THREAD_OPEN"
_USER_STATED = "user_stated"


def describes_deferred_work(text: str) -> bool:
    """True when the text refers to future, queued or in-progress work."""
    return bool(_DEFERRAL_RE.search(text or ""))


# ── D10: a thread that is a REFERENCE to a thread ────────────────────────────
#
# "This is the active work item for the next session." is not a statement of a
# thread. It is a pointer at one, and it carries nothing without the sentence
# it points at. The extractor splits a turn into sentences, so the pointer and
# its own antecedent arrive as two competing THREAD_OPEN events -- and in the
# real Taskflow store the pointer then superseded the antecedent, 44ms apart,
# leaving the store holding the pronoun and not the referent (#30):
#
#   We need to implement JWT authentication end-to-end -- login endpoint,
#   token issuance, and the FastAPI dependency guard.      <- deleted
#   This is the active work item for the next session.     <- kept, rendered
#
# The signal is the D7 shape: a bare demonstrative subject with no noun head.
# It reuses D7's own regex rather than a second copy, but ONLY the bare-pronoun
# half -- D7 also flags discourse-connective openers ("So ...", "And ..."), and
# on the real corpus that half flags "So the next step is picking back up on
# toolbelt/retry.py", a genuine handoff and the correct answer for a graded
# probe. Demoting it would break a passing probe to fix a failing one.
#
# Validated against all 131 real THREAD_OPEN events: the bare-demonstrative
# rule flags 2, both genuine references, and neither is any project's gold
# thread.
#
# It DEMOTES rather than rejects, for the same reason D9 does: the pointer is
# still a true statement and may be the only thing recorded if extraction
# missed its antecedent. Losing it outright would be worse than mis-ranking it.


def has_bare_demonstrative_subject(text: str) -> bool:
    """True when the text opens with a pronoun subject and no noun head."""
    return bool(_BARE_PRONOUN_SUBJECT.match((text or "").strip()))


def detect_anaphoric_thread(
    text: str, event_type: str, authority: str
) -> DetectorHit | None:
    """D10 -- a top-authority thread that only points at another thread.

    Scoped like D9: only THREAD_OPEN, only `user_stated`. A thread already
    below the top tier cannot outrank or supersede its own antecedent, so
    there is nothing to correct.
    """
    if event_type != _THREAD_OPEN or authority != _USER_STATED:
        return None
    if not has_bare_demonstrative_subject(text):
        return None
    return DetectorHit(
        "D10", "thread is a bare-demonstrative reference to another thread — "
               "it must not outrank or supersede the statement it points at",
    )


# ── explicit handoff to a future session ─────────────────────────────────────
#
# A SECOND, deliberately narrower vocabulary, and the reason it is separate
# from _DEFERRAL_MARKERS above is that the two predicates have OPPOSITE failure
# costs.
#
# describes_deferred_work is a recall-first question — "is this user-stated
# thread a queued item rather than an instruction for right now?" A false
# negative there demotes a real thread, so obligation phrasing ("need to",
# "have to") belongs in it: that is how a user states outstanding work.
#
# This one is a precision-first question — "may narration delete this?" It
# shields a thread from recency supersession, so a false positive resurrects
# exactly the narration pile-up that supersession exists to collapse. On the
# real corpus the broad predicate flags 21 of 131 THREAD_OPEN events, but 13 of
# those are narration it matched on incidental wording ("Continuing with tests
# now." hits "continuing with"); shielding all 21 would have restored most of
# the pile-up. This list flags 8, and all 8 are genuine handoffs.
#
# So the signal here is explicit reference to a LATER SESSION or a parked item,
# never mere obligation or in-progress phrasing. Validated against all 131 real
# THREAD_OPEN events: 8 flagged, 0 false positives.
_HANDOFF_MARKERS = (
    # explicit future session / time reference
    "next session", "next time", "tomorrow", "next week", "future session",
    "starting next",
    # explicit next-step handoff
    "next step", "next up", "picking back up", "pick back up", "pick this up",
    "picks this up", "picks up", "come back to", "coming back to", "resume",
    # parked / queued state
    "queued", "backlog", "on hold", "parked", "left to do", "still to do",
    # explicit handoff to the reader's timing
    "whenever you are", "when you're ready",
)
_HANDOFF_RE = re.compile(
    "|".join(re.escape(m) for m in _HANDOFF_MARKERS), re.IGNORECASE
)


def describes_future_session_handoff(text: str) -> bool:
    """True when the text explicitly hands work to a later session.

    Strictly narrower than describes_deferred_work, and intentionally so —
    see the note above _HANDOFF_MARKERS for why the two must not be merged.
    """
    return bool(_HANDOFF_RE.search(text or ""))


def detect_bare_instruction_thread(
    text: str, event_type: str, authority: str
) -> DetectorHit | None:
    """D9 — a top-authority thread that is really just an instruction.

    Scoped narrowly on purpose: only THREAD_OPEN, only `user_stated`. Any
    other type is unaffected, and a thread already below the top tier has
    nothing to demote — the defect is specifically about an ordinary
    instruction competing at the tier reserved for what the user actually
    said is outstanding.
    """
    if event_type != _THREAD_OPEN or authority != _USER_STATED:
        return None
    if describes_deferred_work(text):
        return None
    return DetectorHit(
        "D9", "user-stated thread with no deferred-work reference — reads as an "
              "instruction for the current turn, not a queued item",
    )
