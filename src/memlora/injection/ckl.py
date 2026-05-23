"""CKL (CogniKernel Language) — compact event rendering.

CKL replaces natural-language prose in the hard-constraints, decisions, and
graveyard sections with a prefix-tagged single-line form.

V1 — mechanical prose truncation (always available):
    CSTR: we will never issue SQL DELETE  # recoverability
    DEC:  pagination uses page+page_size — max 100 server-side
    DEAD: celery — broker dependency  # simplicity goal

V2 — triple-syntax when payload["triple"] is present (set at extraction time):
    CSTR: ¬ SQL DELETE  # recoverability
    DEC:  pagination → page+page_size  # max 100
    DEAD: celery ∅  # broker dep

V2 triples are written into event.payload["triple"] by
memlora.extraction.triple.augment_with_triple() at session-end.
Events extracted before V2 was deployed render as V1 automatically.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlora.storage.events import Event


CKL_LEGEND: str = "# CKL: CSTR=hard rule, DEC=decision, DEAD=rejected, SOFT=advisory"
CKL_OPS_LEGEND: str = "# OPS: ¬=not, →=implies, ←=from, ∅=null, |=or, &=and, ==equals, :=is"


def render_event_ckl(
    event: "Event",
    prefix: str,
    desc_cap: int = 100,
    rationale_cap: int = 35,
    v2: bool = False,
) -> str:
    """Render a single event in CKL format.

    With ``v2=True``: uses triple-syntax if ``payload["triple"]`` is present;
    falls back to V1 prose otherwise.  Old events without a triple always
    render as V1 regardless of the ``v2`` flag.
    """
    if v2:
        triple = event.payload.get("triple")
        if triple:
            return _render_triple(
                triple, prefix, event.payload.get("rationale", ""), rationale_cap
            )

    # V1 prose rendering
    desc = event.payload.get("description", "")[:desc_cap].rstrip()
    rat  = event.payload.get("rationale", "")[:rationale_cap].rstrip()
    line = f"{prefix}: {desc}"
    if rat:
        line += f"  # {rat}"
    return line


def _render_triple(
    triple: dict,
    prefix: str,
    rationale: str,
    rationale_cap: int,
) -> str:
    """Render a structured CKL V2 triple.

    Position rules:
        subject + object  →  PREFIX: subject OP object
        subject only      →  PREFIX: subject OP
        object only       →  PREFIX: OP object
        neither           →  PREFIX: OP          (rare; indicates extraction edge case)
    """
    op      = triple.get("operator", "")
    subject = triple.get("subject", "")
    obj     = triple.get("object", "")

    if subject and obj:
        core = f"{subject} {op} {obj}"
    elif subject:
        core = f"{subject} {op}"
    elif obj:
        core = f"{op} {obj}"
    else:
        core = op

    line = f"{prefix}: {core}"
    rat = rationale[:rationale_cap].rstrip()
    if rat:
        line += f"  # {rat}"
    return line
