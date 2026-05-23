"""CKL (CogniKernel Language) V1 — compact event rendering.

CKL replaces natural-language prose in the hard-constraints, decisions, and
graveyard sections with a prefix-tagged single-line form:

    CSTR: we will never issue SQL DELETE  # recoverability
    DEC:  pagination uses page+page_size — max 100 server-side
    DEAD: celery — broker dependency  # simplicity goal

V1 is purely mechanical — no symbolic operator substitution, no LLM call at
extraction time. The token saving comes from:
  (1) shorter desc/rationale caps than the prose renderer uses
  (2) a single-token prefix replacing bullet + section-header framing
  (3) a tight 30-char rationale instead of unbounded prose

V2 (symbolic operators like ¬, →, ∅) is deferred — see
research/injection_format/ckl_and_compression.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlora.storage.events import Event


CKL_LEGEND: str = "# CKL: CSTR=hard rule, DEC=decision, DEAD=rejected, SOFT=advisory"


def render_event_ckl(
    event: "Event",
    prefix: str,
    desc_cap: int = 100,
    rationale_cap: int = 35,
) -> str:
    """Render a single event in CKL V1 format.

    The description is the event's primary claim; the rationale is appended
    after a ' # ' separator if non-empty. Both are truncated to their caps —
    no ellipsis, no wrapping, single line.
    """
    desc = event.payload.get("description", "")[:desc_cap].rstrip()
    rat  = event.payload.get("rationale", "")[:rationale_cap].rstrip()
    line = f"{prefix}: {desc}"
    if rat:
        line += f"  # {rat}"
    return line
