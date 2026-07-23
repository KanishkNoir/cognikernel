"""Composed input for the salience head (P2) — the ONE place train/eval/inference
agree on how role + previous-sentence context wrap the target sentence.

The sentence-only head hit two ceilings the label-ceiling study isolated:
meta-framing (memory_meta 0.56 — "the recall surfaces the earlier decision to use
X" reads as a DECISION) and THREAD's remainder. Both hinge on CONTEXT the isolated
sentence doesn't carry: who is speaking (a user directive vs an assistant claim)
and what was just said (a reference to a prior decision vs a new one).

CRITICAL: a head must be trained on the SAME composition it is scored with. This
function is that single source of truth — the CoT spike proved a train/inference
prompt mismatch silently destroys the model. `compose(text)` with no role/prev is
the identity-ish bare form, so a context-blind (v2) head is unaffected.
"""
from __future__ import annotations

_ROLES = ("user", "assistant")


def compose_head_input(text: str, role: str = "", prev: str = "") -> str:
    """Return the head input string for a sentence given its role + prev sentence.

    Formats (chosen so bge tokenizes the markers cleanly):
      role + prev : "[user] <prev> || <text>"
      role only   : "[user] <text>"
      neither     : "<text>"   (bare — what the v2 context-blind head expects)

    Empty/whitespace role or prev are dropped, so passing "" for either degrades
    gracefully to the smaller form.
    """
    text = (text or "").strip()
    r = (role or "").strip().lower()
    p = (prev or "").strip()
    prefix = f"[{r}] " if r in _ROLES else ""
    ctx = f"{p} || " if p else ""
    return f"{prefix}{ctx}{text}"


def role_for_register(register: str) -> str:
    """Best-effort role for a synthetic/eval item that has a register but no
    recorded speaker — user-voiced registers vs assistant-voiced ones. Used only
    when the real role is unavailable (real store items carry the true role)."""
    reg = (register or "").lower()
    user_regs = {"question", "instruction", "instruction_factish",
                 "casual_chat", "casual_update", "casual", "open_work_casual"}
    return "user" if reg in user_regs else "assistant"
