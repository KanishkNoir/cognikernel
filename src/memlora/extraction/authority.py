"""Authority gating for events — Phase A-4.

Each event carries an `authority` payload field describing where the
information came from:

  user_stated                       The user explicitly said it (high trust).
  assistant_decided                 The assistant declared it (medium trust).
  assistant_answer_to_user_question The assistant answered a user trie-match
                                    sentence (low trust; lands in Pending
                                    Confirmation until user-reaffirmed).
  inferred_from_code                Derived from file mentions / git diff
                                    (informational; never gates constraints).
  llm                               Output of A-5 LLM enrichment pass.

The renderer routes events to sections based on (event_type, authority).
Suppression of co-capture events (the assistant_answer_* class) is done by
normalized-subject match — `normalize_subject` produces a lowercase, punctuation-
stripped, article-free form so paraphrased mentions collapse together.
"""
from __future__ import annotations

import re

# ── authority enum (string constants — no Enum overhead) ─────────────────────

USER_STATED = "user_stated"
ASSISTANT_DECIDED = "assistant_decided"
ASSISTANT_ANSWER_TO_QUESTION = "assistant_answer_to_user_question"
INFERRED_FROM_CODE = "inferred_from_code"
LLM = "llm"

ALL_AUTHORITIES = frozenset({
    USER_STATED,
    ASSISTANT_DECIDED,
    ASSISTANT_ANSWER_TO_QUESTION,
    INFERRED_FROM_CODE,
    LLM,
})

# Authorities whose subjects suppress co-capture (they "confirm" the same fact).
CONFIRMING_AUTHORITIES = frozenset({USER_STATED, ASSISTANT_DECIDED, LLM})


# ── subject normalization for suppression matching ──────────────────────────


_PUNCT = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_ARTICLES_LEADING = re.compile(r"^(?:the|a|an|our|this|that|these|those)\s+", re.IGNORECASE)


def normalize_subject(text: str) -> str:
    """Return a normalized subject for fuzzy suppression match.

    Pipeline:
      lowercase → strip punctuation → collapse whitespace → drop leading article.

    Two subjects compare equal when their normalized forms compare equal:
      "JWT secret"        ≡ "the JWT secret" ≡ "JWT Secret."

    Returns '' for falsy input. The empty string never participates in
    suppression — callers should filter ''.
    """
    if not text:
        return ""
    s = text.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    while True:
        new = _ARTICLES_LEADING.sub("", s)
        if new == s:
            break
        s = new
    return s


# ── default authority assignment ─────────────────────────────────────────────


def default_authority_for_role(role: str) -> str:
    """The natural authority for a trie/pattern hit on a single sentence.

    `role` is the speaker role of the sentence the signal fired on.
    """
    if role == "user":
        return USER_STATED
    if role == "assistant":
        return ASSISTANT_DECIDED
    return ASSISTANT_DECIDED  # safe default for unknown roles
