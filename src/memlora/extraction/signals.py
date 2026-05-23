"""Signal dictionary for Aho-Corasick trie extraction.

Each entry maps a phrase to (event_type, base_confidence).
The event_type matches the storage schema CHECK constraint exactly.
Entries are lower-cased — matching is always case-insensitive.
"""
from __future__ import annotations

# phrase → (event_type, base_confidence)
SIGNAL_DICTIONARY: dict[str, tuple[str, float]] = {
    # ── Decision signals ─────────────────────────────────────────────────────
    "decided":        ("DECISION", 1.0),
    "chose":          ("DECISION", 1.0),
    "went with":      ("DECISION", 0.9),
    "going with":     ("DECISION", 0.9),
    "settled on":     ("DECISION", 0.9),
    "switched to":    ("DECISION", 0.85),
    "switching from": ("DECISION", 0.85),
    "we'll use":      ("DECISION", 0.8),
    "we will use":    ("DECISION", 0.8),
    "instead of":     ("DECISION", 0.7),
    "rather than":    ("DECISION", 0.7),
    "moved to":       ("DECISION", 0.7),
    "opted for":      ("DECISION", 0.8),
    "replacing":      ("DECISION", 0.75),

    # ── Hard constraint signals ───────────────────────────────────────────────
    "must not":       ("CONSTRAINT_HARD", 1.0),
    "must never":     ("CONSTRAINT_HARD", 1.0),
    "cannot":         ("CONSTRAINT_HARD", 1.0),
    "can't":          ("CONSTRAINT_HARD", 0.9),
    "never":          ("CONSTRAINT_HARD", 0.9),
    "not allowed":    ("CONSTRAINT_HARD", 0.9),
    "forbidden":      ("CONSTRAINT_HARD", 0.9),
    "blocked by":     ("CONSTRAINT_HARD", 0.8),
    "requirement":    ("CONSTRAINT_HARD", 0.9),
    "required":       ("CONSTRAINT_HARD", 0.7),
    "mandatory":      ("CONSTRAINT_HARD", 1.0),
    "lock in":        ("CONSTRAINT_HARD", 0.8),
    "from day one":   ("CONSTRAINT_HARD", 0.75),

    # ── Soft constraint signals ───────────────────────────────────────────────
    "should not":     ("CONSTRAINT_SOFT", 0.7),
    "prefer":         ("CONSTRAINT_SOFT", 0.6),
    "avoid":          ("CONSTRAINT_SOFT", 0.7),
    "try not to":     ("CONSTRAINT_SOFT", 0.6),
    "ideally":        ("CONSTRAINT_SOFT", 0.5),
    "better to":      ("CONSTRAINT_SOFT", 0.6),
    "where possible": ("CONSTRAINT_SOFT", 0.5),

    # ── Abandonment signals ───────────────────────────────────────────────────
    "didn't work":    ("APPROACH_ABANDONED", 0.9),
    "doesn't work":   ("APPROACH_ABANDONED", 0.9),
    "does not work":  ("APPROACH_ABANDONED", 0.9),
    "reverted":       ("APPROACH_ABANDONED", 1.0),
    "rolled back":    ("APPROACH_ABANDONED", 1.0),
    "tried but":      ("APPROACH_ABANDONED", 0.8),
    "abandoned":      ("APPROACH_ABANDONED", 1.0),
    "gave up on":     ("APPROACH_ABANDONED", 0.9),

    # ── Do-not-retry signals ──────────────────────────────────────────────────
    "do not try":          ("APPROACH_ABANDONED_DO_NOT_RETRY", 1.0),
    "don't try":           ("APPROACH_ABANDONED_DO_NOT_RETRY", 0.9),
    "never again":         ("APPROACH_ABANDONED_DO_NOT_RETRY", 0.9),
    "do not use":          ("APPROACH_ABANDONED_DO_NOT_RETRY", 0.85),
    "don't use":           ("APPROACH_ABANDONED_DO_NOT_RETRY", 0.8),
    "failed completely":   ("APPROACH_ABANDONED_DO_NOT_RETRY", 0.8),
    "explicitly abandoned": ("APPROACH_ABANDONED_DO_NOT_RETRY", 1.0),
    "will never revisit":  ("APPROACH_ABANDONED_DO_NOT_RETRY", 1.0),
    "ruled out":           ("APPROACH_ABANDONED_DO_NOT_RETRY", 0.85),
    "won't use":           ("APPROACH_ABANDONED_DO_NOT_RETRY", 0.8),
    "not going with":      ("APPROACH_ABANDONED_DO_NOT_RETRY", 0.75),

    # ── Open thread signals ───────────────────────────────────────────────────
    "todo":                ("THREAD_OPEN", 0.7),
    "to do":               ("THREAD_OPEN", 0.7),
    "next step":           ("THREAD_OPEN", 0.8),
    "still need to":       ("THREAD_OPEN", 0.8),
    "in progress":         ("THREAD_OPEN", 0.7),
    "working on":          ("THREAD_OPEN", 0.6),
    "will implement":      ("THREAD_OPEN", 0.7),
    "need to":             ("THREAD_OPEN", 0.5),
    "work thread":         ("THREAD_OPEN", 0.9),
    "active work item":    ("THREAD_OPEN", 0.9),
    "next session":        ("THREAD_OPEN", 0.7),
    "pick up":             ("THREAD_OPEN", 0.6),
    "will need to":        ("THREAD_OPEN", 0.6),
    "come back to":        ("THREAD_OPEN", 0.6),

    # ── Close thread signals ──────────────────────────────────────────────────
    "finished":       ("THREAD_CLOSE", 0.8),
    "completed":      ("THREAD_CLOSE", 0.8),
    "implemented":    ("THREAD_CLOSE", 0.7),
    "merged":         ("THREAD_CLOSE", 0.8),
    "shipped":        ("THREAD_CLOSE", 0.9),
    "done":           ("THREAD_CLOSE", 0.7),
    "resolved":       ("THREAD_CLOSE", 0.8),
}
