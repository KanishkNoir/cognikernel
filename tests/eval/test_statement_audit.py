"""Phase A statement-quality audit — pytest gate over the audit helpers.

The heuristics here are a deliberately crude PROXY. Their job is to split the
corpus into strata for sampling, not to produce the reported defect rate — that
comes from human labels only (see the spec's Phase A section). These tests pin
the heuristics' stated behaviour so a regex typo cannot silently reshape the
strata.

LIMITATION: passing these tests says the heuristics do what they claim, NOT that
they detect defects well. Their real miss rate is measured against human labels
in scripts/audit_report.py and must be reported alongside any heuristic number.
"""
from __future__ import annotations

import collections
import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit_statement_quality")


@pytest.mark.parametrize("text,expected", [
    ("With it, you see three sibling spans.", "DANGLING_REFERENCE"),
    ("But the corpus also carries a clean negative result.", "DANGLING_REFERENCE"),
    ("Now update the cache lookup in router.py.", "NOT_DURABLE"),
    ("Let me check the router config first.", "NOT_DURABLE"),
    ("The Stop hook will persist this at session close.", "META_TALK"),
    ("Network errors are transient blips. |.", "NOT_A_STATEMENT"),
    ("TIMESTAMPTZ", "NOT_A_STATEMENT"),
])
def test_classify_heuristic_flags_known_defects(text, expected):
    assert expected in audit.classify_heuristic(text)


def test_classify_heuristic_passes_clean_statements():
    clean = "All upstream timeouts surface to the client as 504 Gateway Timeout, never as 500."
    assert audit.classify_heuristic(clean) == set()


def test_classify_heuristic_returns_only_codebook_codes():
    got = audit.classify_heuristic("Now update it. |.")
    assert got <= set(audit.DEFECT_CODES)
