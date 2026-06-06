"""Regression gate for extraction quality, anchored to the frozen Relay S1 baseline.

Two roles:
  1. Characterization lock — the harness + gold fixture must keep reporting the
     known baseline (62% precision, 33% clean recall, the 7 named hard-misses,
     12 truncated events). If these drift, the YARDSTICK changed and any later
     "improvement" is suspect. This guards the measuring stick, not the extractor.
  2. v1 target gate — `test_v1_targets_met` xfails until the v1 extractor (segmenter
     fix + salience head) lands and the frozen events are regenerated from it.
     Flip `@pytest.mark.xfail` off when v1 ships.

The harness lives in scripts/eval_extraction.py; this test imports it directly so
the scoring logic has exactly one source of truth.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_GOLD = _ROOT / "tests" / "fixtures" / "relay_s1_gold.json"
_BASELINE_EVENTS = _ROOT / "tests" / "fixtures" / "relay_s1_baseline_events.json"


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "eval_extraction", _ROOT / "scripts" / "eval_extraction.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


@pytest.fixture(scope="module")
def gold():
    return json.loads(_GOLD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def baseline_score(harness, gold):
    events = harness.load_events_from_json(str(_BASELINE_EVENTS))
    return harness.score(events, gold)


# ── 1. Characterization lock — the yardstick must not drift ──────────────────

def test_baseline_event_count(baseline_score):
    assert baseline_score["events"] == 60


def test_baseline_hard_miss_set(baseline_score):
    # The exact 7 facts the failure analysis flagged as un-retrievable.
    assert set(baseline_score["miss_ids"]) == {
        "GT7", "GT8", "GT9", "GT11", "GT13", "GT19", "GT20"
    }


def test_baseline_truncation_count(baseline_score):
    # The smart_truncate 120-char bug severs 12 facts mid-value.
    assert baseline_score["fp_breakdown"].get("truncated", 0) == 12


def test_baseline_regime(baseline_score):
    # Loose bands — locks the regime, not exact floats, so harmless refactors
    # of the scorer don't break the test but a real drift does.
    assert 0.55 <= baseline_score["precision"] <= 0.68
    assert 0.28 <= baseline_score["clean_recall"] <= 0.38
    assert baseline_score["noise_rate"] >= 0.30


def test_baseline_fails_every_v1_target(harness, gold, baseline_score):
    # Sanity: the unimproved baseline must NOT pass the v1 bar.
    # report() returns True only if all targets met.
    assert harness.report(baseline_score, gold) is False


# ── 2. v1 target gate — flip xfail off when the v1 extractor lands ───────────

@pytest.mark.xfail(reason="v1 extractor (segmenter fix + salience head) not yet implemented",
                   strict=True)
def test_v1_targets_met(harness, gold, baseline_score):
    # When v1 ships: regenerate relay_s1_baseline_events.json from the v1 extractor
    # (re-extract dcea0e3e into a scratch DB, dump events) and remove the xfail.
    assert harness.report(baseline_score, gold) is True
