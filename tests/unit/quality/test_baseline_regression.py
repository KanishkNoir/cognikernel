"""CI gate: the fixture corpus must stay detected, and the baseline must parse.

The fixture corpus is committed so anyone cloning the repo can reproduce the
detector behaviour without access to a private store. Text is drawn verbatim
from the 163-store sweep, so these are observed defects rather than invented
ones.
"""
import json
from pathlib import Path

import pytest

from cognikernel.quality.detectors import (
    detect_boilerplate,
    detect_junk_constraint,
    detect_subject_less,
)

_FIXTURES = Path(__file__).parents[2] / "fixtures" / "transcripts" / "defects"
_BASELINE = Path(__file__).parents[3] / "docs" / "metrics" / "injection_defect_baseline.json"


_AUDIT = Path(__file__).parents[3] / "scripts" / "injection_defect_audit.py"


@pytest.mark.skipif(
    not _AUDIT.exists(),
    reason="scripts/injection_defect_audit.py is local-only research tooling",
)
class TestWilsonInterval:
    def test_interval_brackets_the_point_estimate(self) -> None:
        from scripts.injection_defect_audit import wilson_interval
        lo, hi = wilson_interval(30, 100)
        assert lo < 0.30 < hi

    def test_zero_hits_has_zero_lower_bound(self) -> None:
        from scripts.injection_defect_audit import wilson_interval
        lo, _ = wilson_interval(0, 100)
        assert lo == pytest.approx(0.0, abs=1e-9)

    def test_empty_denominator_is_zero_zero(self) -> None:
        from scripts.injection_defect_audit import wilson_interval
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_interval_narrows_as_n_grows(self) -> None:
        from scripts.injection_defect_audit import wilson_interval
        lo_small, hi_small = wilson_interval(5, 50)
        lo_big, hi_big = wilson_interval(500, 5000)
        assert (hi_big - lo_big) < (hi_small - lo_small)


class TestFixtureCorpus:
    def test_every_defect_fixture_is_detected(self) -> None:
        cases = {
            "d2_junk_constraint.txt": lambda t: detect_junk_constraint(t, "CONSTRAINT_HARD"),
            "d4_boilerplate.txt": detect_boilerplate,
            "d7_subject_less.txt": detect_subject_less,
        }
        for name, detector in cases.items():
            text = (_FIXTURES / name).read_text(encoding="utf-8").strip()
            assert detector(text) is not None, f"{name} not detected"

    def test_clean_fixture_is_not_flagged(self) -> None:
        text = (_FIXTURES / "clean.txt").read_text(encoding="utf-8").strip()
        assert detect_subject_less(text) is None
        assert detect_boilerplate(text) is None
        assert detect_junk_constraint(text, "CONSTRAINT_HARD") is None


@pytest.mark.skipif(
    not _BASELINE.exists(),
    reason="docs/metrics/ baseline is local-only research output",
)
class TestBaselineGate:
    def test_baseline_file_exists_and_is_valid(self) -> None:
        data = json.loads(_BASELINE.read_text(encoding="utf-8"))
        assert "classes" in data
        assert "generated_at" in data
        assert data["stores_scanned"] >= 0

    def test_every_class_has_a_wilson_interval(self) -> None:
        data = json.loads(_BASELINE.read_text(encoding="utf-8"))
        for rule, c in data["classes"].items():
            lo, hi = c["wilson_95"]
            assert 0.0 <= lo <= hi <= 1.0, f"{rule} has a malformed interval"
            assert lo <= c["rate"] <= hi, f"{rule} rate outside its own interval"
