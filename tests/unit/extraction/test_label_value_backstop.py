"""The label-value backstop rescues settings, not reply framing.

The salience head calls "Max attempts: 2 (1 initial + 1 retry)" noise, so a
deterministic backstop turns a "Label: value" line back into a DECISION. The same
shape also covers "ANSWER: dead | 14 days …" and "Verified: 13/13 tests pass" —
the head was right about those. In the 2026-09-12 micro benchmark every graded
answer line was stored this way (confidence 0.45) and the top two Key decisions
in the last session were answer lines.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cognikernel.extraction.pipeline import SessionMetadata, _extract_via_head, is_label_value_fact


class TestIsLabelValueFact:
    @pytest.mark.parametrize("line", [
        "Max attempts: 2 (1 initial + 1 retry).",
        "Recovery window: 30 s before the circuit half-opens.",
        "Retry policy: max_attempts_per_target: 2, max_total_attempts: 3.",
        # Discourse labels that carried real facts in the measured stores stay rescued.
        "So: response caching defaults off and is enabled per virtual key (2 settings).",
        "Right now: it retries forever; the cap freezes at 3600 s between attempts.",
    ])
    def test_a_setting_is_rescued(self, line: str) -> None:
        assert is_label_value_fact(line)

    @pytest.mark.parametrize("line", [
        "ANSWER: dead | 14 days (via purge_dead, not automatic).",
        "Answer: integer epoch milliseconds, and payloads up to 262144 bytes.",
        "Summary: the attempt budget was already 3, matching the recorded decision.",
        "Verified: 13/13 gateway tests pass and mypy --strict is clean.",
        "Full suite: 327 passed (324 baseline + 3 new).",
        "Report updated: task-2 report now includes both fixes and 12 test results.",
    ])
    def test_reply_framing_and_status_reports_are_not_rescued(self, line: str) -> None:
        assert not is_label_value_fact(line)


class _AllNoiseHead:
    """A head that calls every sentence noise, so only the backstop can keep one."""

    def classify_scored(self, text: str):
        return "NOISE", 0.9


def test_extraction_keeps_the_setting_and_drops_the_answer_line() -> None:
    sentences = [
        SimpleNamespace(text=text, role="assistant", is_code_block=False)
        for text in (
            "Max attempts: 2 (1 initial + 1 retry).",
            "ANSWER: dead | 14 days (via purge_dead, not automatic).",
        )
    ]

    events = _extract_via_head(sentences, SessionMetadata("p", "s1", 0, 0), head=_AllNoiseHead())

    descriptions = [e.payload["description"] for e in events]
    assert any("Max attempts" in d for d in descriptions)
    assert not any("ANSWER" in d.upper() for d in descriptions)
