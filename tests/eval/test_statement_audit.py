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
report = _load("audit_report")


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


def _rows(n_per_type=25):
    out = []
    for t in ("DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
              "APPROACH_ABANDONED_DO_NOT_RETRY"):
        for i in range(n_per_type):
            out.append({"store": f"s{i % 3}", "event_type": t,
                        "text": f"The {t.lower()} number {i} is recorded plainly here."})
    return out


def test_statement_id_is_stable_and_store_scoped():
    a = audit.statement_id("store1", "same text")
    assert a == audit.statement_id("store1", "same text")
    assert a != audit.statement_id("store2", "same text")
    assert len(a) == 12


def test_stratified_sample_is_deterministic_under_seed():
    rows = _rows()
    first = audit.stratified_sample(rows, n=20, stratum="all", seed=7)
    second = audit.stratified_sample(rows, n=20, stratum="all", seed=7)
    assert [r["text"] for r in first] == [r["text"] for r in second]
    assert audit.stratified_sample(rows, n=20, stratum="all", seed=8) != first


def test_stratified_sample_balances_event_types():
    got = audit.stratified_sample(_rows(), n=20, stratum="all", seed=1)
    counts = collections.Counter(r["event_type"] for r in got)
    assert len(got) == 20
    assert max(counts.values()) - min(counts.values()) <= 1


def test_clean_stratum_excludes_heuristically_flagged():
    rows = _rows() + [{"store": "s9", "event_type": "DECISION",
                       "text": "Now update the router config."}]
    got = audit.stratified_sample(rows, n=30, stratum="clean", seed=3)
    assert all(audit.classify_heuristic(r["text"]) == set() for r in got)


def test_stratified_sample_caps_at_available_rows():
    assert len(audit.stratified_sample(_rows(2), n=500, stratum="all", seed=1)) == 8


def test_statement_id_distinguishes_repeated_text_in_one_store():
    a = audit.statement_id("store1", "same text", 0)
    b = audit.statement_id("store1", "same text", 1)
    assert a != b
    assert audit.statement_id("store1", "same text") == a  # default occurrence=0


def test_stratified_sample_is_order_independent():
    rows = _rows()
    shuffled = list(reversed(rows))
    a = audit.stratified_sample(rows, n=20, stratum="all", seed=7)
    b = audit.stratified_sample(shuffled, n=20, stratum="all", seed=7)
    assert [r["text"] for r in a] == [r["text"] for r in b]


def test_wilson_ci_known_value():
    # 15/60 = 0.25; Wilson 95% CI is approx (0.158, 0.372)
    lo, hi = report.wilson_ci(15, 60)
    assert lo == pytest.approx(0.158, abs=0.002)
    assert hi == pytest.approx(0.372, abs=0.002)


def test_wilson_ci_handles_zero_and_full():
    # At p=0 the centre and half-width are mathematically equal, so the bound is
    # 0 up to float error — assert with a tolerance, not ==.
    assert report.wilson_ci(0, 30)[0] == pytest.approx(0.0, abs=1e-9)
    assert report.wilson_ci(30, 30)[1] == pytest.approx(1.0, abs=1e-9)


def test_wilson_ci_empty_sample_is_full_interval():
    assert report.wilson_ci(0, 0) == (0.0, 1.0)


def test_cohens_kappa_perfect_agreement():
    a = [{"WRONG_TYPE"}, set(), {"COMPOUND"}, set()]
    k = report.cohens_kappa(a, list(a), ("WRONG_TYPE", "COMPOUND"))
    assert k["WRONG_TYPE"] == pytest.approx(1.0)
    assert k["_mean"] == pytest.approx(1.0)


def test_cohens_kappa_chance_agreement_is_zero():
    # A says code on first half, B says code on alternating items -> ~chance
    a = [{"X"}, {"X"}, set(), set()]
    b = [{"X"}, set(), {"X"}, set()]
    assert report.cohens_kappa(a, b, ("X",))["X"] == pytest.approx(0.0, abs=1e-9)


def test_cohens_kappa_degenerate_column_is_none_not_crash():
    # Neither labeler ever used the code — kappa undefined, must not divide by zero
    k = report.cohens_kappa([set(), set()], [set(), set()], ("NEVER",))
    assert k["NEVER"] is None


def _labeled(tmp_path, rows):
    p = tmp_path / "pool.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_load_labeled_rejects_unlabeled_rows(tmp_path):
    p = _labeled(tmp_path, [
        {"id": "a1", "event_type": "DECISION", "text": "x", "labels": ["CLEAN"]},
        {"id": "b2", "event_type": "DECISION", "text": "y", "labels": []},
    ])
    with pytest.raises(ValueError, match="b2"):
        report.load_labeled(p)


def test_defect_rate_counts_any_non_clean_code():
    rows = [{"labels": ["CLEAN"]}, {"labels": ["WRONG_TYPE"]},
            {"labels": ["COMPOUND", "MISSING_SUBJECT"]}, {"labels": ["CLEAN"]}]
    assert report.defect_rate(rows) == (2, 4)


def test_clean_is_exclusive_and_rejected_when_mixed(tmp_path):
    p = _labeled(tmp_path, [{"id": "c3", "event_type": "DECISION", "text": "z",
                             "labels": ["CLEAN", "WRONG_TYPE"]}])
    with pytest.raises(ValueError, match="c3"):
        report.load_labeled(p)


def test_other_requires_a_note(tmp_path):
    p = _labeled(tmp_path, [{"id": "d4", "event_type": "DECISION", "text": "z",
                             "labels": ["OTHER"], "notes": ""}])
    with pytest.raises(ValueError, match="d4"):
        report.load_labeled(p)


def test_heuristic_miss_rate_counts_human_defects_heuristic_missed():
    rows = [{"id": "a", "labels": ["WRONG_TYPE"]}, {"id": "b", "labels": ["CLEAN"]},
            {"id": "c", "labels": ["COMPOUND"]}]
    meta = {"a": {"heuristic": []}, "b": {"heuristic": []},
            "c": {"heuristic": ["NOT_DURABLE"]}}
    got = report.heuristic_miss_rate(rows, meta)
    assert got["human_defective"] == 2
    assert got["missed_by_heuristic"] == 1


def test_load_labeled_rejects_unknown_codes(tmp_path):
    p = _labeled(tmp_path, [
        {"id": "e5", "event_type": "DECISION", "text": "x",
         "labels": ["WRONG_TYPE"]},
        {"id": "f6", "event_type": "DECISION", "text": "y",
         "labels": ["NOT_A_REAL_CODE"]},
    ])
    with pytest.raises(ValueError, match="f6"):
        report.load_labeled(p)


def test_load_labeled_reports_every_problem_at_once(tmp_path):
    p = _labeled(tmp_path, [
        {"id": "g7", "event_type": "DECISION", "text": "x", "labels": []},
        {"id": "h8", "event_type": "DECISION", "text": "y",
         "labels": ["CLEAN", "COMPOUND"]},
        {"id": "i9", "event_type": "DECISION", "text": "z",
         "labels": ["OTHER"], "notes": ""},
    ])
    with pytest.raises(ValueError) as exc:
        report.load_labeled(p)
    message = str(exc.value)
    assert "g7" in message and "h8" in message and "i9" in message


def test_load_labeled_accepts_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert report.load_labeled(p) == []


llm = _load("audit_label_llm")


def test_build_prompt_carries_codebook_and_statement():
    row = {"id": "x1", "event_type": "DECISION", "text": "Some statement text."}
    msgs = llm.build_prompt(row)
    assert msgs[0]["role"] == "system"
    for code in ("DANGLING_REFERENCE", "NOT_A_STATEMENT", "WRONG_TYPE", "NOT_DURABLE",
                 "META_TALK", "COMPOUND", "MISSING_SUBJECT", "OTHER", "CLEAN"):
        assert code in msgs[0]["content"]
    assert "standalone" in msgs[0]["content"].lower()
    assert msgs[-1]["role"] == "user"
    assert "Some statement text." in msgs[-1]["content"]
    assert "DECISION" in msgs[-1]["content"]


def test_parse_labels_accepts_plain_json():
    labels, notes = llm.parse_labels('{"labels": ["CLEAN"], "notes": ""}')
    assert labels == ["CLEAN"]
    assert notes == ""


def test_parse_labels_extracts_json_from_fenced_prose():
    raw = ('Sure, here is my answer:\n```json\n'
           '{"labels": ["WRONG_TYPE"], "notes": ""}\n```\nLet me know if you need more.')
    labels, notes = llm.parse_labels(raw)
    assert labels == ["WRONG_TYPE"]


def test_parse_labels_prefers_last_json_object_when_several():
    raw = ('First I considered {"labels": ["CLEAN"]} but on reflection, '
           '{"labels": ["COMPOUND"], "notes": ""}')
    labels, notes = llm.parse_labels(raw)
    assert labels == ["COMPOUND"]


def test_parse_labels_rejects_clean_mixed_with_other_codes():
    with pytest.raises(ValueError):
        llm.parse_labels('{"labels": ["CLEAN", "COMPOUND"], "notes": ""}')


def test_parse_labels_rejects_other_without_a_note():
    with pytest.raises(ValueError):
        llm.parse_labels('{"labels": ["OTHER"], "notes": ""}')


def test_parse_labels_accepts_other_with_a_note():
    labels, notes = llm.parse_labels('{"labels": ["OTHER"], "notes": "unclear reason"}')
    assert labels == ["OTHER"]
    assert notes == "unclear reason"


def test_parse_labels_rejects_unknown_codes():
    with pytest.raises(ValueError):
        llm.parse_labels('{"labels": ["NOT_A_REAL_CODE"], "notes": ""}')


def test_parse_labels_rejects_unparseable_text():
    with pytest.raises(ValueError):
        llm.parse_labels("I'm not able to help with that request.")


def test_parse_labels_rejects_empty_response():
    with pytest.raises(ValueError):
        llm.parse_labels("")


def test_parse_labels_is_case_insensitive_on_codes():
    labels, notes = llm.parse_labels('{"labels": ["clean"], "notes": ""}')
    assert labels == ["CLEAN"]


def test_parse_labels_allows_multiple_non_clean_codes():
    labels, notes = llm.parse_labels(
        '{"labels": ["COMPOUND", "MISSING_SUBJECT"], "notes": ""}')
    assert set(labels) == {"COMPOUND", "MISSING_SUBJECT"}
