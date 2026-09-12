"""S5 T-501: `cognikernel explain-recall <query>` — why memory retrieved what it did,
and why it did not retrieve something.

Two surfaces share one retrieval core: the MCP `recall` tool returns the fused top
k, and the CK-1 per-prompt push runs the same candidates through an echo filter,
the render-ledger filter and a rank-based dual-evidence gate. Explaining either
means exposing every stage's decision, with a reason, computed by the same code
the live path runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from cognikernel.config import Config
from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
from cognikernel.storage.events import Event, insert_event, set_superseded_by
from cognikernel.storage.migrations import run_migrations


def _hit(id_: int, *, dense: int | None, bm25: int | None, cosine: float | None = None,
         description: str = "retry policy attempts jitter") -> dict:
    return {"id": id_, "event_type": "DECISION", "description": description, "subject": "",
            "score": 0.5, "dense_rank": dense, "bm25_rank": bm25, "cosine": cosine}


class TestGateDecisions:
    def test_both_axes_pass_with_a_reason(self) -> None:
        from cognikernel.integration.query import ck1_gate_decisions

        [(hit, passed, reason)] = ck1_gate_decisions(
            [_hit(1, dense=1, bm25=2, cosine=0.8)], "what is the retry policy for attempts", Config())

        assert passed and "both axes" in reason

    def test_a_rank_past_the_limit_names_the_rank(self) -> None:
        from cognikernel.integration.query import ck1_gate_decisions

        decisions = ck1_gate_decisions(
            [_hit(1, dense=1, bm25=7, cosine=0.8), _hit(2, dense=2, bm25=1, cosine=0.7)],
            "what is the retry policy for attempts", Config())

        passed, reason = decisions[0][1], decisions[0][2]
        assert not passed and "bm25 rank 7 > 5" in reason

    def test_bm25_only_mode_explains_the_term_floor(self) -> None:
        from cognikernel.integration.query import ck1_gate_decisions

        [(_, passed, reason)] = ck1_gate_decisions(
            [_hit(1, dense=None, bm25=1, description="retry cap")], "retry please", Config())

        assert not passed and "shared term" in reason

    def test_one_shared_term_is_singular(self) -> None:
        """Review on #46: "only 1 shared terms"."""
        from cognikernel.integration.query import ck1_gate_decisions

        [(_, _, reason)] = ck1_gate_decisions(
            [_hit(1, dense=None, bm25=1, description="retry cap")], "retry please", Config())

        assert "only 1 shared term (" in reason and "1 shared terms" not in reason

    def test_dense_only_mode_explains_the_cosine_floor(self) -> None:
        from cognikernel.integration.query import ck1_gate_decisions

        [(_, passed, reason)] = ck1_gate_decisions([_hit(1, dense=1, bm25=None, cosine=0.41)], "retry", Config())

        assert not passed and "cosine" in reason

    def test_the_gate_keeps_exactly_what_the_decisions_pass(self) -> None:
        from cognikernel.integration.query import _ck1_dual_evidence, ck1_gate_decisions

        hits = [_hit(1, dense=1, bm25=1, cosine=0.9), _hit(2, dense=9, bm25=1, cosine=0.9),
                _hit(3, dense=2, bm25=3, cosine=0.9), _hit(4, dense=3, bm25=2, cosine=0.9)]
        prompt = "what is the retry policy for attempts and jitter"
        config = Config()

        expected = [h for h, ok, _ in ck1_gate_decisions(hits, prompt, config) if ok][: config.ck1_max_events]

        assert _ck1_dual_evidence(hits, prompt, config) == expected


def _mk(pid: str, desc: str) -> Event:
    return Event(project_id=pid, session_id="s1", event_type="DECISION",
                 payload={"description": desc, "subject": ""},
                 content_hash=hashlib.sha256(desc.encode()).hexdigest())


@pytest.fixture
def cold_model(monkeypatch):
    monkeypatch.setattr("cognikernel.embedding.model.is_ready", lambda: False)
    monkeypatch.setattr("cognikernel.embedding.model.warm", lambda: None)


@pytest.fixture
def store(tmp_path: Path, monkeypatch, cold_model):
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    proj.mkdir()
    pid = hash_project_path(str(proj))
    db = get_db_path(Config.load(), pid)
    db.parent.mkdir(parents=True, exist_ok=True)
    ids: dict[str, int] = {}
    with get_connection(db) as c:
        run_migrations(c)
        ids["policy"] = insert_event(c, _mk(pid, "Retry policy: 6 attempts with full jitter"))
        ids["base"] = insert_event(c, _mk(pid, "Retry backoff base is 1.5 seconds"))
        ids["old"] = insert_event(c, _mk(pid, "Retry policy: 4 attempts, no jitter"))
        ids["sse"] = insert_event(c, _mk(pid, "Streaming uses SSE, not WebSockets"))
        set_superseded_by(c, ids["old"], ids["policy"], reason="cross_encoder")
        c.commit()
    return proj, pid, db, ids


def _explain(db: Path, pid: str, query: str, config: Config | None = None, **kw) -> dict:
    from cognikernel.integration.explain_recall import explain_recall

    with get_connection(db) as conn:
        return explain_recall(conn, pid, query, config or Config(), **kw)


class TestExplainRecall:
    def test_reports_axes_and_the_terms_searched(self, store) -> None:
        _, pid, db, _ = store

        data = _explain(db, pid, "what is the retry policy?")

        assert data["axes"]["lexical"]["status"] == "available"
        assert data["axes"]["dense"]["status"] in {"not loaded", "not installed"}
        assert '"retry"' in data["terms"] and '"policy"' in data["terms"]

    def test_marks_which_candidates_recall_returned(self, store) -> None:
        _, pid, db, ids = store

        data = _explain(db, pid, "retry", limit=1)

        returned = [c["id"] for c in data["candidates"] if c["in_recall"]]
        assert len(returned) == 1
        assert {c["id"] for c in data["candidates"]} >= {ids["policy"], ids["base"]}

    def test_ck1_section_explains_every_gate_decision(self, store) -> None:
        _, pid, db, _ = store

        data = _explain(db, pid, "what is the retry policy for attempts and jitter")

        ck1 = data["ck1"]
        assert ck1["candidates"], "CK-1 must consider candidates"
        assert all("reason" in d for d in ck1["decisions"])
        assert set(ck1["injected"]) <= {d["id"] for d in ck1["decisions"] if d["passed"]}

    def test_why_not_a_superseded_claim(self, store) -> None:
        _, pid, db, ids = store

        data = _explain(db, pid, "retry policy", claim_id=ids["old"])

        verdict = data["claim"]["verdict"]
        assert "superseded" in verdict and "only live claims" in verdict

    def test_why_not_a_claim_neither_axis_surfaced(self, store) -> None:
        _, pid, db, ids = store

        data = _explain(db, pid, "retry policy", claim_id=ids["sse"])

        verdict = data["claim"]["verdict"]
        assert "neither axis" in verdict and "--per-axis" in verdict

    def test_why_a_returned_claim_was_returned(self, store) -> None:
        _, pid, db, ids = store

        data = _explain(db, pid, "retry policy attempts jitter", claim_id=ids["policy"])

        assert "returned by recall at rank 1" in data["claim"]["verdict"]

    def test_no_axis_falls_back_to_the_legacy_scan(self, store, monkeypatch) -> None:
        _, pid, db, ids = store
        monkeypatch.setattr("cognikernel.storage.fts.fts_enabled", lambda c: False)

        data = _explain(db, pid, "retry policy attempts")

        assert data["fallback"] == "legacy lexical scan"
        assert data["candidates"] and data["candidates"][0]["id"] == ids["policy"]

    def test_a_fallback_score_is_called_a_jaccard_score_not_a_fused_one(self, store, monkeypatch) -> None:
        """Review on #46: the legacy scan's score is Jaccard overlap, not RRF."""
        _, pid, db, ids = store
        monkeypatch.setattr("cognikernel.storage.fts.fts_enabled", lambda c: False)

        data = _explain(db, pid, "retry policy attempts", claim_id=ids["policy"])

        assert data["score_kind"] == "jaccard"
        assert "Jaccard" in data["claim"]["verdict"] and "fused" not in data["claim"]["verdict"]

    def test_a_fallback_miss_says_only_that_it_fell_below_the_cut(self, store, monkeypatch) -> None:
        """Review on #46: a claim the Jaccard scan matched but ranked below the cut
        was reported as matching no retrieval axis."""
        _, pid, db, ids = store
        monkeypatch.setattr("cognikernel.storage.fts.fts_enabled", lambda c: False)

        data = _explain(db, pid, "retry policy attempts", limit=1, claim_id=ids["base"])

        verdict = data["claim"]["verdict"]
        assert "top 1" in verdict and "no retrieval axis matched" not in verdict

    def test_dense_status_comes_from_the_same_snapshot_as_the_search(self, store, monkeypatch) -> None:
        """Review on #46: the model finishing loading after the search must not make
        the output claim a dense axis the candidates were computed without."""
        _, pid, db, _ = store
        calls = iter([False])
        monkeypatch.setattr("cognikernel.embedding.model.is_ready", lambda: next(calls, True))
        monkeypatch.setattr("cognikernel.embedding.retrieval.recall", lambda *a, **k: [])
        monkeypatch.setattr("cognikernel.integration.explain_recall._fastembed_installed", lambda: True)

        data = _explain(db, pid, "retry policy")

        assert data["axes"]["dense"]["status"] == "not loaded"

    def test_ck1_applies_the_injection_size_limit(self, store) -> None:
        """Review on #46: the live push drops what does not fit query_injection_max_tokens;
        the explanation must not report those claims as pushed."""
        proj, pid, db, _ = store
        # Three terms shared with the policy claim (the BM25-only floor), without
        # restating it closely enough for the echo filter to drop it.
        prompt = "the webhook worker keeps failing deliveries, which retry policy and how many attempts do we use"
        assert _explain(db, pid, prompt)["ck1"]["injected"], "precondition: something passes the gate"
        tiny = Config(query_injection_max_tokens=5)

        ck1 = _explain(db, pid, prompt, config=tiny)["ck1"]

        from cognikernel.integration.query import recall_for_prompt

        assert ck1["injected"] == []
        assert ck1["over_budget"]
        assert recall_for_prompt(str(proj), prompt, config=tiny) == ""

    def test_uses_the_live_retrieval_defaults(self, store, monkeypatch) -> None:
        """Review on #46: the limits are named once, where the live paths use them."""
        _, pid, db, _ = store
        monkeypatch.setattr("cognikernel.integration.query.RECALL_LIMIT", 1)
        monkeypatch.setattr("cognikernel.integration.query.CK1_CANDIDATES", 1)

        data = _explain(db, pid, "retry")

        assert data["fusion"]["limit"] == 1
        assert len(data["ck1"]["candidates"]) <= 1


class TestExplainRecallCommand:
    def _run(self, proj: Path, query: str, **kw) -> None:
        from cognikernel.integration.cli import _cmd_explain_recall

        args = dict(project_path=str(proj), query=query, limit=8, per_axis=20, claim=None,
                    session=None, model_wait=0.0, as_json=False)
        args.update(kw)
        _cmd_explain_recall(argparse.Namespace(**args))

    def test_text_output_names_axes_candidates_and_ck1(self, store, capsys) -> None:
        proj, _, _, ids = store

        self._run(proj, "retry policy attempts jitter")
        out = capsys.readouterr().out

        assert "explain-recall" in out
        assert "lexical (BM25)" in out and "dense" in out
        assert f"#{ids['policy']}" in out
        assert "CK-1" in out

    def test_text_does_not_claim_bm25_ranking_when_the_legacy_scan_ran(self, store, capsys, monkeypatch) -> None:
        """Review on #46: with no axis at all, recall used the Jaccard scan, not BM25."""
        proj, _, _, _ = store
        monkeypatch.setattr("cognikernel.storage.fts.fts_enabled", lambda c: False)

        self._run(proj, "retry policy attempts jitter")
        out = capsys.readouterr().out

        assert "legacy lexical scan" in out
        assert "BM25 alone" not in out

    def test_text_shows_recall_plus_near_misses_and_counts_the_rest(self, store, capsys) -> None:
        """Measured on a real store: one query fused 27 candidates. The text view keeps
        recall's top k and a few near misses; --json keeps every candidate."""
        proj, pid, db, _ = store
        with get_connection(db) as c:
            for n in range(12):
                insert_event(c, _mk(pid, f"Retry note number {n} about the retry schedule"))

        self._run(proj, "retry", limit=2)
        out = capsys.readouterr().out

        rows = [line for line in out.splitlines() if line.strip().split(" ", 1)[0].isdigit()]
        assert len(rows) == 6  # top 2 + 4 near misses
        assert "more candidate(s) below the cut (see --json)" in out

    def test_json_output(self, store, capsys) -> None:
        proj, _, _, ids = store

        self._run(proj, "retry policy", as_json=True, claim=f"#{ids['old']}")
        data = json.loads(capsys.readouterr().out)

        assert data["query"] == "retry policy"
        assert "superseded" in data["claim"]["verdict"]

    def test_missing_store_exits_nonzero(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))

        with pytest.raises(SystemExit) as exc:
            self._run(tmp_path / "nowhere", "anything")

        assert exc.value.code == 1
