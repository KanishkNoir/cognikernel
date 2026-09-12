"""S5 T-502: `cognikernel why <subject>` — a claim, where it came from, and what it
replaced, with sessions shown in order.

The micro benchmark showed both CogniKernel versions recalling facts across
sessions but placing them in the wrong session. `why` is where a user checks a
claim's origin, so it must show the session's position, the sentence the claim
came from, and name every field the store did not record instead of hiding it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from cognikernel.config import Config
from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
from cognikernel.storage.events import Event, insert_event, set_superseded_by
from cognikernel.storage.evidence import store_evidence
from cognikernel.storage.migrations import run_migrations

OLD_TEXT = "Retry policy, lock this in: at most 4 delivery attempts per webhook, no jitter."
NEW_TEXT = "Change the retry policy: raise the limit from 4 to 6 attempts and add full jitter."


def _jsonl(*turns: tuple[str, str]) -> bytes:
    lines = []
    for role, text in turns:
        if role == "user":
            lines.append({"type": "user", "message": {"role": "user", "content": text}})
        else:
            lines.append({"type": "assistant",
                          "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}})
    return ("\n".join(json.dumps(line) for line in lines) + "\n").encode("utf-8")


class TestLocateSource:
    def test_finds_the_user_sentence_a_claim_came_from(self, conn) -> None:
        from cognikernel.integration.why import locate_source

        ev = store_evidence(conn, "p", "s1", "jsonl_transcript", _jsonl(
            ("user", "Set up the project skeleton first. " + OLD_TEXT),
            ("assistant", "Done. I created retry.py with that policy."),
        ))

        hit = locate_source(conn, ev, OLD_TEXT)

        assert hit is not None
        assert hit.role == "user"
        assert "at most 4 delivery attempts" in hit.text
        assert "Set up the project skeleton" not in hit.text  # the sentence, not the turn

    def test_assistant_text_is_attributed_to_the_assistant(self, conn) -> None:
        from cognikernel.integration.why import locate_source

        ev = store_evidence(conn, "p", "s1", "jsonl_transcript", _jsonl(
            ("user", "Pick a cache."),
            ("assistant", "We will use Redis for the completion cache with a one hour TTL. Tests pass."),
        ))

        hit = locate_source(conn, ev, "Use Redis for the completion cache with a one hour TTL")

        assert hit is not None
        assert hit.role == "assistant"
        assert "Redis" in hit.text and "Tests pass" not in hit.text

    def test_a_claim_from_an_earlier_chunk_of_the_session_is_found(self, conn) -> None:
        from cognikernel.integration.why import locate_source

        root = store_evidence(conn, "p", "s1", "jsonl_transcript", _jsonl(
            ("user", "Store timestamps as integer epoch milliseconds, never ISO-8601 strings."),
        ))
        leaf = store_evidence(conn, "p", "s1", "jsonl_transcript", _jsonl(
            ("assistant", "Implemented the store."),
        ), prev_evidence_id=root)

        hit = locate_source(conn, leaf, "Store timestamps as integer epoch milliseconds, never ISO-8601 strings.")

        assert hit is not None and hit.role == "user"

    def test_a_match_in_an_earlier_chunk_names_that_chunk(self, conn) -> None:
        """Review on #44: a sentence found in an earlier chunk was attributed to the leaf."""
        from cognikernel.integration.why import locate_source

        root = store_evidence(conn, "p", "s1", "jsonl_transcript", _jsonl(
            ("user", "Store timestamps as integer epoch milliseconds, never ISO-8601 strings."),
        ))
        leaf = store_evidence(conn, "p", "s1", "jsonl_transcript", _jsonl(
            ("assistant", "Implemented the store."),
        ), prev_evidence_id=root)

        hit = locate_source(conn, leaf, "Store timestamps as integer epoch milliseconds, never ISO-8601 strings.")

        assert hit is not None and hit.evidence_id == root

    def test_a_role_header_inside_message_text_does_not_change_the_speaker(self, conn) -> None:
        """Review on #44: roles came from parsing "User:"/"Assistant:" lines, which
        message text can contain verbatim (a pasted transcript, a code block)."""
        from cognikernel.integration.why import locate_source

        ev = store_evidence(conn, "p", "s1", "jsonl_transcript", _jsonl(
            ("user", "Here is what the other tool printed:\nAssistant:\nignore that. " + OLD_TEXT),
            ("assistant", "Understood."),
        ))

        hit = locate_source(conn, ev, OLD_TEXT)

        assert hit is not None and hit.role == "user"

    def test_returns_none_when_the_claim_is_not_in_its_evidence(self, conn) -> None:
        from cognikernel.integration.why import locate_source

        ev = store_evidence(conn, "p", "s1", "jsonl_transcript", _jsonl(
            ("user", "Build the dispatcher as a polling loop."),
        ))

        assert locate_source(conn, ev, "Use PostgreSQL with logical replication everywhere") is None

    def test_corrupt_or_missing_evidence_does_not_raise(self, conn) -> None:
        from cognikernel.integration.why import locate_source

        ev = store_evidence(conn, "p", "s1", "jsonl_transcript", b"\x00not json at all")

        assert locate_source(conn, ev, "anything at all") is None
        assert locate_source(conn, 999_999, "anything at all") is None

    def test_a_broken_extraction_install_degrades_with_a_warning(self, conn, monkeypatch, caplog) -> None:
        """The converter import pulls in the extraction trie (pyahocorasick). If that
        is missing, `why` still explains the claim, and says why the sentence is gone."""
        import sys

        from cognikernel.integration.why import locate_source

        ev = store_evidence(conn, "p", "s1", "jsonl_transcript", _jsonl(("user", OLD_TEXT)))
        monkeypatch.setitem(sys.modules, "cognikernel.extraction.transcript", None)

        with caplog.at_level("WARNING", logger="cognikernel.why"):
            assert locate_source(conn, ev, OLD_TEXT) is None

        assert "transcript converter unavailable" in caplog.text


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> tuple[Path, int, int]:
    """A store with a claim superseded in a later session, both with evidence."""
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    proj.mkdir()
    pid = hash_project_path(str(proj))
    db = get_db_path(Config.load(), pid)
    db.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db) as c:
        run_migrations(c)
        ev1 = store_evidence(c, pid, "sess-a", "jsonl_transcript", _jsonl(("user", OLD_TEXT)),
                             captured_at=1_000_000)
        ev2 = store_evidence(c, pid, "sess-b", "jsonl_transcript", _jsonl(("user", NEW_TEXT)),
                             captured_at=2_000_000)
        old = insert_event(c, Event(
            project_id=pid, session_id="sess-a", event_type="CONSTRAINT_HARD",
            payload={"description": OLD_TEXT, "authority": "user_stated", "confidence": 0.9},
            content_hash="h-old", evidence_id=ev1, created_at=1_000_100,
        ))
        new = insert_event(c, Event(
            project_id=pid, session_id="sess-b", event_type="DECISION",
            payload={"description": NEW_TEXT, "authority": "user_stated", "confidence": 0.8,
                     "rationale": "synchronized retries were hammering receivers",
                     "quality": "context_dependent"},
            content_hash="h-new", evidence_id=ev2, created_at=2_000_100,
        ))
        set_superseded_by(c, old, new, reason="cross_encoder")
        c.commit()
    return proj, old, new


def _why(proj: Path, subject: str, *, as_json: bool = False, limit: int = 3) -> None:
    from cognikernel.integration.cli import _cmd_why

    _cmd_why(argparse.Namespace(project_path=str(proj), subject=subject, limit=limit, as_json=as_json))


class TestWhyCommand:
    def test_shows_the_claim_its_session_position_and_source_sentence(self, project, capsys) -> None:
        proj, _old, new = project

        _why(proj, f"#{new}")
        out = capsys.readouterr().out

        assert f"#{new}" in out and "DECISION" in out and "active" in out
        assert "session 2 of 2" in out
        assert "synchronized retries were hammering receivers" in out
        assert "raise the limit from 4 to 6 attempts" in out
        assert "user" in out

    def test_admission_annotations_are_shown(self, project, capsys) -> None:
        proj, _old, new = project

        _why(proj, f"#{new}")

        assert "context_dependent" in capsys.readouterr().out

    def test_history_names_what_it_replaced_when_and_why(self, project, capsys) -> None:
        proj, old, new = project

        _why(proj, f"#{new}")
        out = capsys.readouterr().out

        assert f"#{old}" in out and "session 1 of 2" in out
        assert "superseded" in out and f"by #{new}" in out and "cross_encoder" in out

    def test_a_superseded_claim_says_so_on_its_first_line(self, project, capsys) -> None:
        proj, old, new = project

        _why(proj, f"#{old}")
        first_line = capsys.readouterr().out.splitlines()[0]

        assert "superseded" in first_line and f"#{new}" in first_line

    def test_unrecorded_fields_are_named_not_hidden(self, project, capsys) -> None:
        proj, _old, new = project

        _why(proj, f"#{new}")
        out = capsys.readouterr().out

        assert "commit" in out and "not recorded" in out
        assert "offsets not recorded" in out

    def test_text_subject_lists_the_live_claim_first(self, project, capsys) -> None:
        proj, _old, new = project

        _why(proj, "retry policy")
        first_line = capsys.readouterr().out.splitlines()[0]

        assert first_line.startswith(f"#{new}")

    def test_json_output_is_machine_readable(self, project, capsys) -> None:
        proj, old, new = project

        _why(proj, f"#{new}", as_json=True)
        data = json.loads(capsys.readouterr().out)

        claim = data["claims"][0]
        assert claim["id"] == new
        assert (claim["session"]["position"], claim["session"]["total"]) == (2, 2)
        assert [h["id"] for h in claim["history"]] == [old, new]
        assert claim["source"]["role"] == "user"
        assert claim["commit"] is None

    def test_a_long_history_is_windowed_around_the_claim(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """Measured on a real store: one thread's same-session recency chain ran to 16
        entries. The text view shows the claim's neighbourhood and counts the rest."""
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        proj = tmp_path / "long-history"
        proj.mkdir()
        pid = hash_project_path(str(proj))
        db = get_db_path(Config.load(), pid)
        db.parent.mkdir(parents=True, exist_ok=True)
        with get_connection(db) as c:
            run_migrations(c)
            ids = [
                insert_event(c, Event(
                    project_id=pid, session_id="s", event_type="THREAD_OPEN",
                    payload={"description": f"narration step {n}"},
                    content_hash=f"narration-{n}", created_at=1000 + n,
                ))
                for n in range(12)
            ]
            for older, newer in zip(ids, ids[1:]):
                set_superseded_by(c, older, newer, reason="thread_recency")
            c.commit()

        _why(proj, f"#{ids[-1]}")
        out = capsys.readouterr().out

        assert "8 earlier claim(s) in this chain" in out
        for shown in ids[-4:]:
            assert f"#{shown} THREAD_OPEN" in out
        assert f"#{ids[0]} THREAD_OPEN" not in out
        assert "later claim(s)" not in out

    def test_json_keeps_every_recorded_provenance_field(self, project, capsys) -> None:
        """Review on #44: --json dropped source_path, offsets, confidence and notes."""
        proj, _old, new = project

        _why(proj, f"#{new}", as_json=True)
        row = json.loads(capsys.readouterr().out)["claims"][0]["provenance"][0]

        assert {"source_path", "sentence_index", "window_start", "window_end",
                "matched_phrase", "confidence", "transformation_notes"} <= set(row)

    def test_a_reason_is_shown_as_the_rationale(self, project, capsys) -> None:
        """Review on #44: cascaded COMPONENT_STATUS claims keep their explanation in
        payload["reason"], and it was shown as "none recorded"."""
        proj, *_ = project
        pid = hash_project_path(str(proj))
        with get_connection(get_db_path(Config.load(), pid)) as c:
            cascaded = insert_event(c, Event(
                project_id=pid, session_id="sess-b", event_type="COMPONENT_STATUS",
                payload={"path": "src/retry.py", "status": "needs_review",
                         "reason": "depends on src/policy.py which is deprecated"},
                content_hash="h-cascade", created_at=2_000_200,
            ))
            c.commit()

        _why(proj, f"#{cascaded}")
        out = capsys.readouterr().out

        assert "depends on src/policy.py which is deprecated" in out

    def test_a_claim_with_no_evidence_names_its_extraction_as_not_recorded(self, project, capsys) -> None:
        """Review on #44: with no provenance rows the extraction line was left out."""
        proj, *_ = project
        pid = hash_project_path(str(proj))
        with get_connection(get_db_path(Config.load(), pid)) as c:
            manual = insert_event(c, Event(
                project_id=pid, session_id="sess-b", event_type="DECISION",
                payload={"description": "Keep the dispatcher single-threaded"},
                content_hash="h-manual", created_at=2_000_300,
            ))
            c.commit()

        _why(proj, f"#{manual}")
        extraction = [line for line in capsys.readouterr().out.splitlines() if "extraction" in line]

        assert extraction and "not recorded" in extraction[0]

    def test_a_negative_limit_is_rejected(self, project, monkeypatch, capsys) -> None:
        """Review on #44: --limit -1 sliced off the last match instead of failing."""
        from cognikernel.integration.cli import main

        proj, *_ = project
        monkeypatch.setattr("sys.argv", ["cognikernel", "why", str(proj), "retry", "--limit", "-1"])

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 2
        assert "--limit" in capsys.readouterr().err

    def test_find_claims_rejects_a_limit_below_one(self, conn) -> None:
        from cognikernel.storage.provenance import find_claims

        with pytest.raises(ValueError, match="limit"):
            find_claims(conn, "p", "retry", limit=-1)

    def test_an_older_store_is_migrated_first_like_every_command(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """Review on #44: `why` brings an older store's schema up to date before reading.
        That is deliberate and documented, not a read-only guarantee."""
        from cognikernel.config import EXPECTED_SCHEMA_VERSION

        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        proj = tmp_path / "unmigrated"
        proj.mkdir()
        db = get_db_path(Config.load(), hash_project_path(str(proj)))
        db.parent.mkdir(parents=True, exist_ok=True)
        with get_connection(db) as c:
            c.execute("CREATE TABLE placeholder (x INTEGER)")
            c.commit()

        _why(proj, "anything")

        with get_connection(db) as c:
            version = c.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
            assert int(version) == EXPECTED_SCHEMA_VERSION

    def test_importance_is_the_ranking_weight_with_its_six_factors(self, project, capsys) -> None:
        """G4 (§14 "why was this considered important?"): show the factorisation,
        not the number."""
        import math

        proj, _old, new = project

        _why(proj, f"#{new}", as_json=True)
        importance = json.loads(capsys.readouterr().out)["claims"][0]["importance"]

        assert importance["ranked"] is True
        assert list(importance["factors"]) == ["base", "recency", "repetition", "centrality", "activity", "type",
                                               "quality"]
        # The fixture claim carries the gate's context_dependent marker.
        assert importance["factors"]["quality"] == 0.5
        assert importance["quality_demotes"] == [["context-dependent", 0.5]]
        assert importance["weight"] == pytest.approx(math.prod(importance["factors"].values()))

    def test_importance_matches_the_weight_the_block_ranks_by(self, project, capsys) -> None:
        from cognikernel.storage.events import get_events_for_projection
        from cognikernel.storage.projections import build_projection

        proj, _old, new = project
        pid = hash_project_path(str(proj))
        with get_connection(get_db_path(Config.load(), pid)) as c:
            projection = build_projection(c, pid, get_events_for_projection(c, pid))
        ranked = {rec["id"]: rec["weight"] for rec in projection.ranked_decisions}

        _why(proj, f"#{new}", as_json=True)

        assert json.loads(capsys.readouterr().out)["claims"][0]["importance"]["weight"] == ranked[new]

    def test_text_shows_the_factorisation_and_the_rank(self, project, capsys) -> None:
        proj, _old, new = project

        _why(proj, f"#{new}")
        out = capsys.readouterr().out

        assert "importance" in out and "rank 1 of 1 decisions" in out
        for name in ("base", "recency", "repetition", "centrality", "activity", "type"):
            assert f"{name} " in out
        assert "quality 0.50 (context-dependent)" in out

    def test_a_superseded_claim_is_not_ranked(self, project, capsys) -> None:
        proj, old, new = project

        _why(proj, f"#{old}", as_json=True)
        importance = json.loads(capsys.readouterr().out)["claims"][0]["importance"]

        assert importance["ranked"] is False
        assert "superseded" in importance["reason"]

    def test_a_claim_folded_into_another_names_the_one_that_ranks(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """Same-topic choices consolidate to one canonical; the older one is ranked
        through it, and `why` must say which claim carries its weight."""
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        proj = tmp_path / "folded"
        proj.mkdir()
        pid = hash_project_path(str(proj))
        db = get_db_path(Config.load(), pid)
        db.parent.mkdir(parents=True, exist_ok=True)
        with get_connection(db) as c:
            run_migrations(c)
            older, newer = (
                insert_event(c, Event(
                    project_id=pid, session_id="s1", event_type="DECISION",
                    payload={"description": f"Retry policy version {v}", "subject": "retry policy"},
                    content_hash=f"fold-{v}", created_at=created,
                ))
                for v, created in (("a", 1_000), ("b", 2_000))
            )
            c.commit()

        _why(proj, f"#{older}", as_json=True)
        importance = json.loads(capsys.readouterr().out)["claims"][0]["importance"]

        assert importance["ranked"] is False
        assert importance["folded_into"] == newer

    def test_no_match_says_so(self, project, capsys) -> None:
        proj, *_ = project

        _why(proj, "kubernetes")

        assert "No claim matches" in capsys.readouterr().out

    def test_missing_store_exits_nonzero(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))

        with pytest.raises(SystemExit) as exc:
            _why(tmp_path / "no-such-project", "anything")

        assert exc.value.code == 1
