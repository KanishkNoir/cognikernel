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

    def test_no_match_says_so(self, project, capsys) -> None:
        proj, *_ = project

        _why(proj, "kubernetes")

        assert "No claim matches" in capsys.readouterr().out

    def test_missing_store_exits_nonzero(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))

        with pytest.raises(SystemExit) as exc:
            _why(tmp_path / "no-such-project", "anything")

        assert exc.value.code == 1
