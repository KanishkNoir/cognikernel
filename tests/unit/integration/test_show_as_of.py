"""S5 T-503 (G3): `cognikernel show <project> --as-of <when>`.

Replays what the store believed at a past time and states its own horizon: how
many claims have no recorded end time, and from when end times exist at all.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cognikernel.config import Config
from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
from cognikernel.storage.events import Event, insert_event
from cognikernel.storage.migrations import run_migrations


def _ms(*parts: int) -> int:
    return int(datetime.datetime(*parts).timestamp() * 1000)


T1 = _ms(2026, 9, 1, 10, 0)    # "old limit" and "pre-021 claim" created
T2 = _ms(2026, 9, 5, 10, 0)    # "new limit" created, "old limit" superseded
T3 = _ms(2026, 9, 9, 10, 0)    # "later claim" created


class TestParseWhen:
    def test_a_date_means_the_end_of_that_day(self, tmp_path: Path) -> None:
        from cognikernel.integration.as_of import parse_when

        at, label = parse_when("2026-09-11", tmp_path)

        assert at == int(datetime.datetime(2026, 9, 11, 23, 59, 59, 999000).timestamp() * 1000)
        assert "2026-09-11" in label and "end of day" in label

    def test_a_date_and_time_is_exact(self, tmp_path: Path) -> None:
        from cognikernel.integration.as_of import parse_when

        at, _ = parse_when("2026-09-11 14:30", tmp_path)

        assert at == _ms(2026, 9, 11, 14, 30)

    def test_something_unparseable_says_what_is_accepted(self, tmp_path: Path) -> None:
        from cognikernel.integration.as_of import parse_when

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            parse_when("last tuesday", tmp_path)

    @pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
    def test_a_commit_sha_means_its_commit_time(self, tmp_path: Path) -> None:
        from cognikernel.integration.as_of import parse_when

        repo = tmp_path / "repo"
        repo.mkdir()
        env_date = "2026-01-02T03:04:05+00:00"
        run = lambda *a: subprocess.run(  # noqa: E731
            ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a],
            check=True, capture_output=True, text=True,
            env={**__import__("os").environ, "GIT_AUTHOR_DATE": env_date, "GIT_COMMITTER_DATE": env_date},
        )
        run("init", "-q")
        run("commit", "-q", "--allow-empty", "-m", "anchor")
        sha = run("rev-parse", "HEAD").stdout.strip()

        at, label = parse_when(sha[:10], repo)

        assert at == int(datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc).timestamp() * 1000)
        assert sha[:7] in label


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> tuple[Path, dict[str, int]]:
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    proj.mkdir()
    pid = hash_project_path(str(proj))
    db = get_db_path(Config.load(), pid)
    db.parent.mkdir(parents=True, exist_ok=True)
    ids: dict[str, int] = {}
    with get_connection(db) as c:
        run_migrations(c)

        def claim(key: str, text: str, created: int, event_type: str = "DECISION", session: str = "s1") -> None:
            ids[key] = insert_event(c, Event(
                project_id=pid, session_id=session, event_type=event_type,
                payload={"description": text}, content_hash=key, created_at=created,
            ))

        claim("old", "Dispatcher limit is 2 in-flight deliveries per host", T1, "CONSTRAINT_HARD")
        claim("pre021", "Use a polling loop, not a broker", T1)
        claim("new", "Dispatcher limit is 4 in-flight deliveries per host", T2, "CONSTRAINT_HARD", session="s2")
        claim("later", "Add a stats helper", T3, session="s3")
        c.execute("UPDATE events SET superseded_by = ?, superseded_at = ?, supersede_reason = 'decision_key' "
                  "WHERE id = ?", (ids["new"], T2, ids["old"]))
        c.execute("UPDATE events SET superseded_by = ?, superseded_at = NULL WHERE id = ?",
                  (ids["later"], ids["pre021"]))
        c.commit()
    return proj, ids


def _show(proj: Path, as_of: str | None, *, as_json: bool = False) -> None:
    from cognikernel.integration.cli import _cmd_show

    _cmd_show(argparse.Namespace(project_path=str(proj), as_json=as_json, as_of=as_of))


class TestShowAsOf:
    def test_lists_what_was_live_then_and_nothing_created_later(self, project, capsys) -> None:
        proj, ids = project

        _show(proj, "2026-09-02")
        out = capsys.readouterr().out

        assert "As of 2026-09-02" in out
        assert f"#{ids['old']}" in out and "2 in-flight" in out
        assert f"#{ids['new']} " not in out and "4 in-flight" not in out
        assert "stats helper" not in out

    def test_a_superseded_claim_drops_out_once_it_was_replaced(self, project, capsys) -> None:
        proj, ids = project

        _show(proj, "2026-09-06")
        out = capsys.readouterr().out

        assert "4 in-flight" in out
        assert "2 in-flight" not in out
        assert "1 ended by then" in out

    def test_states_its_horizon(self, project, capsys) -> None:
        proj, ids = project

        _show(proj, "2026-09-06")
        out = capsys.readouterr().out

        assert "1 with no recorded end time" in out
        assert "end times are recorded from 2026-09-05" in out

    def test_claims_with_unknown_timing_are_listed_separately(self, project, capsys) -> None:
        proj, ids = project

        _show(proj, "2026-09-06")
        out = capsys.readouterr().out

        unknown = out.split("Timing unknown", 1)
        assert len(unknown) == 2
        assert f"#{ids['pre021']}" in unknown[1] and "polling loop" in unknown[1]
        assert "polling loop" not in unknown[0]

    def test_before_any_claim_says_nothing_existed(self, project, capsys) -> None:
        proj, _ = project

        _show(proj, "2026-08-01")

        assert "No claims existed yet" in capsys.readouterr().out

    def test_json_output(self, project, capsys) -> None:
        proj, ids = project

        _show(proj, "2026-09-06", as_json=True)
        data = json.loads(capsys.readouterr().out)

        assert data["counts"] == {"live": 1, "ended": 1, "unknown": 1}
        assert [c["id"] for c in data["hard_constraints"]] == [ids["new"]]
        assert [c["id"] for c in data["unknown"]] == [ids["pre021"]]
        assert data["first_recorded_end"] == T2

    def test_an_unparseable_date_exits_nonzero(self, project, capsys) -> None:
        proj, _ = project

        with pytest.raises(SystemExit) as exc:
            _show(proj, "next week")

        assert exc.value.code == 1
        assert "YYYY-MM-DD" in capsys.readouterr().err

    def test_without_as_of_show_is_unchanged(self, project, capsys) -> None:
        proj, _ = project

        _show(proj, None, as_json=True)
        data = json.loads(capsys.readouterr().out)

        assert "hard_constraints" in data and "counts" not in data
