"""Tests for memlora.integration.session._compute_hot_files.

The function aggregates COMPONENT_STATUS events into the `### Most active
files` injection section. It must defensively reject bare-basename rows so
projects whose DB predates the extraction-time filter don't poison the
injection.
"""
from __future__ import annotations

from memlora.integration.session import _compute_hot_files
from memlora.storage.events import Event


def _component(path: str, mentions: int = 2, intent: str = "") -> Event:
    return Event(
        project_id="p1",
        session_id="s1",
        event_type="COMPONENT_STATUS",
        payload={
            "path": path,
            "intent": intent or path,
            "description": f"{path} modified",
            "rationale": "",
        },
        content_hash=("h" + path)[:32].ljust(64, "0"),
        weight=0.6,
        mention_count=mentions,
    )


class TestComputeHotFilesBareBasenames:
    def test_bare_basename_filtered_out(self) -> None:
        """A bare ``env.py`` row from an old DB must not appear in hot files."""
        events = [
            _component("env.py", mentions=8),
            _component("alembic/env.py", mentions=2),
        ]
        hot = _compute_hot_files(events)
        paths = [p for (p, _n, _intent) in hot]
        assert "env.py" not in paths, (
            "_compute_hot_files leaked a bare-basename path into hot files; "
            "the defensive filter against legacy DBs is not firing."
        )
        assert "alembic/env.py" in paths

    def test_qualified_paths_pass_through_unchanged(self) -> None:
        events = [
            _component("backend/app/main.py", mentions=5),
            _component("backend/app/core/config.py", mentions=3),
        ]
        hot = _compute_hot_files(events)
        paths = [p for (p, _n, _intent) in hot]
        assert paths == ["backend/app/main.py", "backend/app/core/config.py"]

    def test_min_mentions_threshold_still_applies(self) -> None:
        events = [
            _component("backend/app/main.py", mentions=1),  # below threshold
            _component("backend/app/core/config.py", mentions=2),  # at threshold
        ]
        hot = _compute_hot_files(events, min_mentions=2)
        paths = [p for (p, _n, _intent) in hot]
        assert paths == ["backend/app/core/config.py"]

    def test_arm_c_regression_paths(self) -> None:
        """Reproduces the exact set of bare-basename rows seen in the
        taskflow_cogni Arm-C-v2 DB: alembic.ini, env.py, config.py,
        components.json, tailwind.config.ts. All must be filtered."""
        bare = [
            _component("alembic.ini", mentions=8),
            _component("env.py", mentions=9),
            _component("config.py", mentions=2),
            _component("components.json", mentions=2),
            _component("tailwind.config.ts", mentions=2),
        ]
        qualified = [
            _component("db/migrations/0001_initial_schema.sql", mentions=9),
            _component("backend/app/core/security.py", mentions=3),
        ]
        hot = _compute_hot_files(bare + qualified)
        paths = [p for (p, _n, _intent) in hot]
        for noise in ("alembic.ini", "env.py", "config.py",
                       "components.json", "tailwind.config.ts"):
            assert noise not in paths
        for keep in ("db/migrations/0001_initial_schema.sql",
                     "backend/app/core/security.py"):
            assert keep in paths
