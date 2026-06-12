import os
from pathlib import Path

import pytest

# `memlora init` now bundles a one-time embedding-model download (~130MB). Each
# init test uses a fresh MEMLORA_DIR, so without this guard the suite would
# re-download per test. Disable the auto-warm for the whole test session;
# embedding behavior is covered explicitly by the embedding tests.
os.environ.setdefault("MEMLORA_DISABLE_AUTO_WARM", "1")

from memlora.storage.connection import get_connection
from memlora.storage.migrations import run_migrations


@pytest.fixture(autouse=True)
def _isolate_memlora_home(tmp_path: Path, monkeypatch) -> None:
    """Tests must never touch the user's real ~/.memlora.

    Without this, any test that exercises session_capture/process_jobs against
    a tmp_path project writes state into the user-global store keyed by the
    project-path hash — which both pollutes the user's data and makes runs
    order/`--basetemp`-dependent (observed: dead-letter state from a previous
    run failing a later one). Tests that need a specific home still win: a
    test-level monkeypatch.setenv or an explicit subprocess env overrides this.
    """
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "_memlora_home"))


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Return path to a freshly migrated SQLite database."""
    db_path = tmp_path / "test.db"
    with get_connection(db_path) as conn:
        run_migrations(conn)
    return db_path


@pytest.fixture
def conn(tmp_db: Path):
    """Yield an open, migrated connection for use within a single test."""
    with get_connection(tmp_db) as c:
        yield c
