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
