from pathlib import Path

import pytest

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
