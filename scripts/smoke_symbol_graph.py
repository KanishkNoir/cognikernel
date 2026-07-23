"""End-to-end smoke test for the Symbol Graph layer.

Creates a temporary project with sample Python files, runs session_end() to
trigger symbol extraction, then calls render_state() to verify the skeleton
section appears in the injection block.
"""
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Ensure CogniKernel src is on the path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

MODELS_PY = '''\
from sqlalchemy import Column, Integer, String
from database import get_db

class Quote(Base):
    id: int
    text: str
    author_id: int

    def create(self, text: str, author_id: int) -> "Quote":
        pass

    def get(self, id: int):
        pass

    def delete(self) -> None:
        pass


class Author(Base):
    id: int
    name: str

    def get_by_name(self, name: str) -> "Author":
        pass
'''

DATABASE_PY = '''\
from sqlalchemy.orm import Session

def get_db() -> Session:
    pass

def init_db() -> None:
    pass
'''

API_PY = '''\
from models import Quote
from database import get_db

def get_quotes() -> list:
    pass

def create_quote(data: dict) -> Quote:
    pass
'''


def main():
    tmp = tempfile.mkdtemp(prefix="ck_smoke_")
    try:
        # Create project structure
        src = Path(tmp) / "src"
        src.mkdir()
        (src / "models.py").write_text(MODELS_PY)
        (src / "database.py").write_text(DATABASE_PY)
        api_dir = src / "api"
        api_dir.mkdir()
        (api_dir / "quotes.py").write_text(API_PY)
        # Create .claude/settings.json so _find_project_root can locate it
        (Path(tmp) / ".claude").mkdir()
        (Path(tmp) / ".claude" / "settings.json").write_text("{}")

        print(f"[smoke] Temp project at: {tmp}")

        # Import cognikernel modules
        from cognikernel.integration.session import session_end, render_state
        from cognikernel.storage.connection import hash_project_path, get_db_path
        from cognikernel.config import Config

        config = Config.load()
        project_id = hash_project_path(tmp)
        db_path = get_db_path(config, project_id)
        print(f"[smoke] DB path: {db_path}")

        # Run session_end with no transcript (no events) but will walk filesystem
        session_end(
            project_path=tmp,
            session_id="smoke-test-001",
            transcript="",
            config=config,
            git_diff=None,
        )
        print("[smoke] session_end() completed")

        # Now render_state
        block = render_state(tmp, config=config)
        print("[smoke] render_state() completed")
        print()
        print("=" * 60)
        print(block.encode("ascii", errors="replace").decode("ascii"))
        print("=" * 60)

        # Assertions
        assert "### Codebase skeleton" in block, "FAIL: '### Codebase skeleton' not found in output"
        assert "Quote" in block, "FAIL: 'Quote' class not in skeleton"
        assert "Author" in block, "FAIL: 'Author' class not in skeleton"
        assert "get_db" in block, "FAIL: 'get_db' function not in skeleton"
        print()
        print("[smoke] ALL ASSERTIONS PASSED")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
