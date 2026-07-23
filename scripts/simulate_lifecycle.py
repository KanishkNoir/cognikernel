"""
CogniKernel Symbol Graph — Full Lifecycle Simulation

Simulates a complete coding session:
  Phase 1: Session 1 — Claude writes files, PostToolUse hook fires after each write
  Phase 2: Session 1 ends — Stop hook runs session_end(), symbol graph snapshotted
  Phase 3: Session 2 starts — MCP get_session_state loads the skeleton injection block

Output shows the DB state after each event so you can see the graph evolving in real time.
"""
from __future__ import annotations

import sys
import os
import shutil
import sqlite3
import tempfile
import textwrap
import time
import json
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colours (degrade to plain if not supported)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    USE_COLOR = True
except Exception:
    USE_COLOR = os.environ.get("TERM") is not None

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def CYAN(t):   return _c("96", t)
def YELLOW(t): return _c("93", t)
def GREEN(t):  return _c("92", t)
def BOLD(t):   return _c("1",  t)
def DIM(t):    return _c("2",  t)
def RED(t):    return _c("91", t)

DIVIDER = "=" * 72

def header(title: str) -> None:
    print()
    print(CYAN(DIVIDER))
    print(CYAN(f"  {title}"))
    print(CYAN(DIVIDER))

def step(label: str, detail: str = "") -> None:
    marker = YELLOW(">>")
    print(f"\n{marker} {BOLD(label)}")
    if detail:
        for line in detail.strip().splitlines():
            print(DIM(f"  {line}"))

def info(msg: str) -> None:
    print(f"  . {msg}")

def ok(msg: str) -> None:
    print(f"  [OK] {GREEN(msg)}")

def show_db_nodes(conn: sqlite3.Connection, project_id: str, label: str = "") -> None:
    rows = conn.execute(
        "SELECT path, node_type, name, parent_name, fields FROM symbol_nodes "
        "WHERE project_id = ? ORDER BY path, node_type, name",
        (project_id,),
    ).fetchall()
    print(f"\n  {DIM('DB symbol_nodes')} {DIM(label)} ({len(rows)} rows):")
    for r in rows:
        path, ntype, name, parent, fields = r
        short_path = path.replace("\\", "/").split("/")[-1]
        tag = {"class": "CLS", "method": "MTH", "function": "FN "}.get(ntype, ntype[:3].upper())
        parent_str = f"/{parent}" if parent else ""
        fields_str = f"  [{fields}]" if fields else ""
        print(f"    {DIM(tag)} {short_path}:{name}{parent_str}{fields_str}")


# ──────────────────────────────────────────────────────────────────────────────
# Sample source files (what Claude "writes" during the session)
# ──────────────────────────────────────────────────────────────────────────────

MODELS_V1 = """\
from sqlalchemy import Column, Integer, String
from database import get_db

class Quote(Base):
    id: int
    text: str

    def create(self, text: str) -> "Quote":
        pass

    def delete(self) -> None:
        pass
"""

MODELS_V2 = """\
from sqlalchemy import Column, Integer, String
from database import get_db

class Quote(Base):
    id: int
    text: str
    author_id: int

    def create(self, text: str, author_id: int) -> "Quote":
        pass

    def get(self, id: int) -> "Quote":
        pass

    def delete(self) -> None:
        pass


class Author(Base):
    id: int
    name: str

    def get_by_name(self, name: str) -> "Author":
        pass
"""

DATABASE_PY = """\
from sqlalchemy.orm import Session

def get_db() -> Session:
    pass

def init_db() -> None:
    pass
"""

API_PY = """\
from models import Quote
from database import get_db

def get_quotes() -> list:
    pass

def create_quote(data: dict) -> Quote:
    pass
"""

SCHEMAS_PY = """\
from dataclasses import dataclass

class QuoteCreate:
    text: str
    author_id: int

class QuoteResponse:
    id: int
    text: str
"""


def simulate_posttool_hook(
    file_path: str,
    project_path: str,
    project_id: str,
    db_path: str,
    conn: sqlite3.Connection,
    label: str,
) -> None:
    """Replicates what cognikernel_posttool_hook.py does after a Write/Edit."""
    from cognikernel.symbols.extractor import build_symbol_update
    from cognikernel.symbols.store import apply_symbol_update
    from cognikernel.extraction.git_augment import FileChange

    rel_path = str(Path(file_path).relative_to(project_path)).replace("\\", "/")
    changed_files = [FileChange(path=rel_path, change_type="modified", lines_changed=0)]
    update = build_symbol_update(project_id, project_path, changed_files)
    apply_symbol_update(conn, update)
    ok(f"PostToolUse hook fired → re-parsed {rel_path}")


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="ck_sim_")
    try:
        _run_simulation(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_simulation(tmp: str) -> None:
    from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
    from cognikernel.storage.migrations import run_migrations
    from cognikernel.config import Config
    from cognikernel.integration.session import session_end, render_state
    from cognikernel.symbols.store import load_symbol_nodes, load_symbol_edges
    from cognikernel.symbols.projection import compress_to_skeleton
    from cognikernel.symbols.render import render_skeleton_section

    config = Config.load()
    project_id = hash_project_path(tmp)
    db_path = get_db_path(config, project_id)

    src = Path(tmp) / "src"
    src.mkdir()
    (Path(tmp) / ".claude").mkdir()
    (Path(tmp) / ".claude" / "settings.json").write_text("{}")

    # Bootstrap DB
    with get_connection(db_path) as conn:
        run_migrations(conn)

    # ──────────────────────────────────────────────────────────────────────────
    header("SESSION 1  --  Claude is coding")
    # ──────────────────────────────────────────────────────────────────────────

    step("Claude writes database.py",
         "Tool: Write  |  file_path: src/database.py")
    (src / "database.py").write_text(DATABASE_PY)
    with get_connection(db_path) as conn:
        simulate_posttool_hook(
            str(src / "database.py"), tmp, project_id, db_path, conn,
            label="after Write database.py",
        )
        show_db_nodes(conn, project_id, "(after write database.py)")

    step("Claude writes models.py  (initial — Quote only, no Author yet)",
         "Tool: Write  |  file_path: src/models.py")
    (src / "models.py").write_text(MODELS_V1)
    with get_connection(db_path) as conn:
        simulate_posttool_hook(
            str(src / "models.py"), tmp, project_id, db_path, conn,
            label="after Write models.py v1",
        )
        show_db_nodes(conn, project_id, "(after write models.py v1)")

    step("Claude writes src/api/quotes.py",
         "Tool: Write  |  file_path: src/api/quotes.py")
    (src / "api").mkdir(exist_ok=True)
    (src / "api" / "quotes.py").write_text(API_PY)
    with get_connection(db_path) as conn:
        simulate_posttool_hook(
            str(src / "api" / "quotes.py"), tmp, project_id, db_path, conn,
            label="after Write quotes.py",
        )
        show_db_nodes(conn, project_id, "(after write quotes.py)")

    step("Claude EDITS models.py — adds Author class + author_id field",
         "Tool: Edit  |  file_path: src/models.py\n"
         "Adds: Author(Base), Quote.author_id, Quote.get() method")
    (src / "models.py").write_text(MODELS_V2)
    with get_connection(db_path) as conn:
        simulate_posttool_hook(
            str(src / "models.py"), tmp, project_id, db_path, conn,
            label="after Edit models.py v2",
        )
        show_db_nodes(conn, project_id, "(after edit models.py v2 — graph updated immediately)")

    step("Claude writes schemas.py  (new file, no imports from other modules)",
         "Tool: Write  |  file_path: src/schemas.py")
    (src / "schemas.py").write_text(SCHEMAS_PY)
    with get_connection(db_path) as conn:
        simulate_posttool_hook(
            str(src / "schemas.py"), tmp, project_id, db_path, conn,
            label="after Write schemas.py",
        )

    # ──────────────────────────────────────────────────────────────────────────
    header("SESSION 1 END  --  Stop hook fires")
    # ──────────────────────────────────────────────────────────────────────────

    step("Stop hook calls session_end()",
         "Transcript extracted → events merged\n"
         "git_diff=None → no changed_files → walk is skipped (hooks already kept graph current)\n"
         "Symbol graph already up-to-date from PostToolUse hook calls")

    FAKE_TRANSCRIPT = textwrap.dedent("""\
        Assistant: I'll start by creating the database module.
        User: Looks good. Now add the Quote model.
        Assistant: We decided to use SQLAlchemy Base as the ORM base class for all models.
        Assistant: We decided to store author_id as a foreign key on Quote for multi-author support.
        User: Can you also add an Author model?
        Assistant: Added Author class with id and name fields.
    """)

    stats = session_end(
        project_path=tmp,
        session_id="sim-session-001",
        transcript=FAKE_TRANSCRIPT,
        config=config,
        git_diff=None,
    )
    ok(f"session_end() complete — events extracted: {stats['extracted']}, inserted: {stats['inserted']}")

    # ──────────────────────────────────────────────────────────────────────────
    header("SESSION 2 START  --  MCP get_session_state fires")
    # ──────────────────────────────────────────────────────────────────────────

    step("render_state() called — loads nodes + edges from DB, compresses to skeleton")

    with get_connection(db_path) as conn:
        nodes = load_symbol_nodes(conn, project_id)
        edges = load_symbol_edges(conn, project_id)

    skeleton = compress_to_skeleton(nodes, edges, budget_tokens=200)

    info(f"Loaded {len(nodes)} symbol nodes, {len(edges)} local edges")
    info(f"Compressed to {len(skeleton)} SkeletonEntry objects")
    total_tokens = sum(e.token_estimate for e in skeleton)
    info(f"Total skeleton token estimate: {total_tokens} / 200 budget")

    step("Skeleton section rendered")
    skeleton_text = render_skeleton_section(skeleton)
    print()
    for line in skeleton_text.encode("ascii", errors="replace").decode("ascii").splitlines():
        print(f"  {line}")

    step("Full injection block (what Claude sees at session start)")
    full_block = render_state(tmp, config=config)
    print()
    for line in full_block.encode("ascii", errors="replace").decode("ascii").splitlines():
        print(f"  {line}")

    # ──────────────────────────────────────────────────────────────────────────
    header("SIMULATION COMPLETE")
    # ──────────────────────────────────────────────────────────────────────────

    ok("Symbol graph stayed current throughout Session 1 via PostToolUse hook")
    ok("Session 2 injection block contains full class/method skeleton")
    ok("Claude can skip exploratory Reads of models.py, database.py, quotes.py")

    saved_tokens = len(nodes) * 8 + len(edges) * 3  # rough heuristic
    print(f"\n  Estimated token savings vs cold-start file reads: ~{saved_tokens} tokens")


if __name__ == "__main__":
    main()
