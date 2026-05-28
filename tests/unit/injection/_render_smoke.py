"""One-off smoke renderer to visually inspect Phase B output.

Not a test — run as `python tests/unit/injection/_render_smoke.py`.
"""
from memlora.injection.template import InjectionContext, render_injection
from memlora.storage.events import Event
from memlora.storage.symbol_files import CoverageStats, RefreshInfo
from memlora.symbols.projection import SkeletonEntry, SkeletonMethod


def main() -> None:
    ctx = InjectionContext(
        project_name="taskflow_benchmark",
        session_number=2,
        total_sessions=2,
        state_version=1,
        hard_constraints=[Event(
            project_id="p", session_id="s1",
            event_type="CONSTRAINT_HARD",
            payload={"description": "Use PostgreSQL only, never SQLite."},
            content_hash="h" * 64, weight=1.0,
        )],
        graveyard=[],
        components=[],
        decisions=[],
        active_threads=[Event(
            project_id="p", session_id="s2",
            event_type="THREAD_OPEN",
            payload={"description": "JWT auth end-to-end"},
            content_hash="t" * 64, weight=1.0,
        )],
        summary_text="Python project.",
        token_budget=2000,
        hook_policy="strict",
        retry_window_seconds=60,
        skeleton=[SkeletonEntry(
            path="app/core/security.py",
            imports=[],
            classes=[],
            functions=[
                SkeletonMethod(name="hash_password", signature="(plain:str)", return_type="str"),
                SkeletonMethod(name="verify_password", signature="(plain:str, hashed:str)", return_type="bool"),
            ],
        )],
        skeleton_coverage=CoverageStats(scanned=17, with_symbols=14, parse_errors=1, ignored=0, pending=0),
        skeleton_refresh=RefreshInfo(
            path="app/core/security.py",
            refreshed_in_session="a22310fd-3b13",
            last_action="Edit",
            refreshed_at=1779629999000,
        ),
    )
    print(render_injection(ctx))


if __name__ == "__main__":
    main()
