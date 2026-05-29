"""Tests for the deterministic injection-block cost meter."""
from __future__ import annotations

from pathlib import Path

import pytest

from memlora.config import Config
from memlora.integration.session import init_project, session_end
from memlora.telemetry.render_cost import (
    diff_reports,
    render_cost_report,
    section_token_report,
)


SAMPLE_BLOCK = """\
## Session context [auto-generated — do not edit]
project: demo · session 3 of 3 · state v1

### Hard constraints — never violate
- Never log secrets.

### Active thread
Working on: JWT auth end-to-end.

### Codebase skeleton
app/auth.py
  .login(user:str)→Token
"""


class TestSectionTokenReport:
    def test_splits_into_named_sections(self) -> None:
        report = section_token_report(SAMPLE_BLOCK)
        names = report["sections"].keys()
        assert "Session context [auto-generated — do not edit]" in names
        assert "Hard constraints — never violate" in names
        assert "Active thread" in names
        assert "Codebase skeleton" in names

    def test_total_tokens_positive_and_per_section_counted(self) -> None:
        report = section_token_report(SAMPLE_BLOCK)
        assert report["total_tokens"] > 0
        assert all(v > 0 for v in report["sections"].values())

    def test_empty_block(self) -> None:
        report = section_token_report("")
        assert report["total_tokens"] == 0
        assert report["sections"] == {}


class TestDiffReports:
    def test_total_and_section_delta(self) -> None:
        before = section_token_report(SAMPLE_BLOCK)
        # Drop the skeleton section.
        trimmed = "\n".join(
            line for line in SAMPLE_BLOCK.split("\n")
            if "Codebase skeleton" not in line and "app/auth.py" not in line
            and ".login" not in line
        )
        after = section_token_report(trimmed)
        d = diff_reports(before, after)
        assert d["total_delta"] <= 0
        assert d["total_before"] == before["total_tokens"]
        assert d["total_after"] == after["total_tokens"]


class TestRenderCostReport:
    def test_report_for_initialised_project(self, tmp_path: Path) -> None:
        cfg = Config(memlora_dir=tmp_path / "memlora")
        project = tmp_path / "proj"
        project.mkdir()
        init_project(project, config=cfg)
        session_end(
            str(project), "s1",
            "Hard constraint: never use synchronous blocking I/O in async paths.",
            config=cfg,
        )
        report = render_cost_report(str(project), config=cfg)
        assert report["total_tokens"] > 0
        assert report["project_path"] == str(project)
        assert isinstance(report["sections"], dict)

    def test_render_stays_under_single_ceiling(self, tmp_path: Path) -> None:
        from memlora.config import DEFAULT_TOKEN_BUDGET
        cfg = Config(memlora_dir=tmp_path / "memlora")
        project = tmp_path / "proj"
        project.mkdir()
        init_project(project, config=cfg)
        session_end(
            str(project), "s1",
            "Hard constraint: never use blocking I/O in async paths. "
            "We decided to use SQLite WAL mode for storage.",
            config=cfg,
        )
        assert cfg.token_budget == DEFAULT_TOKEN_BUDGET
        report = render_cost_report(str(project), config=cfg)
        assert report["total_tokens"] <= cfg.token_budget
