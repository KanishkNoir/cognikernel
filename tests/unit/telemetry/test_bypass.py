"""Tests for H7 bypass-to-update detection (telemetry.bypass).

Synthetic JSONL → known counts. Proves the parser ahead of the first real
session, where it reconciles against the run sheet's hand count.
"""
from __future__ import annotations

import json

from memlora.telemetry.bypass import analyze_bypass

_CWD = "C:/proj"


def _tool_use(name: str, **inp) -> dict:
    return {"type": "tool_use", "name": name, "input": inp}


def _assistant(*blocks: dict, cwd: str = _CWD) -> dict:
    return {"type": "assistant", "cwd": cwd, "message": {"content": list(blocks)}}


def _jsonl(*records: dict) -> str:
    return "\n".join(json.dumps(r) for r in records)


class TestHeadlineDefinition:
    def test_edit_without_read_is_bypass(self) -> None:
        r = analyze_bypass(_jsonl(_assistant(_tool_use("Edit", file_path="app/main.py"))))
        assert r.total_modifications == 1
        assert r.bypass_to_update == 1
        assert r.edit_without_read == 1
        assert r.bypass_rate == 1.0

    def test_read_then_edit_is_not_bypass(self) -> None:
        r = analyze_bypass(_jsonl(
            _assistant(_tool_use("Read", file_path="app/main.py")),
            _assistant(_tool_use("Edit", file_path="app/main.py")),
        ))
        assert r.total_modifications == 1
        assert r.bypass_to_update == 0
        assert r.bypass_rate == 0.0

    def test_read_one_file_edit_another_is_bypass(self) -> None:
        r = analyze_bypass(_jsonl(
            _assistant(_tool_use("Read", file_path="app/a.py")),
            _assistant(_tool_use("Edit", file_path="app/b.py")),
        ))
        assert r.bypass_to_update == 1
        assert r.edit_without_read == 1

    def test_literal_headline_has_no_write_exclusion(self) -> None:
        """Write-then-Edit with no Read: headline counts BOTH (literal run_sheet
        definition); diagnostics split write_first_touch vs edit_without_read."""
        r = analyze_bypass(_jsonl(
            _assistant(_tool_use("Write", file_path="app/new.py")),
            _assistant(_tool_use("Edit", file_path="app/new.py")),
        ))
        assert r.total_modifications == 2
        assert r.bypass_to_update == 2
        assert r.write_first_touch == 1
        assert r.edit_without_read == 1

    def test_event_records_prior_write_for_self_created_files(self) -> None:
        r = analyze_bypass(_jsonl(
            _assistant(_tool_use("Write", file_path="app/new.py")),
            _assistant(_tool_use("Edit", file_path="app/new.py")),
        ))
        assert len(r.events) == 2
        assert r.events[0].prior_write is False  # the Write
        assert r.events[1].prior_write is True   # the Edit of the just-created file


class TestPathNormalization:
    def test_backslash_vs_forward_slash_same_file(self) -> None:
        r = analyze_bypass(_jsonl(
            _assistant(_tool_use("Read", file_path="C:\\proj\\app\\main.py")),
            _assistant(_tool_use("Edit", file_path="C:/proj/app/main.py")),
        ))
        assert r.bypass_to_update == 0

    def test_absolute_read_relative_edit_via_cwd(self) -> None:
        """Read emits an absolute path, Edit a relative one — cwd-based
        canonicalization maps them equal so this is NOT a false bypass."""
        r = analyze_bypass(_jsonl(
            _assistant(_tool_use("Read", file_path="C:/proj/app/main.py")),
            _assistant(_tool_use("Edit", file_path="app/main.py")),
        ))
        assert r.bypass_to_update == 0


class TestParsing:
    def test_top_level_tool_use_record_is_ignored(self) -> None:
        """Tool calls live in assistant message.content; a top-level type=tool_use
        record must not be counted (the bug in count_tool_calls.py)."""
        r = analyze_bypass(_jsonl(
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "x.py"}},
            _assistant(_tool_use("Edit", file_path="y.py")),
        ))
        assert r.total_modifications == 1
        assert r.distinct_files_modified == 1

    def test_notebook_edit_uses_notebook_path_key(self) -> None:
        r = analyze_bypass(_jsonl(_assistant(_tool_use("NotebookEdit", notebook_path="nb.ipynb"))))
        assert r.total_modifications == 1
        assert r.edit_without_read == 1

    def test_non_file_and_search_tools_ignored(self) -> None:
        r = analyze_bypass(_jsonl(_assistant(
            _tool_use("Bash", command="ls"),
            _tool_use("Grep", pattern="foo"),
            _tool_use("Glob", pattern="*.py"),
        )))
        assert r.total_modifications == 0
        assert r.read_calls == 0

    def test_grep_does_not_count_as_a_read(self) -> None:
        """Glob/Grep don't establish file content, so editing after them is a bypass."""
        r = analyze_bypass(_jsonl(
            _assistant(_tool_use("Grep", pattern="def foo")),
            _assistant(_tool_use("Edit", file_path="app/main.py")),
        ))
        assert r.bypass_to_update == 1

    def test_malformed_lines_are_skipped(self) -> None:
        text = "not json\n" + _jsonl(_assistant(_tool_use("Edit", file_path="a.py"))) + "\n{bad"
        r = analyze_bypass(text)
        assert r.total_modifications == 1

    def test_no_modifications_yields_zero_rate(self) -> None:
        r = analyze_bypass(_jsonl(_assistant(_tool_use("Read", file_path="a.py"))))
        assert r.total_modifications == 0
        assert r.bypass_rate == 0.0
        assert r.read_calls == 1
