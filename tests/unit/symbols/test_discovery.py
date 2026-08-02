"""Scoping rules for symbol-graph file discovery (spec D3).

Motivation: a store sweep found one project whose symbol graph was 4,655 of
4,934 nodes (94%) from vendored and cache trees — .uv-cache and .pytest_tmp —
while the rendered Codebase skeleton was the single largest section of the
injection block. The budget was being spent describing the `attrs` library
instead of the user's own code.
"""
from pathlib import Path

from cognikernel.symbols.extractor import _discover_project_paths


def _touch(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x = 1\n", encoding="utf-8")


class TestSkipDirs:
    def test_skips_uv_cache(self, tmp_path: Path) -> None:
        _touch(tmp_path, "src/app.py")
        _touch(tmp_path, ".uv-cache/archive-v0/attr/_make.py")
        assert set(_discover_project_paths(tmp_path)) == {"src/app.py"}

    def test_skips_claude_and_pytest_tmp(self, tmp_path: Path) -> None:
        _touch(tmp_path, "src/app.py")
        _touch(tmp_path, ".claude/hooks/x.py")
        _touch(tmp_path, ".pytest_tmp/basetemp/proj/main.py")
        assert set(_discover_project_paths(tmp_path)) == {"src/app.py"}


class TestGitignore:
    def test_respects_gitignore_entry(self, tmp_path: Path) -> None:
        _touch(tmp_path, "src/app.py")
        _touch(tmp_path, "generated/pb2.py")
        (tmp_path / ".gitignore").write_text("generated/\n", encoding="utf-8")
        assert set(_discover_project_paths(tmp_path)) == {"src/app.py"}

    def test_absent_gitignore_is_not_an_error(self, tmp_path: Path) -> None:
        _touch(tmp_path, "src/app.py")
        assert set(_discover_project_paths(tmp_path)) == {"src/app.py"}

    def test_negation_pattern_is_ignored_not_misapplied(self, tmp_path: Path) -> None:
        # We do not implement negation; a '!' line must be skipped rather than
        # treated as a literal pattern that could hide real source.
        _touch(tmp_path, "src/app.py")
        (tmp_path / ".gitignore").write_text("!src\n", encoding="utf-8")
        assert set(_discover_project_paths(tmp_path)) == {"src/app.py"}


class TestBudgetPriority:
    def test_deep_first_party_source_beats_shallow_noise(self, tmp_path: Path) -> None:
        """rglob yields shallow files first, so collection order alone favours
        shallow paths regardless of whose code they are. A nested module under
        src/ must still make the budget ahead of 600 shallow third-party files.
        """
        for i in range(600):
            _touch(tmp_path, f"third_party/mod{i}.py")
        _touch(tmp_path, "src/cognikernel/storage/deep/connection.py")
        found = _discover_project_paths(tmp_path)
        assert "src/cognikernel/storage/deep/connection.py" in found
        assert len(found) <= 500

    def test_budget_is_enforced(self, tmp_path: Path) -> None:
        for i in range(700):
            _touch(tmp_path, f"src/mod{i}.py")
        assert len(_discover_project_paths(tmp_path)) == 500
