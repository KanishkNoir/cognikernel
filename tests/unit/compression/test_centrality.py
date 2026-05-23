"""Tests for PageRank-based file centrality."""
import pytest
from memlora.compression.centrality import centrality_factor, compute_file_centrality


class TestComputeFileCentrality:
    def test_empty_graph_returns_empty(self) -> None:
        assert compute_file_centrality({}) == {}

    def test_single_file_no_imports(self) -> None:
        result = compute_file_centrality({"a.py": []})
        assert "a.py" in result
        assert result["a.py"] > 0.0

    def test_all_files_present_in_result(self) -> None:
        graph = {"a.py": ["b.py", "c.py"], "b.py": ["c.py"], "c.py": []}
        result = compute_file_centrality(graph)
        assert "a.py" in result
        assert "b.py" in result
        assert "c.py" in result

    def test_hub_has_higher_centrality_than_leaf(self) -> None:
        # c.py is imported by both a and b → higher centrality
        graph = {"a.py": ["c.py"], "b.py": ["c.py"], "c.py": []}
        result = compute_file_centrality(graph)
        assert result["c.py"] > result["a.py"]

    def test_scores_are_positive(self) -> None:
        graph = {"a.py": ["b.py"], "b.py": ["c.py"], "c.py": []}
        result = compute_file_centrality(graph)
        assert all(v > 0.0 for v in result.values())

    def test_imported_only_file_in_result(self) -> None:
        # "d.py" appears only as a target, not as a source key
        graph = {"a.py": ["d.py"]}
        result = compute_file_centrality(graph)
        assert "d.py" in result

    def test_linear_chain_source_lower_than_sink(self) -> None:
        # a → b → c: c gets inbound from b, b gets inbound from a → c should be highest
        graph = {"a.py": ["b.py"], "b.py": ["c.py"], "c.py": []}
        result = compute_file_centrality(graph)
        assert result["c.py"] > result["a.py"]

    def test_custom_damping(self) -> None:
        graph = {"a.py": ["b.py"], "b.py": []}
        r85 = compute_file_centrality(graph, damping=0.85)
        r50 = compute_file_centrality(graph, damping=0.50)
        # Both produce valid results, scores differ
        assert r85["b.py"] != pytest.approx(r50["b.py"], rel=1e-3)


class TestCentralityFactor:
    def test_empty_file_paths_returns_one(self) -> None:
        assert centrality_factor([], {"a.py": 0.5}) == pytest.approx(1.0)

    def test_empty_centrality_map_returns_one(self) -> None:
        assert centrality_factor(["a.py"], {}) == pytest.approx(1.0)

    def test_max_centrality_returns_one_point_five(self) -> None:
        cmap = {"a.py": 1.0}
        assert centrality_factor(["a.py"], cmap) == pytest.approx(1.5)

    def test_zero_centrality_returns_point_five(self) -> None:
        cmap = {"a.py": 0.0, "b.py": 1.0}
        assert centrality_factor(["a.py"], cmap) == pytest.approx(0.5)

    def test_unknown_file_returns_low_end(self) -> None:
        cmap = {"a.py": 1.0}
        result = centrality_factor(["unknown.py"], cmap)
        assert result == pytest.approx(0.5)

    def test_uses_max_across_files(self) -> None:
        cmap = {"a.py": 0.2, "b.py": 0.8, "c.py": 1.0}
        # max centrality = 1.0 (c.py); best match = b.py at 0.8 → 0.5 + 0.8 = 1.3
        result = centrality_factor(["a.py", "b.py"], cmap)
        assert result == pytest.approx(1.3)

    def test_range_always_between_point_five_and_one_point_five(self) -> None:
        cmap = {"high.py": 0.9, "low.py": 0.1, "max.py": 1.0}
        for path in ["high.py", "low.py", "max.py"]:
            f = centrality_factor([path], cmap)
            assert 0.5 <= f <= 1.5 + 1e-9

    def test_all_zero_centrality_map_returns_one(self) -> None:
        cmap = {"a.py": 0.0, "b.py": 0.0}
        assert centrality_factor(["a.py"], cmap) == pytest.approx(1.0)
