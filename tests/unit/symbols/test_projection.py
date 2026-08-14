"""Unit tests for symbol graph compression (projection)."""
import pytest
from cognikernel.symbols.extractor import SymbolNode, SymbolEdge
from cognikernel.symbols.projection import compress_to_skeleton, SkeletonEntry


def _node(path: str, node_type: str, name: str, parent: str = "",
          sig: str = "", ret: str = "", fields: str = "") -> SymbolNode:
    return SymbolNode(
        path=path, node_type=node_type, name=name, parent_name=parent,
        signature=sig, return_type=ret, fields=fields,
        project_id="p1", updated_at=0,
    )


def _edge(from_path: str, to_path: str, external: bool = False) -> SymbolEdge:
    return SymbolEdge(
        project_id="p1", from_path=from_path, to_path=to_path,
        edge_type="imports", is_external=external,
    )


class TestCompressToSkeleton:
    def test_empty_returns_empty(self) -> None:
        assert compress_to_skeleton([], []) == []

    def test_central_file_survives_budget_over_leaf(self) -> None:
        """Unit 6: under budget pressure the PageRank-central file is kept and a
        leaf with the same raw symbol count is dropped first."""
        nodes = [
            _node("hub.py", "function", "h", sig="()"),
            _node("leaf.py", "function", "l", sig="()"),
            _node("a.py", "function", "a", sig="()"),
            _node("b.py", "function", "b", sig="()"),
        ]
        # a and b both import hub → hub is central; nobody imports leaf.
        edges = [_edge("a.py", "hub.py"), _edge("b.py", "hub.py")]
        entries = compress_to_skeleton(nodes, edges, budget_tokens=3)
        paths = {e.path for e in entries}
        assert "hub.py" in paths
        assert "leaf.py" not in paths

    def test_functions_ranked_by_importance_not_alphabetical(self) -> None:
        """Unit 6: a private helper sorting first alphabetically is dropped before
        an API route when the per-file function cap bites."""
        nodes = [
            _node("api.py", "function", f"_helper{i}", sig="()")
            for i in range(10)
        ]
        nodes.append(_node("api.py", "function", "zzz_route", sig="()", fields="GET /x"))
        entries = compress_to_skeleton(nodes, [])
        names = {f.name for f in entries[0].functions}
        assert "zzz_route" in names  # route kept despite sorting last alphabetically

    def test_single_file_class_and_method(self) -> None:
        nodes = [
            _node("src/models.py", "class", "Quote", fields="id:int, text:str"),
            _node("src/models.py", "method", "create", parent="Quote", sig="(text:str)", ret="Quote"),
        ]
        entries = compress_to_skeleton(nodes, [])
        assert len(entries) == 1
        assert entries[0].path == "src/models.py"
        assert len(entries[0].classes) == 1
        assert entries[0].classes[0].name == "Quote"
        assert entries[0].classes[0].fields == "id:int, text:str"

    def test_top_level_function_included(self) -> None:
        nodes = [_node("src/db.py", "function", "get_db", sig="()", ret="Session")]
        entries = compress_to_skeleton(nodes, [])
        assert entries[0].functions[0].name == "get_db"

    def test_class_limit_five(self) -> None:
        nodes = [_node("src/m.py", "class", f"Class{i}") for i in range(8)]
        entries = compress_to_skeleton(nodes, [])
        assert len(entries[0].classes) <= 5

    def test_method_limit_five(self) -> None:
        nodes = [_node("src/m.py", "class", "Foo")]
        nodes += [_node("src/m.py", "method", f"m{i}", parent="Foo") for i in range(8)]
        entries = compress_to_skeleton(nodes, [])
        assert len(entries[0].classes[0].methods) <= 5

    def test_local_imports_in_entry(self) -> None:
        nodes = [_node("src/api.py", "function", "ep")]
        edges = [_edge("src/api.py", "src/models.py")]
        entries = compress_to_skeleton(nodes, edges)
        assert "models.py" in entries[0].imports

    def test_external_imports_excluded_from_entry(self) -> None:
        nodes = [_node("src/api.py", "function", "ep")]
        edges = [_edge("src/api.py", "sqlalchemy", external=True)]
        entries = compress_to_skeleton(nodes, edges)
        assert entries[0].imports == []

    def test_budget_enforcement_reduces_content(self) -> None:
        # Build many files to exceed budget
        all_nodes = []
        for i in range(20):
            path = f"src/module{i}.py"
            all_nodes.append(_node(path, "class", f"Class{i}", fields="x:int, y:str"))
            all_nodes += [_node(path, "method", f"m{j}", parent=f"Class{i}") for j in range(5)]
        entries = compress_to_skeleton(all_nodes, [], budget_tokens=200)
        total = sum(e.token_estimate for e in entries)
        assert total <= 210  # small tolerance for estimate rounding

    def test_token_estimate_populated(self) -> None:
        nodes = [_node("src/m.py", "class", "Foo", fields="x:int")]
        entries = compress_to_skeleton(nodes, [])
        assert entries[0].token_estimate > 0


class TestOmittedCounts:
    """classes_omitted/functions_omitted feed render.py's honest '+N more
    public symbols not shown' marker on the import hint — see #29. Must count
    only PUBLIC symbols cut by the file-level class/function caps: a cut
    private symbol was never part of the import line's completeness claim."""

    def test_functions_beyond_cap_are_counted(self) -> None:
        # 12 public top-level functions, cap is 10 -> 2 omitted.
        nodes = [_node("src/m.py", "function", f"fn{i:02d}") for i in range(12)]
        entries = compress_to_skeleton(nodes, [], budget_tokens=100_000)
        assert entries[0].functions_omitted == 2

    def test_classes_beyond_cap_are_counted(self) -> None:
        # 7 classes, cap is 5 -> 2 omitted.
        nodes = [_node("src/m.py", "class", f"C{i}") for i in range(7)]
        entries = compress_to_skeleton(nodes, [], budget_tokens=100_000)
        assert entries[0].classes_omitted == 2

    def test_cut_private_functions_are_not_counted(self) -> None:
        # 10 public (fills the cap exactly) + 3 private -> all 3 private ones
        # are what gets cut, and none of them were ever going into the import
        # hint, so the omitted count must be 0.
        nodes = [_node("src/m.py", "function", f"fn{i:02d}") for i in range(10)]
        nodes += [_node("src/m.py", "function", f"_priv{i}") for i in range(3)]
        entries = compress_to_skeleton(nodes, [], budget_tokens=100_000)
        assert entries[0].functions_omitted == 0

    def test_nothing_cut_means_zero_omitted(self) -> None:
        nodes = [_node("src/m.py", "function", "fn"),
                 _node("src/m.py", "class", "C")]
        entries = compress_to_skeleton(nodes, [], budget_tokens=100_000)
        assert entries[0].functions_omitted == 0
        assert entries[0].classes_omitted == 0

    def test_unlimited_caps_omit_nothing(self) -> None:
        from cognikernel.symbols.projection import Caps
        nodes = [_node("src/m.py", "function", f"fn{i:02d}") for i in range(30)]
        nodes += [_node("src/m.py", "class", f"C{i}") for i in range(10)]
        entries = compress_to_skeleton(
            nodes, [], budget_tokens=100_000, caps=Caps.unlimited())
        assert entries[0].functions_omitted == 0
        assert entries[0].classes_omitted == 0
        assert len(entries[0].functions) == 30
        assert len(entries[0].classes) == 10
