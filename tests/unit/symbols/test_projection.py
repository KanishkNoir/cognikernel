"""Unit tests for symbol graph compression (projection)."""
import pytest
from memlora.symbols.extractor import SymbolNode, SymbolEdge
from memlora.symbols.projection import compress_to_skeleton, SkeletonEntry


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
