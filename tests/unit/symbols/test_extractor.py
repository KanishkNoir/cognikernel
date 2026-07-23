"""Unit tests for the Python AST symbol extractor."""
import pytest
from cognikernel.symbols.extractor import (
    PythonASTExtractor,
    SymbolNode,
    SymbolEdge,
    build_symbol_update,
    extract_file,
    EXTRACTORS,
)

_EXTRACTOR = PythonASTExtractor()
_PID = "test-project"
_KNOWN: frozenset[str] = frozenset({"src/models.py", "src/database.py", "src/api/quotes.py"})


def _extract(source: str, path: str = "test.py") -> tuple[list[SymbolNode], list[SymbolEdge]]:
    return _EXTRACTOR.extract(path, source, _PID, _KNOWN)


class TestClassExtraction:
    def test_simple_class(self) -> None:
        nodes, _ = _extract("class Foo:\n    pass\n")
        assert any(n.node_type == "class" and n.name == "Foo" for n in nodes)

    def test_class_with_base(self) -> None:
        nodes, _ = _extract("class Quote(Base):\n    pass\n")
        class_node = next(n for n in nodes if n.node_type == "class")
        assert class_node.signature == "Base"

    def test_annotated_class_fields(self) -> None:
        src = "class Quote:\n    id: int\n    text: str\n"
        nodes, _ = _extract(src)
        class_node = next(n for n in nodes if n.node_type == "class")
        assert "id:int" in class_node.fields
        assert "text:str" in class_node.fields

    def test_init_self_assignment_captured(self) -> None:
        src = "class Foo:\n    def __init__(self):\n        self.x = 1\n"
        nodes, _ = _extract(src)
        class_node = next(n for n in nodes if n.node_type == "class")
        assert "x" in class_node.fields

    def test_init_annotated_self_field(self) -> None:
        src = "class Foo:\n    def __init__(self):\n        self.y: str = 'hello'\n"
        nodes, _ = _extract(src)
        class_node = next(n for n in nodes if n.node_type == "class")
        assert "y:str" in class_node.fields

    def test_init_method_not_in_method_list(self) -> None:
        src = "class Foo:\n    def __init__(self):\n        pass\n"
        nodes, _ = _extract(src)
        method_nodes = [n for n in nodes if n.node_type == "method"]
        assert not any(m.name == "__init__" for m in method_nodes)


class TestMethodExtraction:
    def test_methods_extracted_with_parent(self) -> None:
        src = "class Foo:\n    def bar(self, x: int) -> str:\n        pass\n"
        nodes, _ = _extract(src)
        method = next(n for n in nodes if n.node_type == "method")
        assert method.name == "bar"
        assert method.parent_name == "Foo"
        assert "x:int" in method.signature
        assert method.return_type == "str"

    def test_multiple_methods(self) -> None:
        src = "class Foo:\n    def a(self): pass\n    def b(self): pass\n"
        nodes, _ = _extract(src)
        methods = [n for n in nodes if n.node_type == "method"]
        assert {m.name for m in methods} == {"a", "b"}

    def test_self_stripped_from_signature(self) -> None:
        src = "class Foo:\n    def do(self, x): pass\n"
        nodes, _ = _extract(src)
        method = next(n for n in nodes if n.node_type == "method")
        assert "self" not in method.signature
        assert "x" in method.signature


class TestFunctionExtraction:
    def test_top_level_function(self) -> None:
        src = "def get_db() -> Session:\n    pass\n"
        nodes, _ = _extract(src)
        assert any(n.node_type == "function" and n.name == "get_db" for n in nodes)

    def test_function_return_type(self) -> None:
        src = "def get_db() -> Session:\n    pass\n"
        nodes, _ = _extract(src)
        fn = next(n for n in nodes if n.node_type == "function")
        assert fn.return_type == "Session"

    def test_async_function(self) -> None:
        src = "async def fetch() -> None:\n    pass\n"
        nodes, _ = _extract(src)
        assert any(n.node_type == "function" and n.name == "fetch" for n in nodes)


class TestImportEdges:
    def test_local_import_resolved(self) -> None:
        src = "from models import Quote\n"
        _, edges = _extract(src, path="api/quotes.py")
        local = [e for e in edges if not e.is_external]
        assert any("models.py" in e.to_path for e in local)

    def test_external_import_flagged(self) -> None:
        src = "import sqlalchemy\n"
        _, edges = _extract(src)
        assert any(e.is_external and "sqlalchemy" in e.to_path for e in edges)

    def test_from_import(self) -> None:
        src = "from database import get_db\n"
        _, edges = _extract(src, path="api/quotes.py")
        assert any("database.py" in e.to_path and not e.is_external for e in edges)

    def test_no_duplicate_edges(self) -> None:
        src = "import sqlalchemy\nimport sqlalchemy\n"
        _, edges = _extract(src)
        to_paths = [e.to_path for e in edges]
        assert len(to_paths) == len(set(to_paths))


class TestGracefulDegradation:
    def test_syntax_error_returns_empty(self) -> None:
        src = "def broken(\n"
        nodes, edges = _extract(src)
        assert nodes == []
        assert edges == []

    def test_empty_source_returns_empty(self) -> None:
        nodes, edges = _extract("")
        assert nodes == []
        assert edges == []

    def test_unknown_extension_skipped(self) -> None:
        assert EXTRACTORS.get(".rb") is None
        nodes, edges = extract_file("script.rb", "/fake/script.rb", _PID, frozenset())
        assert nodes == []
        assert edges == []
