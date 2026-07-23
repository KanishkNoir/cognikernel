"""Unit tests for symbol skeleton rendering."""
from cognikernel.symbols.projection import SkeletonEntry, SkeletonClass, SkeletonMethod
from cognikernel.symbols.render import render_skeleton_section, _render_method


def _entry(path: str = "src/models.py", imports=None, classes=None, functions=None) -> SkeletonEntry:
    return SkeletonEntry(
        path=path,
        imports=imports or [],
        classes=classes or [],
        functions=functions or [],
    )


def _cls(name: str, bases: str = "", fields: str = "", methods=None) -> SkeletonClass:
    return SkeletonClass(name=name, bases=bases, fields=fields, methods=methods or [])


def _method(name: str, sig: str = "()", ret: str = "") -> SkeletonMethod:
    return SkeletonMethod(name=name, signature=sig, return_type=ret)


class TestRenderSkeletonSection:
    def test_empty_returns_empty_string(self) -> None:
        assert render_skeleton_section([]) == ""

    def test_section_header_present(self) -> None:
        entry = _entry()
        out = render_skeleton_section([entry])
        assert "### Codebase skeleton" in out

    def test_file_path_in_output(self) -> None:
        entry = _entry("src/models.py")
        out = render_skeleton_section([entry])
        assert "src/models.py" in out

    def test_imports_arrow_format(self) -> None:
        entry = _entry("src/api.py", imports=["models.py", "database.py"])
        out = render_skeleton_section([entry])
        assert "src/api.py → models.py, database.py" in out

    def test_no_arrow_when_no_imports(self) -> None:
        entry = _entry("src/models.py", imports=[])
        out = render_skeleton_section([entry])
        assert "→" not in out

    def test_class_with_bases_format(self) -> None:
        cls = _cls("Quote", bases="Base", fields="id:int, text:str")
        entry = _entry(classes=[cls])
        out = render_skeleton_section([entry])
        assert "Quote(Base): id:int, text:str" in out

    def test_class_without_bases(self) -> None:
        cls = _cls("Foo", bases="", fields="x:int")
        entry = _entry(classes=[cls])
        out = render_skeleton_section([entry])
        assert "Foo: x:int" in out

    def test_class_indent_two_spaces(self) -> None:
        cls = _cls("Quote", bases="Base")
        entry = _entry(classes=[cls])
        out = render_skeleton_section([entry])
        assert "\n  Quote" in out

    def test_method_indent_four_spaces(self) -> None:
        cls = _cls("Quote", methods=[_method("create", "(text:str)", "Quote")])
        entry = _entry(classes=[cls])
        out = render_skeleton_section([entry])
        assert "\n    .create" in out

    def test_methods_joined_on_one_line(self) -> None:
        methods = [_method("a"), _method("b")]
        cls = _cls("Foo", methods=methods)
        entry = _entry(classes=[cls])
        out = render_skeleton_section([entry])
        assert ".a() | .b()" in out

    def test_method_return_type_arrow(self) -> None:
        m = _method("get", "(id:int)", "Quote|None")
        assert "→Quote|None" in _render_method(m)

    def test_method_no_return_type(self) -> None:
        m = _method("do", "(x)")
        assert "→" not in _render_method(m)

    def test_top_level_function_two_space_indent(self) -> None:
        fn = _method("get_db", "()", "Session")
        entry = _entry(functions=[fn])
        out = render_skeleton_section([entry])
        assert "\n  .get_db" in out

    def test_multiple_files_sorted_by_path(self) -> None:
        e1 = _entry("src/z.py")
        e2 = _entry("src/a.py")
        out = render_skeleton_section([e1, e2])
        assert out.index("src/a.py") < out.index("src/z.py")
