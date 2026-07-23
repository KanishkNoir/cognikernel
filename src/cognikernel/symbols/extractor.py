"""AST-based symbol extraction for the CogniKernel symbol graph.

Extracts classes, methods, functions, and import edges from source files.
Language-agnostic interface; Python via stdlib ast, TypeScript/JS via tree-sitter.
"""
from __future__ import annotations

import ast
import logging
import posixpath
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

_log = logging.getLogger("cognikernel.symbols")
_TS_DEP_WARNED = False  # one-shot: don't spam the log on every TS file


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class SymbolNode:
    path: str           # relative path within project, e.g. "src/models.py"
    node_type: str      # "class" | "function" | "method" | "import"
    name: str           # "Quote", "get_db", "__init__"
    parent_name: str    # class name for methods; "" for top-level
    signature: str      # "(text:str, author_id:int)" or ""
    return_type: str    # "Quote|None" or ""
    fields: str         # "id:int, text:str" (class nodes only)
    project_id: str
    updated_at: int     # epoch-ms


@dataclass
class SymbolEdge:
    project_id: str
    from_path: str      # "src/api/quotes.py"
    to_path: str        # "src/models.py" (local) or "sqlalchemy" (external)
    edge_type: str      # "imports" | "extends" | "implements"
    is_external: bool   # True → external library, excluded from skeleton rendering


@dataclass
class SymbolUpdate:
    """Side-channel output of symbol extraction — applied by store.apply_symbol_update."""
    project_id: str
    upsert_nodes: list[SymbolNode]
    upsert_edges: list[SymbolEdge]
    delete_paths: list[str]


# ── extractor protocol ────────────────────────────────────────────────────────

@runtime_checkable
class SymbolExtractor(Protocol):
    """Language-agnostic extraction interface. Implementations are stateless."""

    def extract(
        self,
        path: str,
        source: str,
        project_id: str,
        known_project_paths: frozenset[str],
    ) -> tuple[list[SymbolNode], list[SymbolEdge]]:
        """Parse source and return symbol nodes + import edges.

        Must return ([], []) on SyntaxError or any parse failure.
        Must never raise.
        """
        ...


# ── Python AST extractor ──────────────────────────────────────────────────────

class PythonASTExtractor:
    """Extracts symbols from Python source using the stdlib ast module."""

    def extract(
        self,
        path: str,
        source: str,
        project_id: str,
        known_project_paths: frozenset[str],
    ) -> tuple[list[SymbolNode], list[SymbolEdge]]:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return [], []
        except Exception:
            return [], []

        nodes: list[SymbolNode] = []
        edges: list[SymbolEdge] = []
        now = int(time.time() * 1000)

        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                fields_str = _extract_class_fields(stmt)
                nodes.append(SymbolNode(
                    path=path, node_type="class", name=stmt.name,
                    parent_name="", signature=_format_bases(stmt),
                    return_type="", fields=fields_str,
                    project_id=project_id, updated_at=now,
                ))
                for item in stmt.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == "__init__":
                            continue  # __init__ data lives in class.fields
                        nodes.append(SymbolNode(
                            path=path, node_type="method", name=item.name,
                            parent_name=stmt.name,
                            signature=_format_signature(item),
                            return_type=_format_return(item),
                            fields="", project_id=project_id, updated_at=now,
                        ))

            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                route_descriptor, route_model = _extract_fastapi_route_info(stmt)
                nodes.append(SymbolNode(
                    path=path, node_type="function", name=stmt.name,
                    parent_name="", signature=_format_signature(stmt),
                    return_type=route_model or _format_return(stmt),
                    fields=route_descriptor,
                    project_id=project_id, updated_at=now,
                ))

        edges.extend(_extract_import_edges(path, tree, project_id, known_project_paths))
        return nodes, edges


# ── TypeScript/JavaScript extractor ──────────────────────────────────────────

class TypeScriptExtractor:
    """Extracts symbols from TypeScript/JavaScript source using tree-sitter."""

    def __init__(self, language: str = "typescript") -> None:
        self._language = language

    def extract(
        self,
        path: str,
        source: str,
        project_id: str,
        known_project_paths: frozenset[str],
    ) -> tuple[list[SymbolNode], list[SymbolEdge]]:
        if not source.strip():
            return [], []
        # Fail-open in both branches (a symbol-graph gap must never break ingest),
        # but distinguish the causes so a silent empty graph is diagnosable
        # (audit P3): a missing parser dependency is a one-shot config warning;
        # a parse failure on a real file is a per-file warning.
        try:
            from tree_sitter_language_pack import get_parser  # type: ignore[import]
        except ImportError as exc:
            global _TS_DEP_WARNED
            if not _TS_DEP_WARNED:
                _log.warning(
                    "symbols.typescript_unavailable: tree-sitter-language-pack not "
                    "importable (%s) — TS/JS files yield no symbol graph. Install the "
                    "dependency; run 'cognikernel doctor' to confirm.", exc,
                )
                _TS_DEP_WARNED = True
            return [], []
        try:
            parser = get_parser(self._language)
            source_bytes = source.encode("utf-8")
            tree = parser.parse_bytes(source_bytes)
            root = tree.root_node()
        except Exception as exc:
            _log.warning("symbols.typescript_parse_failed: %s (%s)", path, exc)
            return [], []

        nodes: list[SymbolNode] = []
        edges: list[SymbolEdge] = []
        seen_edges: set[tuple[str, str]] = set()
        now = int(time.time() * 1000)

        def _get(node) -> str:
            return source_bytes[node.start_byte():node.end_byte()].decode("utf-8", errors="replace")

        for i in range(root.named_child_count()):
            child = root.named_child(i)
            kind = child.kind()

            if kind == "import_statement":
                edge = _ts_import_edge(child, path, project_id, known_project_paths, _get)
                if edge:
                    key = (edge.from_path, edge.to_path)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges.append(edge)

            elif kind == "export_statement":
                decl = _ts_unwrap_export(child)
                if decl is not None:
                    _ts_handle_decl(decl, path, project_id, _get, nodes, now)

            elif kind in ("class_declaration", "abstract_class_declaration"):
                nodes.extend(_ts_extract_class(child, path, project_id, _get, now))

            elif kind == "function_declaration":
                fn = _ts_extract_function(child, path, project_id, _get, now)
                if fn:
                    nodes.append(fn)

        return nodes, edges


# ── TypeScript AST helpers ────────────────────────────────────────────────────

def _ts_unwrap_export(export_node) -> object | None:
    """Return the inner declaration from an export_statement, or None."""
    for i in range(export_node.named_child_count()):
        child = export_node.named_child(i)
        if child.kind() in (
            "class_declaration", "abstract_class_declaration", "class",
            "function_declaration", "lexical_declaration",
        ):
            return child
    return None


def _ts_handle_decl(decl, path, project_id, _get, nodes, now) -> None:
    kind = decl.kind()
    if kind in ("class_declaration", "abstract_class_declaration", "class"):
        nodes.extend(_ts_extract_class(decl, path, project_id, _get, now))
    elif kind == "function_declaration":
        fn = _ts_extract_function(decl, path, project_id, _get, now)
        if fn:
            nodes.append(fn)


def _ts_extract_class(cls_node, path, project_id, _get, now) -> list[SymbolNode]:
    name_node = cls_node.child_by_field_name("name")
    if name_node is None:
        return []
    class_name = _get(name_node)

    signature = ""
    for i in range(cls_node.named_child_count()):
        child = cls_node.named_child(i)
        if child.kind() == "class_heritage":
            for j in range(child.named_child_count()):
                ext = child.named_child(j)
                if ext.kind() == "extends_clause" and ext.named_child_count() > 0:
                    signature = _get(ext.named_child(0))
                    break
            break

    body = cls_node.child_by_field_name("body")
    fields_str = _ts_extract_fields(body, _get) if body is not None else ""
    methods = _ts_extract_methods(body, class_name, path, project_id, _get, now) if body is not None else []

    return [
        SymbolNode(
            path=path, node_type="class", name=class_name,
            parent_name="", signature=signature,
            return_type="", fields=fields_str,
            project_id=project_id, updated_at=now,
        ),
        *methods,
    ]


def _ts_extract_fields(class_body, _get) -> str:
    fields: dict[str, str] = {}
    for i in range(class_body.named_child_count()):
        child = class_body.named_child(i)
        if child.kind() != "public_field_definition":
            continue
        name_node = child.child_by_field_name("name")
        if name_node is None:
            continue
        name = _get(name_node)
        type_node = child.child_by_field_name("type")
        type_str = ""
        if type_node is not None and type_node.named_child_count() > 0:
            type_str = _get(type_node.named_child(0))
        fields[name] = type_str
        if len(fields) >= 10:
            break
    if not fields:
        return ""
    return ", ".join(f"{n}:{t}" if t else n for n, t in fields.items())


def _ts_extract_methods(class_body, class_name, path, project_id, _get, now) -> list[SymbolNode]:
    methods = []
    for i in range(class_body.named_child_count()):
        child = class_body.named_child(i)
        if child.kind() != "method_definition":
            continue
        name_node = child.child_by_field_name("name")
        if name_node is None:
            continue
        method_name = _get(name_node)
        if method_name == "constructor":
            continue
        params_node = child.child_by_field_name("parameters")
        sig = _ts_format_params(params_node, _get) if params_node else "()"
        ret_node = child.child_by_field_name("return_type")
        ret = _get(ret_node.named_child(0)) if (ret_node and ret_node.named_child_count() > 0) else ""
        methods.append(SymbolNode(
            path=path, node_type="method", name=method_name,
            parent_name=class_name, signature=sig,
            return_type=ret, fields="",
            project_id=project_id, updated_at=now,
        ))
    return methods


def _ts_extract_function(fn_node, path, project_id, _get, now) -> SymbolNode | None:
    name_node = fn_node.child_by_field_name("name")
    if name_node is None:
        return None
    name = _get(name_node)
    params_node = fn_node.child_by_field_name("parameters")
    sig = _ts_format_params(params_node, _get) if params_node else "()"
    ret_node = fn_node.child_by_field_name("return_type")
    ret = _get(ret_node.named_child(0)) if (ret_node and ret_node.named_child_count() > 0) else ""
    return SymbolNode(
        path=path, node_type="function", name=name,
        parent_name="", signature=sig,
        return_type=ret, fields="",
        project_id=project_id, updated_at=now,
    )


def _ts_format_params(params_node, _get) -> str:
    parts = []
    for i in range(params_node.named_child_count()):
        p = params_node.named_child(i)
        if p.kind() not in ("required_parameter", "optional_parameter"):
            continue
        name_node = None
        type_ann = None
        for j in range(p.named_child_count()):
            child = p.named_child(j)
            if child.kind() == "identifier":
                name_node = child
            elif child.kind() == "type_annotation":
                type_ann = child
        if name_node is None:
            continue
        name_str = _get(name_node)
        if type_ann is not None and type_ann.named_child_count() > 0:
            parts.append(f"{name_str}:{_get(type_ann.named_child(0))}")
        else:
            parts.append(name_str)
    return f"({', '.join(parts)})"


def _ts_import_edge(import_node, path, project_id, known, _get) -> SymbolEdge | None:
    n = import_node.named_child_count()
    if n == 0:
        return None
    str_node = import_node.named_child(n - 1)
    if str_node.kind() != "string" or str_node.named_child_count() == 0:
        return None
    specifier = _get(str_node.named_child(0))
    to_path, is_ext = _ts_resolve_import(specifier, path, known)
    return SymbolEdge(
        project_id=project_id,
        from_path=path,
        to_path=to_path,
        edge_type="imports",
        is_external=is_ext,
    )


def _ts_resolve_import(specifier: str, from_path: str, known: frozenset[str]) -> tuple[str, bool]:
    if not specifier.startswith("."):
        return specifier, True
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(from_path), specifier))
    for ext in (".ts", ".tsx", ".js", ".jsx"):
        c = resolved + ext
        if c in known:
            return c, False
    if resolved in known:
        return resolved, False
    stem = posixpath.basename(resolved)
    for kp in known:
        if posixpath.splitext(posixpath.basename(kp))[0] == stem:
            return kp, False
    return specifier, True


# ── extractor registry + dispatch ─────────────────────────────────────────────

def typescript_support_status() -> tuple[bool, str]:
    """Probe whether TS/JS symbol extraction is actually available (audit P3).

    The extractor fails open to an empty graph, so a missing parser is otherwise
    invisible. `cognikernel doctor` calls this to surface the cause. Returns
    (available, human-readable detail).
    """
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore[import]
    except ImportError as exc:
        return False, f"tree-sitter-language-pack not importable ({exc})"
    try:
        parser = get_parser("typescript")
        parser.parse_bytes(b"const x = 1;").root_node()
        return True, "tree-sitter typescript parser OK"
    except Exception as exc:  # pragma: no cover - environment-dependent
        return False, f"parser init/parse failed ({exc})"


EXTRACTORS: dict[str, SymbolExtractor] = {
    ".py": PythonASTExtractor(),
    ".ts":  TypeScriptExtractor("typescript"),
    ".tsx": TypeScriptExtractor("tsx"),
    ".js":  TypeScriptExtractor("javascript"),
    ".jsx": TypeScriptExtractor("javascript"),
}


def extract_file(
    path: str,
    abs_path: str,
    project_id: str,
    known_project_paths: frozenset[str],
) -> tuple[list[SymbolNode], list[SymbolEdge]]:
    """Dispatch extraction by file extension. Returns ([], []) for unknown types."""
    suffix = Path(path).suffix.lower()
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        return [], []
    try:
        source = Path(abs_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], []
    return extractor.extract(path, source, project_id, known_project_paths)


def build_symbol_update(
    project_id: str,
    project_path: str,
    changed_files: list,
) -> SymbolUpdate:
    """Build a SymbolUpdate from a list of FileChange objects.

    For deleted files: schedule deletion, no parse.
    For added/modified/renamed files: parse and upsert.
    First session (no changed_files): walk entire project up to 500 files.
    """
    project_root = Path(project_path).resolve()
    known_paths = _discover_project_paths(project_root)
    known_frozenset = frozenset(known_paths.keys())

    upsert_nodes: list[SymbolNode] = []
    upsert_edges: list[SymbolEdge] = []
    delete_paths: list[str] = []

    if not changed_files:
        # First session / cold cache — walk entire project
        for rel, abs_p in known_paths.items():
            nodes, edges = extract_file(rel, abs_p, project_id, known_frozenset)
            upsert_nodes.extend(nodes)
            upsert_edges.extend(edges)
    else:
        for fc in changed_files:
            if fc.change_type == "deleted":
                delete_paths.append(fc.path)
            else:
                # For renames, delete the old path if tracked
                old_path = getattr(fc, "old_path", "")
                if old_path:
                    delete_paths.append(old_path)
                abs_p = str(project_root / fc.path)
                nodes, edges = extract_file(fc.path, abs_p, project_id, known_frozenset)
                upsert_nodes.extend(nodes)
                upsert_edges.extend(edges)

    return SymbolUpdate(
        project_id=project_id,
        upsert_nodes=upsert_nodes,
        upsert_edges=upsert_edges,
        delete_paths=delete_paths,
    )


# ── path discovery ────────────────────────────────────────────────────────────

_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build", ".eggs",
})
_MAX_FILES = 500


def _discover_project_paths(project_root: Path) -> dict[str, str]:
    """Walk project for supported source files, skip noise dirs. Returns {rel_path: abs_path}."""
    result: dict[str, str] = {}
    patterns = ("*.py", "*.ts", "*.tsx", "*.js", "*.jsx")
    for pattern in patterns:
        for abs_p in project_root.rglob(pattern):
            if any(part in _SKIP_DIRS for part in abs_p.parts):
                continue
            try:
                rel = str(abs_p.relative_to(project_root)).replace("\\", "/")
            except ValueError:
                continue
            result[rel] = str(abs_p)
            if len(result) >= _MAX_FILES:
                return result
    return result


# ── AST helpers ───────────────────────────────────────────────────────────────

def _format_bases(class_node: ast.ClassDef) -> str:
    """Return first base class as string, or ''."""
    if not class_node.bases:
        return ""
    try:
        return ast.unparse(class_node.bases[0])
    except Exception:
        return ""


def _format_signature(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Format argument list, stripping 'self'/'cls'. Returns '(a:T, b)' form."""
    try:
        args = func_node.args
        all_args = args.posonlyargs + args.args + args.kwonlyargs
        filtered = [a for a in all_args if a.arg not in ("self", "cls")]
        parts: list[str] = []
        for arg in filtered:
            if arg.annotation:
                try:
                    parts.append(f"{arg.arg}:{ast.unparse(arg.annotation)}")
                except Exception:
                    parts.append(arg.arg)
            else:
                parts.append(arg.arg)
        return f"({', '.join(parts)})"
    except Exception:
        return ""


def _format_return(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if func_node.returns is None:
        return ""
    try:
        return ast.unparse(func_node.returns)
    except Exception:
        return ""


def _extract_fastapi_route_info(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str]:
    """Return (route_descriptor, response_model) for FastAPI route decorators.

    route_descriptor: "GET /path" or ""
    response_model: "list[NoteResponse]" or ""
    Returns ("", "") for non-route functions.
    """
    _HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete"})
    for dec in func_node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        func = dec.func
        if not isinstance(func, ast.Attribute):
            continue
        method = func.attr.lower()
        if method not in _HTTP_METHODS:
            continue
        path = ""
        if dec.args:
            try:
                path = ast.literal_eval(dec.args[0])
            except Exception:
                try:
                    path = ast.unparse(dec.args[0])
                except Exception:
                    pass
        response_model = ""
        for kw in dec.keywords:
            if kw.arg == "response_model":
                try:
                    response_model = ast.unparse(kw.value)
                except Exception:
                    pass
                break
        if path:
            return f"{method.upper()} {path}", response_model
    return "", ""


def _extract_class_fields(class_node: ast.ClassDef) -> str:
    """Extract typed fields from class body and __init__. Returns 'id:int, text:str'."""
    fields: dict[str, str] = {}

    # Class-level annotated assignments: id: int = ...
    for stmt in class_node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            try:
                fields[stmt.target.id] = ast.unparse(stmt.annotation)
            except Exception:
                fields[stmt.target.id] = ""

    # self.x assignments in __init__
    for stmt in class_node.body:
        if (isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and stmt.name == "__init__"):
            for item in stmt.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if (isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                                and target.attr not in fields):
                            fields[target.attr] = ""
                elif isinstance(item, ast.AnnAssign):
                    if (isinstance(item.target, ast.Attribute)
                            and isinstance(item.target.value, ast.Name)
                            and item.target.value.id == "self"):
                        try:
                            fields[item.target.attr] = ast.unparse(item.annotation)
                        except Exception:
                            fields[item.target.attr] = ""

    if not fields:
        return ""
    parts = [f"{n}:{t}" if t else n for n, t in fields.items()]
    return ", ".join(parts[:10])  # cap at 10 fields


def _extract_import_edges(
    path: str,
    tree: ast.Module,
    project_id: str,
    known_project_paths: frozenset[str],
) -> list[SymbolEdge]:
    """Walk import statements and produce edges to local or external modules."""
    edges: list[SymbolEdge] = []
    seen: set[tuple[str, str]] = set()

    def _resolve(module_name: str) -> tuple[str, bool]:
        candidates = [
            module_name.replace(".", "/") + ".py",
            "src/" + module_name.replace(".", "/") + ".py",
        ]
        for c in candidates:
            if c in known_project_paths:
                return c, False
        stem = module_name.split(".")[-1]
        for kp in known_project_paths:
            if kp.endswith(f"/{stem}.py") or kp == f"{stem}.py":
                return kp, False
        return module_name, True

    for stmt in tree.body:
        module_names: list[str] = []
        if isinstance(stmt, ast.Import):
            module_names = [alias.name for alias in stmt.names]
        elif isinstance(stmt, ast.ImportFrom) and stmt.module:
            module_names = [stmt.module]

        for module_name in module_names:
            to_path, is_ext = _resolve(module_name)
            key = (path, to_path)
            if key not in seen:
                seen.add(key)
                edges.append(SymbolEdge(
                    project_id=project_id,
                    from_path=path,
                    to_path=to_path,
                    edge_type="imports",
                    is_external=is_ext,
                ))

    return edges
