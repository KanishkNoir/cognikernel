"""Unit tests for the TypeScript/JavaScript symbol extractor."""
from __future__ import annotations

import logging
import sys

import pytest
from memlora.symbols import extractor as _extractor_mod
from memlora.symbols.extractor import (
    TypeScriptExtractor,
    SymbolNode,
    SymbolEdge,
    EXTRACTORS,
)

_EXTRACTOR = TypeScriptExtractor()
_PID = "test-project"
_KNOWN: frozenset[str] = frozenset({"src/models.ts", "src/database.ts", "src/api/quotes.ts"})


def _extract(source: str, path: str = "test.ts") -> tuple[list[SymbolNode], list[SymbolEdge]]:
    return _EXTRACTOR.extract(path, source, _PID, _KNOWN)


class TestTSClassExtraction:
    def test_simple_class(self) -> None:
        nodes, _ = _extract("class Foo {}")
        assert any(n.node_type == "class" and n.name == "Foo" for n in nodes)

    def test_exported_class(self) -> None:
        nodes, _ = _extract("export class Bar {}")
        assert any(n.node_type == "class" and n.name == "Bar" for n in nodes)

    def test_class_with_extends(self) -> None:
        nodes, _ = _extract("class Quote extends Base {}")
        class_node = next(n for n in nodes if n.node_type == "class")
        assert "Base" in class_node.signature

    def test_typed_class_fields(self) -> None:
        src = "class Quote { id: number; text: string; }"
        nodes, _ = _extract(src)
        class_node = next(n for n in nodes if n.node_type == "class")
        assert "id" in class_node.fields
        assert "number" in class_node.fields

    def test_constructor_not_in_methods(self) -> None:
        src = "class Foo { constructor(private db: Database) {} }"
        nodes, _ = _extract(src)
        method_nodes = [n for n in nodes if n.node_type == "method"]
        assert not any(m.name == "constructor" for m in method_nodes)

    def test_abstract_class(self) -> None:
        nodes, _ = _extract("abstract class Repo { abstract find(): void; }")
        assert any(n.node_type == "class" and n.name == "Repo" for n in nodes)


class TestTypeScriptDiagnostics:
    """Audit P3: the extractor still fails open to an empty graph, but the cause
    (missing parser dep vs parse failure) is now logged + doctor-probeable instead
    of silently swallowed."""

    def test_support_status_available_in_this_env(self) -> None:
        ok, detail = _extractor_mod.typescript_support_status()
        assert ok is True
        assert "OK" in detail

    def test_missing_dependency_fails_open_and_warns_once(
        self, monkeypatch, caplog
    ) -> None:
        # Setting the module to None makes `from ... import get_parser` raise
        # ImportError — i.e. the dependency is effectively absent.
        monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
        monkeypatch.setattr(_extractor_mod, "_TS_DEP_WARNED", False)
        with caplog.at_level(logging.WARNING, logger="memlora.symbols"):
            r1 = _EXTRACTOR.extract("a.ts", "class Foo {}", _PID, _KNOWN)
            r2 = _EXTRACTOR.extract("b.ts", "class Bar {}", _PID, _KNOWN)
        assert r1 == ([], [])
        assert r2 == ([], [])
        unavailable = [
            r for r in caplog.records if "typescript_unavailable" in r.getMessage()
        ]
        assert len(unavailable) == 1  # one-shot config warning, not per-file spam

    def test_parse_failure_fails_open_and_warns(self, monkeypatch, caplog) -> None:
        import tree_sitter_language_pack as tslp

        def _boom(_lang):
            raise RuntimeError("parser blew up")

        monkeypatch.setattr(tslp, "get_parser", _boom)
        with caplog.at_level(logging.WARNING, logger="memlora.symbols"):
            result = _EXTRACTOR.extract("c.ts", "class Foo {}", _PID, _KNOWN)
        assert result == ([], [])
        assert any(
            "typescript_parse_failed" in r.getMessage() for r in caplog.records
        )


class TestTSMethodExtraction:
    def test_method_extracted_with_parent(self) -> None:
        src = "class Foo { getQuote(id: number): string { return ''; } }"
        nodes, _ = _extract(src)
        method = next(n for n in nodes if n.node_type == "method")
        assert method.name == "getQuote"
        assert method.parent_name == "Foo"

    def test_return_type_captured(self) -> None:
        src = "class Svc { fetch(): Promise<string[]> { return []; } }"
        nodes, _ = _extract(src)
        method = next(n for n in nodes if n.node_type == "method")
        assert "Promise" in method.return_type

    def test_async_method_extracted(self) -> None:
        src = "class Svc { async fetch(): Promise<void> {} }"
        nodes, _ = _extract(src)
        assert any(n.node_type == "method" and n.name == "fetch" for n in nodes)

    def test_multiple_methods(self) -> None:
        src = "class Foo { a() {} b() {} c() {} }"
        nodes, _ = _extract(src)
        methods = [n for n in nodes if n.node_type == "method"]
        assert {m.name for m in methods} == {"a", "b", "c"}

    def test_method_signature_includes_params(self) -> None:
        src = "class Foo { greet(name: string, times: number): void {} }"
        nodes, _ = _extract(src)
        method = next(n for n in nodes if n.node_type == "method")
        assert "name" in method.signature
        assert "times" in method.signature


class TestTSFunctionExtraction:
    def test_top_level_function(self) -> None:
        src = "function getDb(): Database { return null; }"
        nodes, _ = _extract(src)
        assert any(n.node_type == "function" and n.name == "getDb" for n in nodes)

    def test_exported_function(self) -> None:
        src = "export function helper(x: string): boolean { return true; }"
        nodes, _ = _extract(src)
        assert any(n.node_type == "function" and n.name == "helper" for n in nodes)

    def test_function_return_type(self) -> None:
        src = "function getDb(): Session { return null; }"
        nodes, _ = _extract(src)
        fn = next(n for n in nodes if n.node_type == "function")
        assert fn.return_type == "Session"

    def test_async_function(self) -> None:
        src = "async function loadData(): Promise<void> {}"
        nodes, _ = _extract(src)
        assert any(n.node_type == "function" and n.name == "loadData" for n in nodes)


class TestTSImportEdges:
    def test_relative_import_is_local(self) -> None:
        src = "import { Quote } from './models';"
        _, edges = _extract(src, path="src/api/quotes.ts")
        local = [e for e in edges if not e.is_external]
        assert any("models.ts" in e.to_path for e in local)

    def test_bare_import_is_external(self) -> None:
        src = "import express from 'express';"
        _, edges = _extract(src)
        assert any(e.is_external and "express" in e.to_path for e in edges)

    def test_scoped_package_is_external(self) -> None:
        src = "import { useState } from 'react';"
        _, edges = _extract(src)
        assert any(e.is_external for e in edges)

    def test_no_duplicate_edges(self) -> None:
        src = "import { a } from 'express'; import { b } from 'express';"
        _, edges = _extract(src)
        to_paths = [e.to_path for e in edges]
        assert len(to_paths) == len(set(to_paths))

    def test_from_path_set_correctly(self) -> None:
        src = "import { x } from 'pkg';"
        _, edges = _extract(src, path="src/api/quotes.ts")
        assert all(e.from_path == "src/api/quotes.ts" for e in edges)


class TestTSGracefulDegradation:
    def test_empty_source_returns_empty(self) -> None:
        nodes, edges = _extract("")
        assert nodes == []
        assert edges == []

    def test_no_crash_on_malformed_source(self) -> None:
        nodes, edges = _extract("class { broken ===")
        # tree-sitter recovers; just must not raise

    def test_whitespace_only_returns_empty(self) -> None:
        nodes, edges = _extract("   \n  \t  ")
        assert nodes == []
        assert edges == []


class TestTSRegistry:
    def test_ts_registered(self) -> None:
        assert ".ts" in EXTRACTORS

    def test_tsx_registered(self) -> None:
        assert ".tsx" in EXTRACTORS

    def test_js_registered(self) -> None:
        assert ".js" in EXTRACTORS

    def test_jsx_registered(self) -> None:
        assert ".jsx" in EXTRACTORS

    def test_ts_extractor_is_typescript_extractor(self) -> None:
        assert isinstance(EXTRACTORS[".ts"], TypeScriptExtractor)
