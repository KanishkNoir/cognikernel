# Multi-language symbol extraction — Go, Rust, Java

**Status:** design approved, not yet implemented
**Date:** 2026-08-02
**Follows:** the injection-correctness cycle, which fixed *which files* reach the
symbol graph. This cycle extends *which languages* can populate it at all.

---

## 0. Why

CogniKernel's skeleton is AST-derived, and AST extraction currently covers
Python (stdlib `ast`) and TypeScript/JavaScript (tree-sitter). Every other
language yields **no symbols at all** — a Go or Rust project gets an empty
Codebase skeleton section, which is the largest section of the injected block.

`tree-sitter-language-pack` is **already a hard dependency** (`pyproject.toml`)
and ships 306 grammars, including `go`, `rust`, `java`, `ruby`, `php`,
`kotlin`, `swift`, `cpp`, `c`, `scala`. So the cost here is extractor logic,
not install size.

### What this is not

Adding a language is **not** a registry line. `TypeScriptExtractor` takes a
`language` parameter, but that only selects the parser; every node kind it
walks (`class_declaration`, `import_statement`, `class_heritage`,
`public_field_definition`) is TypeScript-specific. Each language needs its own
name extraction, signature formatting, parent linkage, and import resolution.

### Research basis

Node kinds and field names below were verified against the **installed**
grammars, not documentation, and the resulting mapping was then run over **real
production source**:

| File | Size | Symbols yielded | Unhandled top-level kinds |
|---|---|---|---|
| `golang/go` `src/net/http/request.go` | 50 KB | 63 | `const_declaration` ×2, `var_declaration` ×7, comments |
| `BurntSushi/ripgrep` `searcher/mod.rs` | 40 KB | 56 | `attribute_item` ×8, comments |
| `google/guava` `ImmutableList.java` | 30 KB | 50 | comments |

Import-resolution rules come from the Rust Reference (`items/modules.html`)
and the Go module reference (`go.dev/ref/mod`).

**One approach was investigated and ruled out.** Declarative tree-sitter
`.scm` query files — the idiomatic way most tooling does this — are
**unusable here**: `tree_sitter_language_pack.get_language()` returns a
vendored `Language` type that `tree_sitter.Query.__new__` rejects
(`argument 1 must be tree_sitter.Language, not builtins.Language`). Verified
directly. Extraction must therefore be hand-written node walking.

---

## 1. Architecture

`symbols/extractor.py` is ~700 lines; three more languages inline would roughly
double it. Split into a package:

```
symbols/
  extractor.py            dataclasses, EXTRACTORS registry, extract_file
                          dispatch, discovery/scoping helpers
  extractors/
    __init__.py
    shared.py             byte-slice reader, named-child iteration,
                          signature joining, generic-type unwrapping
    resolvers.py          manifest-aware import resolution per language
    python.py             PythonASTExtractor   (relocated, unchanged)
    typescript.py         TypeScriptExtractor  (relocated, unchanged)
    go.py                 GoExtractor
    rust.py               RustExtractor
    java.py               JavaExtractor
```

Relocating Python and TypeScript is a **pure move with no behaviour change**;
the existing 120 tests under `tests/unit/symbols/` are the guard.

Each extractor satisfies the existing `SymbolExtractor` protocol:

```python
def extract(
    self, path: str, source: str, project_id: str,
    known_project_paths: frozenset[str],
) -> tuple[list[SymbolNode], list[SymbolEdge]]
```

---

## 2. Mapping onto the existing four node types

The schema constrains `node_type` to `class | function | method | import`
(`002_symbol_graph.sql`). **No migration.** Every construct maps cleanly, and
keeping the renderer untouched is worth more than typing fidelity.

| Language | Construct | Stored as | Notes |
|---|---|---|---|
| Go | `type_spec` whose `type` is `struct_type` | `class` | fields from `struct_type` |
| Go | `type_spec` whose `type` is `interface_type` | `class`, signature `"interface"` | |
| Go | `function_declaration` | `function` | |
| Go | `method_declaration` | `method` | `parent_name` from `receiver` |
| Rust | `struct_item`, `trait_item`, `enum_item` | `class` | |
| Rust | `function_item` at top level or in `mod_item` | `function` | |
| Rust | `function_item` inside `impl_item` | `method` | `parent_name` = impl target |
| Java | `class_declaration`, `interface_declaration`, `enum_declaration`, `record_declaration` | `class` | |
| Java | `method_declaration` in a class body | `method` | `parent_name` = enclosing type |

### 2.1 Go — descend, never read the top level

**`type_declaration` carries no name.** Names live in nested `type_spec`
children, and a grouped block holds several:

```
type ( User struct{…}; Store interface{…} )
└─ type_declaration                    name=(none)
     ├─ type_spec  name=User   type.kind=struct_type
     └─ type_spec  name=Store  type.kind=interface_type
```

Emitting one symbol per `type_declaration` would produce a single unnamed node
where real Go code expects several named ones. Grouped declarations are
idiomatic, so this is the common case, not an edge case.

- struct vs interface: `type_spec.child_by_field_name("type").kind()`
- struct fields: descend into `struct_type` → `field_declaration_list` →
  `field_declaration` → `name`
- generics: `type_spec` and `function_declaration` expose `type_parameters`
- methods are **top-level** with a `receiver` field; `parent_name` is the
  receiver's type name with `*` and parens stripped

**Package-level `const`/`var` are deliberately skipped.** They appear 9 times
in `request.go` and include genuine exported API (`ErrMissingFile`), but they
do not fit the four node types. Python and TypeScript extractors likewise emit
no module-level constants, so skipping keeps the languages consistent and the
node types honest. Revisit only on user demand.

### 2.2 Rust — unwrap the impl target

`impl_item` has **no `name`**, but does expose `type` and `trait` fields. The
target must be unwrapped through generics and references to get a base name
usable as `parent_name`. All four shapes verified:

| Source | `type.kind` | base name |
|---|---|---|
| `impl<T: Clone> Store for User<T>` | `generic_type` | `User` |
| `impl Pool` | `type_identifier` | `Pool` |
| `impl<'a> Trait for &'a Foo` | `reference_type` | `Foo` |
| `impl Store for Vec<Item>` | `generic_type` | `Vec` |

Unwrap loop: while `kind()` is `generic_type` or `reference_type`, follow
`child_by_field_name("type")`.

- `mod foo { … }` (has `body`) → recurse inline
- `mod foo;` (no `body`) → file-backed; resolve per §3
- `#[derive(...)]` parses as a **sibling** `attribute_item`, not a wrapper, so
  it does not interfere with finding the item it decorates
- `type_item` (aliases) is skipped, consistent with skipping Go's consts

### 2.3 Java — recursion is mandatory

Guava's `ImmutableList.java` contains exactly **one** top-level
`class_declaration` and yields 50 symbols. All real content is nested inside
`class_body`, including further `class_declaration` nodes for inner classes.
A top-level-only walker would return 1 symbol from a 30 KB file.

- annotations parse as a `modifiers` **child** of the declaration, not a
  wrapper — discovery is unaffected
- `field_declaration` exposes `declarator`, not `name`; descend for the
  identifier
- `record_declaration` carries its components in `parameters`
- nesting is recursive; `parent_name` is the immediately enclosing type.
  Recursion depth is capped at 3 to bound pathological files.

---

## 3. Import resolution

Import edges feed the PageRank centrality that ranks the skeleton. Without
local resolution every import is external, the graph has no internal edges, and
ranking degrades to arbitrary order. Full resolution is therefore in scope.

`resolvers.py` exposes one function per language, each pure over
`(specifier, from_path, known_paths, manifest)`.

### Go

From `go.dev/ref/mod`: *"A package path is the module path joined with the
subdirectory containing the package, relative to the module root."*

1. Read the `module` line from `go.mod` at the project root.
2. An import starting with that module path is **first-party**: strip the
   prefix, and the remainder is a directory relative to the module root. Emit
   an edge to each known `.go` file in that directory.
3. **Standard library** is identified by *no dot in the first path element*
   (`fmt`, `net/http`) — cheap and reliable.
4. Anything else is external.
5. `vendor/` is already excluded by discovery's skip list.

### Rust

From the Rust Reference: *"the path to the file mirrors the logical module
path… Ancestor module path components are directories."* And crucially:
*"It is not allowed to have both `util.rs` and `util/mod.rs`."* — so probing is
unambiguous.

1. Crate root is `src/lib.rs` or `src/main.rs`.
2. `crate::util::config` → probe `src/util/config.rs`, then
   `src/util/config/mod.rs`.
3. `super::` resolves relative to the parent module's directory; `self::`
   relative to the current one.
4. A leading segment that is not `crate`/`super`/`self` and not a known local
   module is an external crate.
5. `#[path = "…"]` overrides are **not supported**; such modules resolve as
   external. Rare, and a wrong local edge is worse than a missing one.

### Java

1. Detect the source root by matching the `package` declaration against the
   file's own directory suffix — this derives the root instead of assuming
   `src/main/java`, so Gradle, Maven, and flat layouts all work.
2. `com.example.db.Pool` → `<src-root>/com/example/db/Pool.java`.
3. Wildcard `import com.example.*` → edges to every known file in that package
   directory.
4. `import static a.b.C.method` → strip the trailing member, resolve `a.b.C`.
5. `java.*` / `javax.*` and unresolvable packages are external.

**Fallback:** when the manifest is missing or unparseable, fall back to the
heuristic stem match TypeScript already uses (last path segment against known
file stems). Never raise.

---

## 4. Discovery and the budget

Add `*.go`, `*.rs`, `*.java` to the discovery pattern tuple. `target/`,
`vendor/`, and `build/` are already in `_SKIP_DIRS`.

### 4.1 Depth must be normalised per language

`_src_rank` currently orders by `(not-first-party, depth-from-project-root)`.
Raw depth is only meaningful **within** one language's conventions. Java's
canonical `src/main/java/com/example/db/Pool.java` is six segments deep; a Go
package sits two from the root. In a polyglot repo, every Java file would rank
last and lose the 500-file budget — the vendored-skeleton failure in a new
costume: the right files present, but outranked.

**Fix: rank by depth relative to the median depth of that file's own
language.**

```
rank_key(path) = (not_first_party, depth(path) - median_depth(language_of(path)))
```

Self-calibrating: no source-root table, no per-language conventions, and it
behaves correctly on monorepos and non-standard layouts because every file is
compared against its own peers. A Java file at depth 6 in an all-Java repo
scores like a Go file at depth 2 in an all-Go repo.

Medians are computed once per discovery run over the candidate list, before
truncation to `_MAX_FILES`.

**This is a no-op for single-language projects**, which matters because every
existing user is one. When all candidates share a language, subtracting a
common median shifts every key by the same constant, so the relative ordering —
and therefore which files survive the budget — is unchanged. The new behaviour
only engages where languages actually compete, which is exactly the case that
is broken today. A single-file language is likewise well-behaved: its median is
its own depth, giving a normalised score of 0.

`_MAX_FILES` stays 500. Measured parse speed is <5 ms/file, so three more
languages do not change the cost model.

---

## 5. Error handling

Unchanged contract: **a symbol-graph gap must never break ingest.**

- Missing `tree-sitter-language-pack` → one warning per language (module-level
  flag, mirroring `_TS_DEP_WARNED`), return `([], [])`.
- Parse failure on a file → per-file warning, return `([], [])`.
- Missing/unparseable manifest → heuristic fallback (§3), no warning.
- Unrecognised constructs → skipped silently; they are not errors.
- `doctor` gains a per-language support line, generalising
  `typescript_support_status()` into `language_support_status()`.

---

## 6. Testing

- **Relocation guard:** the existing 120 `tests/unit/symbols/` tests must pass
  unchanged after Python/TypeScript move into `extractors/`.
- **Per-language extraction tests** using the constructs verified in §2:
  grouped Go `type (…)`, Go methods with pointer and value receivers, all four
  Rust impl shapes, Rust `mod` inline vs file-backed, Java nested classes,
  records, enums, annotated members.
- **Import-resolution tests** building a real project tree under `tmp_path`
  with a genuine `go.mod` / `Cargo.toml` / package directories, asserting
  first-party edges resolve and external ones are marked external.
- **Depth-normalisation tests**: (a) a mixed Go+Java tree where the Java files
  are deeper — assert Java files survive the budget; (b) a single-language tree
  — assert discovery output is **byte-identical** to the pre-change behaviour,
  pinning the no-op property that makes this safe for existing users.
- **Degradation tests** per language: syntactically invalid source yields
  `([], [])` and does not raise.
- **Real-file smoke test**: the three production files used in §0 are *not*
  committed (licence/size); instead the tests assert symbol counts on trimmed
  excerpts checked into `tests/fixtures/symbols/`.

---

## 7. Decisions recorded

1. **Go, Rust, Java** this cycle. Ruby/PHP/C#/Kotlin deferred — breadth without
   depth is the failure mode this project has already been bitten by.
2. **No schema migration**; the four node types absorb every construct.
3. **Full local import resolution**, because PageRank is worthless without
   internal edges.
4. **Extractors package split**, one module per language.
5. **Go package-level const/var skipped**, consistent with Python and
   TypeScript.
6. **Depth normalised per language**, rejecting both the hardcoded
   source-root table and raising `_MAX_FILES`.
7. **tree-sitter queries rejected** — the language pack's `Language` type is
   incompatible with `tree_sitter.Query`, verified directly.

## 8. Open questions

- Whether Rust `mod foo;` chains should be followed transitively to build
  module→file edges, or only direct `use` statements produce edges. Start with
  `use` only; revisit if centrality looks wrong on a real Rust project.
- Whether Java wildcard imports should emit an edge per file in the package
  (accurate but edge-heavy) or a single edge to the package's most central
  file. Start with per-file; measure edge counts on a real Java project.
