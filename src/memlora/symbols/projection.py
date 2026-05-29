"""Compress symbol graph nodes + edges into token-efficient SkeletonEntry objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlora.symbols.extractor import SymbolEdge, SymbolNode

_SKELETON_TOKEN_BUDGET = 800   # default; callers pass config.skeleton_budget
_MAX_CLASSES_PER_FILE = 5
_MAX_METHODS_PER_CLASS = 5
_MAX_FUNCTIONS_PER_FILE = 10
_MAX_IMPORTS_PER_FILE = 8


@dataclass
class SkeletonMethod:
    name: str
    signature: str
    return_type: str
    route_info: str = ""   # "GET /path" for FastAPI routes; "" otherwise


@dataclass
class SkeletonClass:
    name: str
    bases: str          # first base class name, e.g. "Base"
    fields: str         # "id:int, text:str"
    methods: list[SkeletonMethod] = field(default_factory=list)


@dataclass
class SkeletonEntry:
    path: str
    imports: list[str]              # local file basenames only
    classes: list[SkeletonClass] = field(default_factory=list)
    functions: list[SkeletonMethod] = field(default_factory=list)
    token_estimate: int = 0


def compress_to_skeleton(
    nodes: list["SymbolNode"],
    edges: list["SymbolEdge"],
    budget_tokens: int = _SKELETON_TOKEN_BUDGET,
    hot_paths: frozenset[str] | None = None,
) -> list[SkeletonEntry]:
    """Compress symbol graph into SkeletonEntry list fitting within budget_tokens.

    hot_paths: set of recently-active file paths that should be prioritised
               over lower-activity files when the budget forces drops.
    """
    if not nodes and not edges:
        return []

    _hot = hot_paths or frozenset()

    # Build path → nodes lookup
    by_path: dict[str, list["SymbolNode"]] = {}
    for node in nodes:
        by_path.setdefault(node.path, []).append(node)

    # Build path → local import targets lookup, plus the import graph used for
    # PageRank centrality (transitive importance, not just raw in-degree).
    by_from: dict[str, list[str]] = {}
    import_graph: dict[str, list[str]] = {}
    for edge in edges:
        if edge.is_external:
            continue
        by_from.setdefault(edge.from_path, [])
        basename = edge.to_path.rsplit("/", 1)[-1]
        by_from[edge.from_path].append(basename)
        import_graph.setdefault(edge.from_path, []).append(edge.to_path)

    from memlora.compression.centrality import compute_file_centrality
    centrality = compute_file_centrality(import_graph) if import_graph else {}

    all_paths = sorted(set(by_path.keys()) | set(by_from.keys()))
    entries: list[SkeletonEntry] = []

    for path in all_paths:
        path_nodes = by_path.get(path, [])
        entry = _build_entry(
            path, path_nodes, by_from.get(path, []),
            _MAX_METHODS_PER_CLASS,
        )
        entries.append(entry)

    # Estimate tokens for each entry using the single canonical counter, so the
    # skeleton budget is enforced in the same unit as the global ceiling.
    from memlora.compression.token_count import count_tokens
    from memlora.symbols.render import _render_entry
    for entry in entries:
        entry.token_estimate = max(1, count_tokens(_render_entry(entry)))

    total = sum(e.token_estimate for e in entries)
    if total <= budget_tokens:
        return entries

    # Budget phase 1: reduce methods per class (5 → 3 → 1)
    for method_limit in (3, 1):
        for entry in entries:
            for cls in entry.classes:
                cls.methods = cls.methods[:method_limit]
        for entry in entries:
            entry.token_estimate = max(1, count_tokens(_render_entry(entry)))
        total = sum(e.token_estimate for e in entries)
        if total <= budget_tokens:
            return entries

    # Budget phase 2: drop whole files.
    # Score = symbol density + PageRank centrality bonus + hot-file bonus.
    # Higher score = keep longer; lowest-score file dropped first. PageRank
    # captures transitive import importance (a file imported by central files
    # ranks above one imported by leaves with the same raw in-degree).
    def _file_score(e: SkeletonEntry) -> float:
        symbol_density = len(e.classes) * 3 + len(e.functions) + 1
        centrality_bonus = centrality.get(e.path, 0.0) * 100.0
        hot_bonus = 20 if e.path in _hot else 0
        return symbol_density + centrality_bonus + hot_bonus

    entries.sort(key=_file_score, reverse=True)
    while total > budget_tokens and len(entries) > 1:
        dropped = entries.pop()
        total -= dropped.token_estimate

    return entries


def _build_entry(
    path: str,
    path_nodes: list["SymbolNode"],
    import_basenames: list[str],
    method_limit: int,
) -> SkeletonEntry:
    class_nodes = [n for n in path_nodes if n.node_type == "class"]
    method_nodes = [n for n in path_nodes if n.node_type == "method"]
    func_nodes = [n for n in path_nodes if n.node_type == "function"]

    # Score classes by method count + field count
    def _class_score(name: str) -> int:
        methods = sum(1 for m in method_nodes if m.parent_name == name)
        # Estimate field count from comma-separated string
        node = next((n for n in class_nodes if n.name == name), None)
        fields_count = len(node.fields.split(",")) if node and node.fields else 0
        return methods + fields_count

    top_class_names = sorted(
        (n.name for n in class_nodes),
        key=_class_score,
        reverse=True,
    )[:_MAX_CLASSES_PER_FILE]

    skeleton_classes: list[SkeletonClass] = []
    for name in top_class_names:
        class_node = next((n for n in class_nodes if n.name == name), None)
        if class_node is None:
            continue
        cls_methods = [
            SkeletonMethod(
                name=m.name,
                signature=m.signature,
                return_type=m.return_type,
            )
            for m in method_nodes
            if m.parent_name == name
        ][:method_limit]
        skeleton_classes.append(SkeletonClass(
            name=name,
            bases=class_node.signature,  # signature field holds bases for class nodes
            fields=class_node.fields,
            methods=cls_methods,
        ))

    # Rank functions by importance, not alphabetically: API routes first (the
    # `fields` slot holds the route descriptor for functions), then public
    # functions, then name as a stable tiebreak. So when a file exceeds
    # _MAX_FUNCTIONS_PER_FILE the private helpers are dropped, not whatever
    # sorts last alphabetically.
    def _func_sort_key(n):
        is_route = 1 if n.fields else 0
        is_public = 1 if not n.name.startswith("_") else 0
        return (-is_route, -is_public, n.name)

    top_functions = sorted(func_nodes, key=_func_sort_key)[:_MAX_FUNCTIONS_PER_FILE]
    skeleton_funcs = [
        SkeletonMethod(name=f.name, signature=f.signature, return_type=f.return_type, route_info=f.fields)
        for f in top_functions
    ]

    # Deduplicate and cap imports
    seen_imports: set[str] = set()
    deduped_imports: list[str] = []
    for imp in import_basenames:
        if imp not in seen_imports:
            seen_imports.add(imp)
            deduped_imports.append(imp)
    deduped_imports = deduped_imports[:_MAX_IMPORTS_PER_FILE]

    return SkeletonEntry(
        path=path,
        imports=deduped_imports,
        classes=skeleton_classes,
        functions=skeleton_funcs,
    )
