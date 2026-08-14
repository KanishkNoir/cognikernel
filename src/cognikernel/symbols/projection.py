"""Compress symbol graph nodes + edges into token-efficient SkeletonEntry objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognikernel.symbols.extractor import SymbolEdge, SymbolNode

_SKELETON_TOKEN_BUDGET = 800   # default; callers pass config.skeleton_budget
_MAX_CLASSES_PER_FILE = 5
_MAX_METHODS_PER_CLASS = 5
_MAX_FUNCTIONS_PER_FILE = 10
_MAX_IMPORTS_PER_FILE = 8


@dataclass(frozen=True)
class Caps:
    """Per-file member caps. `unlimited()` is what the `skeleton` MCP tool needs:

    that tool's docstring promises "the token budget is lifted... this is the
    full-fidelity pull path" for an explicit single-file query, but the budget
    was never what capped output — these constants, applied in `_build_entry`
    before any budget logic, were. Passing `budget_tokens=1_000_000` alone (as
    resources.py did) left a class with 18 methods returning 5 regardless, and
    under strict-mode read denial the assistant is routed to exactly this tool
    for "the full signatures" and cannot get them. Caps must be threaded
    explicitly; there is no budget value that raises them.
    """
    classes: int = _MAX_CLASSES_PER_FILE
    methods: int = _MAX_METHODS_PER_CLASS
    functions: int = _MAX_FUNCTIONS_PER_FILE
    imports: int = _MAX_IMPORTS_PER_FILE

    @staticmethod
    def unlimited() -> "Caps":
        big = 1_000_000
        return Caps(classes=big, methods=big, functions=big, imports=big)


DEFAULT_CAPS = Caps()

# Score component weights. These are meaningful only because centrality is
# normalised to [0,1] against the graph max before being weighted — see
# _file_score. symbol_density is an unweighted integer count (~1-26), so these
# are calibrated against that range: structure is worth roughly as much as a
# mid-sized file's symbol count, and recency somewhat more.
_CENTRALITY_WEIGHT = 15.0
_HOT_WEIGHT = 20.0

# Directories whose files describe how the project is exercised rather than what
# it does. Their symbol counts are inflated by construction, so without a
# penalty they crowd out first-party source.
_SUPPORT_PREFIXES = ("tests/", "test/", "scripts/", "docs/", "examples/",
                     "benchmarks/", "fixtures/")
_SUPPORT_PENALTY = 0.35


def _is_support_path(path: str) -> bool:
    """True for test/tooling paths, which are deprioritised but never excluded."""
    norm = path.replace("\\", "/")
    if norm.startswith(_SUPPORT_PREFIXES):
        return True
    return "/tests/" in norm or "/test/" in norm


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
    # Public classes/functions cut by caps.classes/caps.functions — i.e. ones
    # that WOULD have appeared in render.py's `Import:` line had they not been
    # capped. Computed once at build time; unlike per-class method residuals
    # (not tracked — that's the separate, unimplemented §3.1 work) nothing
    # downstream mutates the class/function list, so these can't go stale.
    classes_omitted: int = 0
    functions_omitted: int = 0


def compress_to_skeleton(
    nodes: list["SymbolNode"],
    edges: list["SymbolEdge"],
    budget_tokens: int = _SKELETON_TOKEN_BUDGET,
    hot_paths: frozenset[str] | None = None,
    caps: Caps = DEFAULT_CAPS,
) -> list[SkeletonEntry]:
    """Compress symbol graph into SkeletonEntry list fitting within budget_tokens.

    hot_paths: set of recently-active file paths that should be prioritised
               over lower-activity files when the budget forces drops.
    caps: per-file member caps. Pass `Caps.unlimited()` for an explicit
          single-file pull where every member should render regardless of
          budget — see `Caps` docstring.
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

    from cognikernel.compression.centrality import compute_file_centrality
    centrality = compute_file_centrality(import_graph) if import_graph else {}

    all_paths = sorted(set(by_path.keys()) | set(by_from.keys()))
    entries: list[SkeletonEntry] = []

    for path in all_paths:
        path_nodes = by_path.get(path, [])
        entry = _build_entry(
            path, path_nodes, by_from.get(path, []),
            caps.methods, caps=caps,
        )
        entries.append(entry)

    # Estimate tokens for each entry using the single canonical counter, so the
    # skeleton budget is enforced in the same unit as the global ceiling.
    from cognikernel.compression.token_count import count_tokens
    from cognikernel.symbols.render import _render_entry
    for entry in entries:
        entry.token_estimate = max(1, count_tokens(_render_entry(entry)))

    total = sum(e.token_estimate for e in entries)
    if total <= budget_tokens:
        return entries

    # Score = symbol density + normalised centrality + hot-file bonus.
    # Higher score = keep longer; lowest-score file dropped first.
    #
    # CENTRALITY IS NORMALISED AGAINST THE GRAPH MAX, not multiplied by a bare
    # constant. PageRank is normalised to sum to 1 over the graph, so raw values
    # scale as 1/N: on a 10-file project the mean is 0.10 and `* 100` yields a
    # bonus of ~10, while on a 240-file project the same expression yields
    # ~0.42. symbol_density is an integer count that does not shrink with N, so
    # the relative weight of structure against size silently changed by more
    # than an order of magnitude with project size — a different scoring
    # function per project. Dividing by the graph's own max makes the term
    # scale-free, so the weights below mean the same thing everywhere.
    cmax = max(centrality.values(), default=0.0) or 1.0

    def _file_score(e: SkeletonEntry) -> float:
        symbol_density = len(e.classes) * 3 + len(e.functions) + 1
        centrality_bonus = (centrality.get(e.path, 0.0) / cmax) * _CENTRALITY_WEIGHT
        hot_bonus = _HOT_WEIGHT if e.path in _hot else 0
        score = symbol_density + centrality_bonus + hot_bonus
        # Test and tooling files inflate symbol_density by construction — a test
        # class per scenario, a test method per case — so they outranked real
        # source in the measured baseline (a test module was the second entry in
        # this repo's own skeleton, above every src/ file). Deprioritise rather
        # than exclude: agents genuinely do work on tests (11% of observed
        # file-touches), and excluding them outright measured no better than
        # penalising them.
        if _is_support_path(e.path):
            score *= _SUPPORT_PENALTY
        return score

    # DROP BEFORE DEGRADE. Previously this ran the method-limit ladder
    # (5 -> 3 -> 1) across EVERY file before dropping a single one, so the
    # budget was spent on 1-method stubs of irrelevant files instead of full
    # signatures of relevant ones. Freeing budget by removing a file the agent
    # will not open is strictly better than blinding every file it will.
    entries.sort(key=_file_score, reverse=True)
    while total > budget_tokens and len(entries) > 1:
        dropped = entries.pop()
        total -= dropped.token_estimate

    # Only now, if a single entry still exceeds the budget, degrade its detail.
    # This is the genuine last resort: there is nothing left to drop.
    if total > budget_tokens:
        for method_limit in (3, 1):
            for entry in entries:
                for cls in entry.classes:
                    cls.methods = cls.methods[:method_limit]
                entry.token_estimate = max(1, count_tokens(_render_entry(entry)))
            total = sum(e.token_estimate for e in entries)
            if total <= budget_tokens:
                break

    return entries


def _build_entry(
    path: str,
    path_nodes: list["SymbolNode"],
    import_basenames: list[str],
    method_limit: int,
    caps: Caps = DEFAULT_CAPS,
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

    ranked_class_names = sorted(
        (n.name for n in class_nodes),
        key=_class_score,
        reverse=True,
    )
    top_class_names = ranked_class_names[:caps.classes]
    # Only PUBLIC cut classes count: a cut private class was never going to
    # appear in the import hint even if kept, so it isn't part of the
    # completeness claim being made.
    classes_omitted = sum(
        1 for nm in ranked_class_names[caps.classes:] if not nm.startswith("_")
    )

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

    ranked_funcs = sorted(func_nodes, key=_func_sort_key)
    top_functions = ranked_funcs[:caps.functions]
    functions_omitted = sum(
        1 for f in ranked_funcs[caps.functions:] if not f.name.startswith("_")
    )
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
    deduped_imports = deduped_imports[:caps.imports]

    return SkeletonEntry(
        path=path,
        imports=deduped_imports,
        classes=skeleton_classes,
        functions=skeleton_funcs,
        classes_omitted=classes_omitted,
        functions_omitted=functions_omitted,
    )
