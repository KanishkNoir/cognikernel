"""PageRank-based file centrality for code import graphs."""
from __future__ import annotations


def compute_file_centrality(
    import_graph: dict[str, list[str]],
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    """Compute PageRank centrality for files in an import graph.

    Args:
        import_graph: {source_file: [imported_files, ...]}
        damping: PageRank damping factor (standard 0.85).
        iterations: Fixed iteration count; converges well before 20 for
            typical codebases (< 1000 files).

    Returns:
        Centrality score per file — higher means more central.
    """
    all_files: set[str] = set(import_graph.keys())
    for targets in import_graph.values():
        all_files.update(targets)

    files = sorted(all_files)
    n = len(files)
    if n == 0:
        return {}

    centrality: dict[str, float] = {f: 1.0 / n for f in files}

    for _ in range(iterations):
        new_centrality: dict[str, float] = {f: (1.0 - damping) / n for f in files}
        for source, targets in import_graph.items():
            if not targets:
                continue
            contribution = damping * centrality.get(source, 1.0 / n) / len(targets)
            for target in targets:
                new_centrality[target] = new_centrality.get(target, 0.0) + contribution
        centrality = new_centrality

    return centrality


def centrality_factor(
    file_paths: list[str],
    centrality_map: dict[str, float],
) -> float:
    """Map file centrality scores to a [0.5, 1.5] multiplier.

    - No files affected → 1.0 (neutral)
    - Zero centrality  → 0.5 (mild suppression)
    - Max centrality   → 1.5 (boost)

    Uses the most central affected file (max, not average), because an event
    touching one highly-central file is at least as important as that file's
    centrality implies.
    """
    if not file_paths or not centrality_map:
        return 1.0
    max_c = max(centrality_map.values())
    if max_c == 0.0:
        return 1.0
    best = max(centrality_map.get(p, 0.0) for p in file_paths)
    return 0.5 + 1.0 * best / max_c
