"""Budget enforcement and score-scale invariants for the skeleton.

Two defects motivated these tests, both found while measuring ranking quality:

(a) SCALE MISMATCH. `_file_score` summed `symbol_density` (an integer count,
    ~1-26, independent of project size) with `centrality * 100`. PageRank is
    normalised to sum to 1 over the graph, so mean centrality is 1/N and the
    bonus scales as 100/N: ~10 on a 10-file project, ~0.42 on a 238-file one.
    The relative weight of structure vs size therefore changed ~24x with
    project size — effectively a different scoring function per project.

(b) DEGRADE BEFORE DROP. Budget phase 1 stripped every class to 3 then 1
    methods across ALL files before phase 2 dropped a single low-ranked file.
    The budget was spent on 1-method stubs of irrelevant files instead of full
    signatures of relevant ones.
"""
from __future__ import annotations

from cognikernel.symbols.extractor import SymbolNode


def _nodes(path: str, n_classes: int = 1, n_methods: int = 4, project="p"):
    out = []
    for c in range(n_classes):
        cname = f"C{c}_{path.replace('/', '_').replace('.py', '')}"
        out.append(SymbolNode(path=path, node_type="class", name=cname,
                              parent_name="", signature="", return_type="",
                              fields="", project_id=project, updated_at=0))
        for m in range(n_methods):
            out.append(SymbolNode(path=path, node_type="method", name=f"m{m}",
                                  parent_name=cname, signature="(self, a:int)",
                                  return_type="str", fields="",
                                  project_id=project, updated_at=0))
    return out


class TestScoreScaleInvariance:
    """The centrality term must contribute on a comparable scale regardless of
    project size.

    NOTE ON SCOPE: this pins the ARITHMETIC, not a ranking flip. A synthetic
    hub-and-spokes case does NOT flip, because PageRank concentrates mass on a
    hub and its share stays high as N grows. The 1/N shrinkage bites
    MIDDLING-centrality files, whose absolute bonus collapses while
    symbol_density (an integer count) does not. Whether that reorders any real
    project is an empirical question measured separately; what is provable here
    is that the raw term is size-dependent and the normalised one is not.
    """

    def test_raw_centrality_bonus_is_size_dependent(self) -> None:
        """Documents the defect: identical graph shape, different scale."""
        from cognikernel.compression.centrality import compute_file_centrality

        def mean_bonus(n: int) -> float:
            g = {f"f{i}.py": [f"f{(i + 1) % n}.py"] for i in range(n)}
            c = compute_file_centrality(g)
            return (sum(c.values()) / len(c)) * 100.0

        small, large = mean_bonus(10), mean_bonus(240)
        assert small / large > 10, (
            f"expected the raw bonus to shrink with N (got {small:.2f} vs "
            f"{large:.2f}) — if this fails the premise changed"
        )

    def test_normalised_centrality_is_size_invariant(self) -> None:
        """The fix: normalising against the graph max is scale-free."""
        from cognikernel.compression.centrality import compute_file_centrality

        def mean_norm(n: int) -> float:
            g = {f"f{i}.py": [f"f{(i + 1) % n}.py"] for i in range(n)}
            c = compute_file_centrality(g)
            cmax = max(c.values()) or 1.0
            return sum(v / cmax for v in c.values()) / len(c)

        assert abs(mean_norm(10) - mean_norm(240)) < 0.05


class TestDropBeforeDegrade:
    """Detail must not be stripped from files that are being kept while
    lower-ranked files are still occupying budget."""

    def test_kept_file_retains_methods_when_budget_forces_drops(self) -> None:
        from cognikernel.symbols.projection import compress_to_skeleton

        nodes = _nodes("src/important.py", n_classes=1, n_methods=5)
        for i in range(25):
            nodes += _nodes(f"src/other{i}.py", n_classes=1, n_methods=5)

        # A budget that cannot hold everything: some files MUST be dropped.
        entries = compress_to_skeleton(nodes, [], budget_tokens=300)
        assert entries, "budget enforcement dropped everything"

        # Whatever survived should carry real detail, not 1-method stubs,
        # because dropping files is the correct way to free budget.
        kept_methods = [len(c.methods) for e in entries for c in e.classes]
        assert kept_methods, "no classes survived"
        assert max(kept_methods) > 1, (
            f"every kept class was stripped to <=1 method ({kept_methods}); "
            "detail was degraded before low-ranked files were dropped"
        )

    def test_budget_is_respected(self) -> None:
        from cognikernel.symbols.projection import compress_to_skeleton

        nodes = []
        for i in range(30):
            nodes += _nodes(f"src/f{i}.py", n_classes=2, n_methods=5)
        entries = compress_to_skeleton(nodes, [], budget_tokens=400)
        assert sum(e.token_estimate for e in entries) <= 400 or len(entries) == 1

    def test_everything_fits_is_unchanged(self) -> None:
        from cognikernel.symbols.projection import compress_to_skeleton

        nodes = _nodes("src/a.py", n_classes=1, n_methods=3)
        entries = compress_to_skeleton(nodes, [], budget_tokens=100_000)
        assert len(entries) == 1
        assert len(entries[0].classes[0].methods) == 3


class TestObjective:
    """What the ranking optimises for.

    Measured on 17 projects / 35 session transitions, scoring "of the files the
    agent touched next session, how many did the skeleton hold": the shipped
    objective reached 37.5% recall while a recency-led objective that requires
    real symbols and deprioritises tests reached ~60%. These pin the three
    behaviours responsible for that gap.
    """

    def test_files_with_no_symbols_are_not_selected(self) -> None:
        """An empty __init__.py is cheap, so under budget pressure it displaces
        a real module while telling the agent nothing."""
        from cognikernel.symbols.projection import compress_to_skeleton

        nodes = _nodes("src/real.py", n_classes=2, n_methods=4)
        # a path present only as an import target — no symbols of its own
        from cognikernel.symbols.extractor import SymbolEdge
        edges = [SymbolEdge(project_id="p", from_path="src/real.py",
                            to_path="src/__init__.py",
                            edge_type="imports", is_external=False)]
        entries = compress_to_skeleton(nodes, edges, budget_tokens=60)
        paths = [e.path for e in entries]
        assert "src/real.py" in paths
        assert "src/__init__.py" not in paths, (
            f"a symbol-less file occupied budget: {paths}"
        )

    def test_recently_touched_outranks_merely_bulky(self) -> None:
        """Recency is the strongest measured predictor of next-touch; a big
        untouched file must not displace a smaller recently-worked one."""
        from cognikernel.symbols.projection import compress_to_skeleton

        nodes = _nodes("src/worked_on.py", n_classes=1, n_methods=2)
        nodes += _nodes("src/big_untouched.py", n_classes=5, n_methods=5)
        entries = compress_to_skeleton(
            nodes, [], budget_tokens=90,
            hot_paths=frozenset({"src/worked_on.py"}),
        )
        paths = [e.path for e in entries]
        assert paths and paths[0] == "src/worked_on.py", (
            f"recently-touched file did not rank first: {paths}"
        )

    def test_graded_hot_weight_flips_ranking_that_flat_binary_gets_wrong(self) -> None:
        """Reproduces the real failure the graded hot bonus fixes (found by
        diagnosing an actual store): a file edited in the most recent session
        but mentioned only once must not lose out to a file with more total
        mentions but nothing recent. A flat binary hot_paths set can only
        express "cleared the mention threshold or not", so the older,
        more-mentioned file wins; a graded weight (as
        session._compute_hot_weights produces) can express "less mentioned but
        more recent" and gets it right."""
        from cognikernel.symbols.projection import compress_to_skeleton

        nodes = _nodes("src/old_favorite.py", n_classes=1, n_methods=2)
        nodes += _nodes("src/just_edited.py", n_classes=1, n_methods=2)

        # Old scheme: only the multiply-mentioned file clears the min_mentions
        # cliff upstream, so it alone lands in the hot set.
        entries_flat = compress_to_skeleton(
            nodes, [], budget_tokens=1,
            hot_paths=frozenset({"src/old_favorite.py"}),
        )
        assert [e.path for e in entries_flat] == ["src/old_favorite.py"], (
            "fixture drifted — the flat scheme no longer drops the recently-"
            "edited file, so this test no longer demonstrates the bug it "
            "exists to catch"
        )

        # New scheme: recency outweighs the older file's extra mentions.
        entries_graded = compress_to_skeleton(
            nodes, [], budget_tokens=1,
            hot_paths={"src/old_favorite.py": 0.4, "src/just_edited.py": 1.0},
        )
        assert [e.path for e in entries_graded] == ["src/just_edited.py"]

    def test_tests_are_deprioritised_against_source(self) -> None:
        """Test files inflate symbol counts (test classes + test methods) and
        outranked src/ in the measured baseline."""
        from cognikernel.symbols.projection import compress_to_skeleton

        nodes = _nodes("src/core.py", n_classes=1, n_methods=3)
        nodes += _nodes("tests/test_core.py", n_classes=3, n_methods=8)
        entries = compress_to_skeleton(nodes, [], budget_tokens=110)
        paths = [e.path for e in entries]
        assert paths and paths[0] == "src/core.py", (
            f"a test file outranked source: {paths}"
        )
