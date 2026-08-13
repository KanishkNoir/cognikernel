# v0.1.1 — symbol extraction repair

**If your project contains TypeScript or JavaScript, upgrade.** Symbol
extraction for those languages was returning nothing on every `pip install` of
0.1.0, and the failure was silent.

```
pip install --upgrade cognikernel
```

---

## The headline bug

`tree-sitter-language-pack` changed its Python binding at 1.13.0 — the native
`parse_bytes()` / `root_node()` / `kind()` **methods** became py-tree-sitter
`parse()` / `.root_node` / `.type` **properties**. CogniKernel declared an
unbounded `tree-sitter-language-pack>=1.0`, so a fresh install resolved the new
release and every accessor raised `AttributeError`.

Extraction fails open by design — a symbol-graph gap must never break your
session — so nothing surfaced. It simply returned nothing:

```
typescript_support_status()  ->  False
valid TypeScript             ->  0 nodes, 0 edges
```

Our development environment was pinned by `uv.lock` to a version that still
worked, and CI installed from that lock. **CI was structurally incapable of
seeing the outage.** A lock file protects development and hides what users get.

Extraction now routes through adapters accepting **both** binding shapes, so
nobody on an older pack is stranded. Verified producing identical output on
1.9.1, 1.14.0, and 1.14.3.

A new `fresh-install` CI job installs with plain pip resolution — no lock — and
asserts the extractors actually emit symbols. A smoke import would have sailed
straight through this outage, so the assertion inspects real output.

---

## Also fixed

**Whole classes of file paths were invisible.** The path pattern's lookbehind
blocked anything preceded by `.`, `/`, or `\`, and its body class carried no
backslash. So `.claude/settings.json`, `./src/foo.py`, `/srv/app/main.py`, and
every `C:\...\file.py` produced no match at all — silently absent rather than
wrong.

**Catastrophic backtracking in that same pattern.** A run like `a./a./a./…`
without a valid extension backtracked exponentially: 3.3 s at 22 repetitions
and climbing, enough to stall extraction on a hostile transcript. Now linear,
with regression tests.

**Vendored dependencies were crowding out your code in the skeleton.**
Discovery now respects `.gitignore`, skips `.uv-cache`, `.claude`,
`.pytest_tmp`, `target`, `vendor` and similar, and ranks candidates so
first-party source wins the file budget instead of directory walk order
deciding. Stale out-of-scope nodes are pruned from existing graphs.

Measured on this repository: the symbol graph went from **4,655 of 4,934 nodes
(94%) vendored to zero**, and the Codebase skeleton section changed from
describing the `attrs` library's internals to describing the actual project.

**Descriptions beginning with a file path were corrupted** by sentence-case
capitalisation rewriting `src/…` into `Src/…`, which then matched nothing.

---

## New: quality gate on stored memory

Every extracted event now passes admission control before storage, returning
admit / downgrade / reject.

- Harness and compaction boilerplate ("Pick up the last task as if the break
  never happened") is rejected outright — it is never a project fact.
- Statements whose subject exists only in unstated context ("It must not be
  able to take down the pipeline") are **downgraded, not dropped**: weight
  collapses so they fall off the ranked block, while staying reachable through
  `recall` and `find_related`.
- **Path grounding** checks referential integrity between stored memory and the
  real codebase. Unknown paths downgrade rather than drop, since new files are
  legitimate; only near-miss truncations of a known path are rejected.

Measured across 8,840 real stored statements: **0.42% rejected, 3.98%
downgraded** — deliberately conservative, since a gate that over-fires is worse
than the defect it fixes.

A new `quality_telemetry` table records per-rule activity, surfaced by
`cognikernel doctor`.

---

## Notes

- **Schema migrates v18 → v19 automatically** on first open. Existing stores are
  not rewritten: these fixes prevent new defects rather than repairing old ones,
  and existing low-quality entries decay out normally.
- **Multi-language extraction (Go, Rust, Java) is designed but not shipped in
  this release.** Those files are tracked in the component map but produce no
  symbols yet.

**Full changelog:** [`CHANGELOG.md`](CHANGELOG.md) ·
[`v0.1.0...v0.1.1`](https://github.com/KanishkNoir/cognikernel/compare/v0.1.0...v0.1.1)
