# Changelog

All notable changes to CogniKernel are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed

- **New projects no longer refuse the first read of every file in the
  skeleton.** `cognikernel init` used to switch on "strict" mode, which turns
  away Claude's first attempt to read any file listed in the project skeleton
  and lets a second attempt through, on the idea that the skeleton's
  signatures would often be enough. They rarely were: across the four-project
  benchmark, 89% of those refusals were followed straight away by the same
  read, so each one cost an extra round trip — more time and more tokens — and
  saved nothing. New projects now start in "advisory" mode. The rule that did
  work stays on for everyone: re-reading a file already read in the same
  session is still refused, and that refusal was almost never retried.
  Existing projects keep whatever their `.cognikernel/config.toml` says; set
  `hook_policy = "strict"` there to opt back in.

### Fixed

- **A fresh install had no memory tools at all.** CogniKernel asked for any
  version of the `mcp` library from 1.0 up. `mcp` 2.0, released 2026-07-28,
  removed a module CogniKernel's server is built on, so a new
  `pip install cognikernel` pulled in 2.x and the server stopped at startup.
  Claude Code showed it as "failed": the session-context block still arrived,
  but `recall`, `find_related`, `skeleton` and `get_session_state` were
  missing. This hit 0.1.2 for anyone who didn't already have `mcp` 1.x
  installed. The requirement is now capped below 2, and the fresh-install
  check in CI now starts the real server and requires it to list its tools.
  On 0.1.2, run `pip install "mcp<2"` to get the tools back.

- **`cognikernel telemetry` counted most API responses two or three times.**
  Claude Code writes one transcript line per piece of a response (text,
  thinking, each tool call) and repeats the response's token usage on every
  one of those lines. Telemetry added them all up, so the cache and token
  figures `cognikernel doctor` showed were inflated — by 1.9× to 3.0× on the
  benchmark projects, and by different amounts on different projects, so they
  could not even be compared with each other. Usage is now counted once per
  response. Rows recorded the old way are labelled and kept out of the
  figures; running `cognikernel telemetry <project_path>` again re-counts any
  session whose transcript still exists.

### Added

- **`cognikernel why <project_path> <subject>` explains a claim.** Give it a claim id
  (`#123`) or words from the claim, and it shows what the claim says, which
  session it came from — as "session 2 of 4", in the order the sessions
  happened, not as an opaque id — and the sentence in the conversation it was
  extracted from, and who said it. It also shows what the quality gate noted
  when the claim was admitted, why it ranks where it does — its weight in the
  session block broken into the six factors that produce it (type, how
  recently and how often it came up, how central and how active its files
  are), or which claim it was folded into — and its history: the claims it
  replaced or was replaced by, when, and which rule decided it. Replaced claims can be looked
  up too, which is the point when you are asking why memory changed its mind.
  Anything the store did not record — the commit a claim was captured
  against, or a precise sentence position, which the current extractor never
  saves — is named as "not recorded" instead of being left out. The source
  sentence is found by searching the claim's own evidence, so it is shown as
  located, not as recorded. Long histories are shortened around the claim;
  `--json` keeps everything. It changes nothing in memory, but like every
  command it first brings an older store's schema up to date.

- **`cognikernel show --as-of <when>` shows what memory believed at a past
  time.** Give it a date (the end of that day), a date and time, or a git
  commit, and it lists the claims that were live then: anything created later
  is left out, and anything replaced or archived at a recorded time by then is
  counted as ended.
  It also says how far back it can be trusted. CogniKernel only started
  recording *when* a claim was replaced or archived in this release, so on an
  older store every replaced claim has no end time. Those are listed
  separately under "Timing unknown" — never guessed to be live or gone — and
  the output states from which date end times exist. Ranking is rebuilt from
  the sessions up to that time; how central a file is still comes from
  today's code, and the output says so.

- **`cognikernel explain-recall <project_path> <query>` shows why memory
  retrieved what it did — and why it didn't retrieve something.** It lists the words actually
  searched, whether each search method was available (keyword search, and
  meaning-based search, which needs the embedding model loaded), and every
  candidate each method found, with its rank and whether it made the cut for
  the `recall` tool. It then walks the per-prompt push step by step: what was
  dropped for just repeating the prompt, what the session had already seen,
  and a reason for every accept or reject ("dense rank 7 > 5", "only 1 shared
  term, needs 2"), and which claims the injection size limit leaves out. Add
  `--claim #123` to ask about one claim — including one that was replaced,
  which recall never searches. The explanation runs the
  same code recall and the push run, so it can't describe something they
  don't do.

- **`cognikernel doctor` shows how many extra round trips CogniKernel's own
  tools caused.** It now reports how many API responses a project's sessions
  took, and what share of them CogniKernel added: responses that only called
  a memory tool, reads it refused, and retries of those refused reads. This is
  the cost that matters for "does memory cost more than not having it" — the
  injected block itself is only 1–2% of a session's cost.

- **A `tool_guidance` setting in `.cognikernel/config.toml`.** `"eager"` (the
  default, unchanged) tells Claude to check the skeleton or call `recall`
  before reading files. `"lean"` keeps the tools and the skeleton but stops
  asking for a check before every read: on the benchmark, a `skeleton` call
  was followed by a read of the same file 41–67% of the time anyway. It exists
  to measure whether that advice pays for itself; the default changes only if
  it does.

### Notes

- Schema migrates automatically on first open (v20 → v22): belief-history
  columns (021) and round-trip telemetry columns (022). Existing rows are not
  rewritten.

---

## [0.1.2] — 2026-08-15

Four fixes to how CogniKernel decides which files belong in your project
skeleton (the summary Claude sees instead of reading every file), one fix
to a tool that was silently missing an entire category of edits, and one
promise the `skeleton` tool made that it wasn't actually keeping.

### Fixed

- **Large projects got worse file rankings than small ones, for no reason
  related to the code itself.** A scoring bug meant a file's "how central
  is this file to the rest of the codebase" score counted for a lot on a
  small project and almost nothing on a large one — so the same file could
  rank very differently depending only on how many files were around it,
  not on anything about the file. This is now scale-independent: a file's
  importance is measured the same way regardless of project size.

- **CogniKernel was making files worse before it removed them.** When a
  project had too many files to fit in the budget, it used to first strip
  detail (methods, signatures) from *every* file, and only then start
  removing whole files. That meant you'd sometimes get thin, stripped-down
  summaries of files you'd never touch, while a file you actually needed
  got cut anyway. It now drops the least useful whole files first, and only
  trims detail as a last resort if one remaining file still doesn't fit.

- **Test files were pushing real source code out of the picture.** Test
  files naturally contain lots of small functions (one per test case), which
  made them look "important" by the old scoring and let them outrank the
  actual application code — in this project's own skeleton, a test file
  ranked *above every real source file*. Test and tooling files are now
  deliberately weighted lower (not hidden — you still work on tests), so
  real source code wins the limited space by default.

- **A "this file was recently worked on" bonus couldn't tell files apart.**
  Any file mentioned a couple of times got the exact same boost, whether
  that activity was from an hour ago or several sessions back — and a file
  touched only once, even in the session that just ended, got no boost at
  all. In a real, measured case this caused a file someone had *just
  edited* to be dropped from the skeleton in favor of an older file that
  happened to be mentioned more times, further in the past. The boost is
  now a sliding scale that weighs both how recently and how often a file
  came up, so a just-edited file reliably outranks a stale one it previously
  lost to.

- **Editing files with `MultiEdit` didn't update CogniKernel's understanding
  of them at all.** `MultiEdit` — the tool Claude Code uses for most
  multi-part edits — was never wired up to refresh a file's entry after a
  change, so its skeleton listing could silently go stale the moment you
  used it. This is now treated exactly like a normal edit.

- **The `skeleton` tool's "you'll get the full picture" promise wasn't
  fully true.** This tool exists so you can ask for one file's complete,
  unabridged details — notably, it's what CogniKernel itself tells you to
  use after declining to let you re-read a file it's already summarized.
  But a separate internal limit meant a class with, say, 18 methods still
  only showed 5 of them, even through this "full detail" path. It now
  genuinely returns everything for that file.

- **Trimmed-down file listings didn't say anything was missing.** If a file
  had more methods or functions than fit in the summary, the extra ones were
  quietly dropped with no indication anything had been cut — it just looked
  complete. Listings now say so explicitly, e.g. `(+3 more public symbols
  not shown)`, so you know to look closer instead of assuming you've seen
  everything.

### Added

- **The groundwork for smarter "recently active" tracking.** CogniKernel now
  records every real file edit per work session in its own dedicated table.
  Nothing uses this data yet in this release — it's being collected so a
  future release can rank files by genuine edit activity instead of the
  current mention-based heuristic.

### Notes

- Schema migrates automatically on first open (v19 → v20). Existing stores
  are not rewritten.
- Adaptive per-project token budgets remain designed but not shipped in this
  release.

---

## [0.1.1] — 2026-08

### Fixed

- **TypeScript/JavaScript symbol extraction was dead for every `pip install`.**
  `tree-sitter-language-pack` changed its binding at 1.13.0 — the native
  `parse_bytes()` / `root_node()` / `kind()` *methods* became py-tree-sitter
  `parse()` / `.root_node` / `.type` *properties*. `pyproject.toml` declared an
  unbounded `>=1.0`, so a fresh install resolved 1.14.0 and every accessor
  raised `AttributeError`. Extraction fails open by design, so the failure was
  silent: `typescript_support_status()` returned `False` and valid TypeScript
  produced **0 nodes, 0 edges**. Our own environment was pinned to 1.9.1 by
  `uv.lock`, which is why CI never saw it.

  Extraction now routes through adapters that accept **both** binding shapes,
  so no one on an older pack is stranded. Verified producing identical output
  on 1.9.1 and 1.14.0.

- **Grounding penalised languages the symbol walk cannot parse.** The path
  inventory is built from a discovery walk that globs only
  `.py/.ts/.tsx/.js/.jsx`, so a real `internal/db/pool.go` was absent for
  reasons unrelated to whether it exists — and was marked `unverified` with its
  weight halved. Every component event in a Go, Rust, Java, or C# project would
  have been pushed off the budget-ranked block. Grounding now applies only to
  file types the inventory actually covers.

- **File paths written as dotfiles, `./`-relative, absolute, or with Windows
  separators were invisible.** The path pattern's lookbehind blocked any path
  preceded by `.`, `/`, or `\`, and its body class carried no backslash, so
  `.claude/settings.json`, `./src/foo.py`, `/srv/app/main.py`, and every
  `C:\...\file.py` produced no match at all. Absolute paths additionally
  required a project root to resolve, which is now threaded through.

- **Catastrophic backtracking in the path pattern.** A run such as
  `a./a./a./…` with no valid extension backtracked exponentially — 3.3 s at 22
  repetitions and climbing, enough to stall extraction on a hostile transcript.
  The pre-existing pattern shared the flaw. Now linear, with regression tests.

- **Descriptions beginning with a file path were corrupted** by sentence-case
  capitalisation rewriting `src/…` to `Src/…`, which then matched nothing in
  the codebase.

- **Vendored and cached files crowded out first-party code in the skeleton.**
  Discovery now respects `.gitignore`, skips `.uv-cache`, `.claude`,
  `.pytest_tmp`, `.ruff_cache`, `target`, `vendor` and similar, and ranks
  candidates so project code wins the file budget rather than walk order
  deciding. Stale out-of-scope nodes are pruned from existing graphs. Measured
  on this repository: the symbol graph went from 4,655 of 4,934 nodes (94%)
  vendored to zero, and the Codebase skeleton section changed from describing
  the `attrs` library to describing the project.

### Added

- **Quality gate on the event write path.** Every extracted event now passes
  through admission control before storage, returning admit / downgrade /
  reject. Harness and compaction boilerplate is rejected outright; statements
  whose subject exists only in unstated context are downgraded so they fall off
  the ranked block while remaining reachable through `recall` and
  `find_related`. Measured across 8,840 real stored statements: 0.42% rejected,
  3.98% downgraded.

- **Path grounding** — referential integrity between stored memory and the
  actual codebase. Unknown paths are downgraded rather than dropped, since new
  files are legitimate; only near-miss truncations of a known path are
  rejected.

- **`quality_telemetry` table** (schema v19) recording per-rule gate activity,
  surfaced by `cognikernel doctor`.

- **Render-time structural invariants** — box-drawing artifacts and cross-type
  duplicate statements are filtered from the injected block. The filter runs
  over the event set before assembly so the render ledger stays accurate.

### Notes

- Multi-language symbol extraction (Go, Rust, Java) is **designed but not
  shipped** in this release. Those files are tracked in the component map but
  produce no symbols.
- Schema migrates from v18 to v19 automatically on first open. Existing stores
  are not rewritten; the fixes prevent new defects rather than repairing old
  ones, and existing low-quality entries decay out.

---

## [0.1.0] — 2026-07

Initial public release.
