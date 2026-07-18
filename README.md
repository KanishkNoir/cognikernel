# CogniKernel

**Persistent, structured project memory for Claude Code — and Codex.** CogniKernel
watches a coding session through its hook surfaces, extracts the *decisions,
constraints, and abandoned approaches* worth keeping, consolidates them into an
event-sourced store, and injects them back as a compact context block the next
time you work — so the agent stops re-deciding what you already decided. The store
is keyed on the project path, so memory made in one tool travels to the other.

It is **not** a vector-database wrapper. It is an event-sourced log of *typed*
memory with lexical-primary retrieval, write-time consolidation, and a fail-open
reliability spine designed never to break your session.

**And there is no LLM in the loop.** Most memory tools work by sending your
transcripts to a generative model to "summarize what mattered" — another API
key, per-session token cost, added latency, and your session content leaving
the machine. CogniKernel treats extraction as *classification, not generation*:
a deterministic sanitize → classify → consolidate pipeline, with two small
fine-tuned encoder models (~130 MB ONNX, run locally on CPU in milliseconds)
scoring salience and detecting when a new decision supersedes an old one. No
API calls, no tokens billed, nothing leaves your machine. The only LLM involved
is the coding agent you already run — CogniKernel makes it remember.

> **Naming:** CogniKernel is the project; `memlora` (package `memlora-edge`) is
> the Python module and CLI it ships as — the working name the code grew up
> under. One project, two names: `memlora init`, `memlora doctor`, etc.

---

## The memory loop

Everything CogniKernel does is one loop: **observe → extract → consolidate →
store → retrieve → assemble → inject**. The left rail captures; the right rail
recalls; the spine underneath keeps both honest.

```
   +========================= CLAUDE CODE SESSION =========================+
   | working memory  -  the context window the agent reasons in            |
   +--------------+-----------------------------------------+--------------+
                  | Stop hook captures transcript           | inject block
                  v                                         ^
   +--------------+---------------+         +---------------+--------------+
   | [1] EXTRACTION PIPELINE      |         | [6] COMPRESSION + INJECTION  |
   |   sanitize -> classify ->    |         |   authority-weighted budget, |
   |   salience (ONNX) ->         |         |   drop-to-fit (keep every    |
   |   decision-key + contracts   |         |   constraint) -> block       |
   +--------------+---------------+         +---------------+--------------+
                  | enqueue                                 | rank + fit
                  v                                         ^
   +--------------+---------------+         +---------------+--------------+
   | [2] WORKER + CONSOLIDATION   |         | [5] RETRIEVAL                |
   |   claim -> delta-merge ->    |         |   FTS5 BM25 + dense -> RRF   |
   |   supersede (latest-wins) -> |         |   prohibition_search (K1)    |
   |   project   (idempotent)     |         |   skeleton graph (PageRank)  |
   +--------------+---------------+         +---------------+--------------+
                  | persist (atomic)                        | recall
                  v                                         ^
   +--------------+-----------------------------------------+--------------+
   | [3] EVENT-SOURCED STORE   *   SQLite (WAL)                            |
   | typed events | evidence | provenance | FTS5 | embeddings | ledger     |
   +----------------------------------------------------------------------+
   | [4] RELIABILITY SPINE   atomic migrations | idempotent replay |       |
   | doctor --strict | fail-open hooks | import-linter | CI gate           |
   +----------------------------------------------------------------------+
```

---

## The four hook surfaces

CogniKernel attaches to a session at four points. Each is fail-open — if memory
is unavailable or errors, the hook logs at `WARNING`, returns cleanly, and the
session continues.

| Surface | Hook | Authority | What it does |
|---|---|---|---|
| **Session block** | `SessionStart` | advisory | injects the canonical decisions/constraints/skeleton block |
| **CK-1 recall** | `UserPromptSubmit` | advisory | surfaces prompt-relevant memory, dual-evidence gated, dedup'd via render ledger |
| **Read/Edit gate** | `PreToolUse` | **hard / JIT** | read-efficiency gate on Read/Grep; **just-in-time prohibition surfacing** on Write/Edit (K2) |
| **Capture** | `Stop` | side-effect | extracts and persists decisions — you never write memory to CLAUDE.md by hand |

---

## What gets remembered

Memory is **typed**, not free-text chunks. The type drives ranking, rendering,
and supersession:

- `DECISION` — a choice that was made ("use Redis for the rate limiter")
- `CONSTRAINT_HARD` / `CONSTRAINT_SOFT` — rules, graded by deontic force
- `APPROACH_ABANDONED_DO_NOT_RETRY` — a dead end, kept in the **graveyard** so
  the agent doesn't re-attempt it
- conventions, config facts, schema decisions (canonical role keys)

A **decision key** lets a later restatement *supersede* an earlier one
(latest-wins), so the store self-consolidates instead of accumulating
contradictions. An optional cross-encoder adds semantic supersession when the
encoder backend is installed.

---

## Retrieval

Lexical-primary, with dense as a fused signal — never pure vector:

- **Hybrid core** — FTS5 BM25 ∪ optional dense embeddings → Reciprocal Rank Fusion
- **prohibition_search** — a type-restricted lexical pool so a "do not do X" rule
  can't be crowded out by topically-similar prose at the moment you're about to do X
- **Skeleton** — an AST symbol graph ranked by PageRank; `find_related` unions
  **semantic (embedding) neighbours** with **import-graph-adjacent** events to
  surface what a change touches (the semantic axis needs the `embedding` extra)
- **Golden-record consolidation at read** — latest-wins reconciliation so recall
  returns one coherent answer, not a pile of revisions

---

## What it saves you

Benchmarked in a three-arm comparison — CogniKernel vs flat curated notes vs no
memory — with real agent sessions across four multi-session projects:

- **File reads: the universal win.** The CogniKernel arm made the fewest file
  reads in *every* project — typically **2–4× fewer** (23 vs 63, 16 vs 47/53,
  40 vs 89/83), and in the best case **3 reads vs 29** because the injected
  block + AST skeleton carried the whole repo's shape. Fewer reads means fewer
  tool round-trips and more of the context window left for actual work — your
  session gets *longer* before compaction, not just cheaper.
- **Tokens: up to ~20% cheaper where memory matters.** Price-weighted token
  cost (cache-write 1.25×, cache-read 0.1×, output 5×) came out **18–23% lower**
  on projects with evolving decisions and cross-file dependencies — and roughly
  a wash on small implementation-heavy projects where the code itself is cheap
  to re-read. We publish the honest number, not the raw-token one (raw sums
  look ~30–40% better, but ~95% of any session's bill is discounted cache-read).
- **Recall instead of re-derivation.** Where memory earns its keep is projects
  whose state is too large, too evolving, or too long-lived to re-derive
  cheaply: the agent starts already knowing the decisions, constraints, and
  dead ends, instead of spending the first quarter of the session rediscovering
  them.

---

## Reliability — the spine

The system is designed to degrade *legibly*, never silently:

- **Atomic migrations** — each numbered migration applies its body + version bump
  in one transaction; safe to crash mid-script
- **Idempotent replay** — a re-run worker job can't double-count or drift decay (evidence-provenance guard)
- **Fail-open hooks** — every surface swallows its own failure *and logs at WARNING*; silence never reads as success
- **`memlora doctor --strict`** — per-subsystem health (schema, FTS5, embeddings, symbols, worker queue); non-zero exit when degraded
- **Architecture enforcement** — `import-linter` layered contracts, guarded by a meta-test so a typo can't silently disable them
- **CI promotion gate** — lint + full suite (incl. `tests/reliability/` failure-injection) on every PR; see [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## Cross-platform (Codex)

The store is platform-neutral — one SQLite DB per logical project, so Claude Code
and Codex working in the same directory share one memory. Project resolution is
**alias-aware**: `C:\repo` and `/mnt/c/repo` resolve to the same store, so memory
follows the checkout across Windows, WSL, and native mounts; for genuinely
different checkout paths, an opt-in `project_identity` key in
`.memlora/config.toml` pins them to one shared store. Codex reads memory through
the registered MCP server (`get_session_state` / `recall`); the capture direction
is **pull-based**, because Codex has no `Stop`-hook equivalent:

- **`memlora codex-sync <project>`** scans `~/.codex/sessions` for rollouts whose
  recorded `cwd` maps to the project and captures the delta through the *same*
  extraction pipeline (a rollout→transcript adapter is the only Codex-specific
  code; delta/dedup/idempotency are shared and unchanged).
- **Automatic at the handoff** — Claude's SessionStart drains pending Codex
  rollouts before building the block, and the MCP server's queue drainer pulls
  new rollouts each cycle, so a live Claude session picks up Codex-side decisions
  without waiting for the next session; on the Codex side, `init` writes an
  `AGENTS.md` instruction + a `ck-sync` skill so Codex pulls at session start.
- **`init` provisions both** — `.mcp.json` (Claude) and `.codex/config.toml`
  (with the server's `cwd` + project env pinned) + `AGENTS.md` (Codex),
  idempotently and without clobbering existing settings.
- **`memlora doctor`** reports a `codex` health line (sessions dir + rollout
  count, or "nothing to sync" — Codex is optional, so its absence is healthy).

A decision made in Codex reaches the next Claude session's block, and
vice versa. The action-point surfaces (CK-1, PreToolUse gate) are Claude-only —
Codex has no per-prompt/per-tool hook — so on Codex the loop degrades to the shared
block + MCP recall.

---

## Interfaces

**MCP tools** (the session block is injected automatically; these are for targeted use):
`recall` · `find_related` · `skeleton` · `get_session_state`

**CLI:**
- `memlora init <project>` — register the project and install the session hooks
- `memlora doctor [--strict] <project>` — subsystem health report
- `memlora codex-sync <project>` — capture Codex CLI sessions for this project
- `memlora install-heads` — install the trained encoder artifacts (salience + cross-encoder ONNX bodies): downloaded from the [`heads-v1` release](https://github.com/KanishkNoir/cognikernel/releases/tag/heads-v1) and sha256-verified, or copied from a local `models/` export when present
- `memlora show <project>` / `memlora reset <project>` — inspect / clear stored memory

---

## Quickstart

```sh
uv sync                      # core (lexical-only)
uv sync --extra embedding    # + dense retrieval (fastembed + numpy)

uv run memlora init .            # register this project + install hooks
uv run memlora doctor .          # subsystem health

uv run memlora install-heads     # optional: trained encoder heads (~270 MB download);
                                 # without them extraction/supersession fall back to
                                 # the legacy/lexical path — everything still works
```

Then start a Claude Code session in the project — the memory block appears
automatically at session start, and decisions are captured when the session ends.

---

## Project layout

```
src/memlora/
  integration/   hooks, CLI, MCP server, session/worker orchestration
  extraction/    sanitize -> classify -> salience -> decision-key pipeline
  delta/         delta-merge + supersession (latest-wins; cross-encoder optional)
  retrieval/     hybrid BM25 + dense -> RRF
  storage/       event-sourced SQLite, FTS5, migrations, render ledger
  embedding/     optional dense vectors (fastembed)
  symbols/       AST skeleton + PageRank graph
  compression/   authority-weighted drop-to-fit budget
  injection/     block template assembly
  model.py       Event — the dependency-free domain primitive
tests/
  unit/          per-subsystem
  reliability/   crash-replay · worker-contention · corrupt-input injection
```

---

## Status

Schema **v18** (includes the Codex cross-platform capture). Architecture
contracts: 3 kept / 0 broken. CI gate: lint + full suite on Ubuntu (3.11/3.12) and
Windows. See `CONTRIBUTING.md` for the Definition of Done that gates every change.
