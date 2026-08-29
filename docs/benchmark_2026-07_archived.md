# Benchmarks

This document backs the numbers in the README. It explains **what we measured,
how, and where CogniKernel does and does not help.**

> **Status: directional, not publication-grade.** These are internal
> measurements from a single evaluator across four multi-session projects
> (3–5 sessions each), one agent-model family. The methodology below is built
> to be bias-resistant and every number ships with its caveat, but this is not
> an independent, peer-reviewed study, and the graded per-session transcripts
> and project fixtures are not published. Read the numbers as *strong
> directional evidence*, not a leaderboard score.
>
> **Two measurement vintages, two agent models.** Correctness (QA) and the
> causal honor suite were **freshly measured on the current stack (July 2026)**
> after the encoder re-tuning and edit-time-binding fixes; the efficiency figures
> are carried from the earlier same-script run. Critically, those runs used
> **two different agent models** (the earlier three-arm benchmark on one, the
> later CogniKernel-arm runs on a newer one). Because token *consumption* is
> highly model-dependent — different tokenizers, different verbosity — **we do
> not publish precise per-project token numbers here**; they would conflate two
> models. We report the token result as a **direction and a range** and defer
> exact figures to a single-model re-benchmark. File-read counts (a behavioral
> metric, far less model-sensitive) and correctness are reported concretely.

---

## The one-paragraph summary

Across four projects, in a three-arm comparison (CogniKernel vs. flat curated
notes vs. no memory): **file reads are the universal, unambiguous win —
CogniKernel made the fewest reads in every single project, typically 2–4×
fewer.** On correctness, CogniKernel **wins decisively where decisions evolve
across sessions** (Relay: 92 vs. 70 vs. 0) and **ties the flat arms where the
codebase is small or cheap to re-read** (Taskflow ~96 vs. 97; Conductor ~98 vs.
100) — with, in those tie cases, the added benefits of fewer reads, zero manual
upkeep, and the fewest genuine cross-session mistakes of any arm (1, vs. 4 for
flat notes). On token cost, CogniKernel runs roughly **30–40% leaner than the
auto-memory arm** on the recall/dependency projects and is about a wash on
implementation-heavy work (precise figures deferred to a single-model
re-benchmark — see the note above).

---

## Why not just report a QA accuracy score?

The standard way to benchmark agent memory is a recall-QA score: ask the agent
questions about earlier sessions, grade the answers. We measure that too — but
it is **not** our only headline, because in our runs it *saturated and
mispredicted behavior*:

- One project scored **92% recall twice** across a real change in the agent
  model — the metric couldn't distinguish the two.
- In another, the agent answered **every** superseded-decision chain correctly
  (7/7) and *then wrote code that violated a graveyarded constraint anyway*.

**Reciting a decision and acting on it are different things.** So alongside
recall we measure reads, token cost, and a counterfactual
"does-the-memory-cause-the-behavior" signal (see
[Methodology](#methodology--how-we-keep-this-honest)). This matches the public
finding that systems near-saturating LoCoMo-style recall can still fail in
agentic settings.

---

## Setup

**Three arms**, identical multi-session scripts, only the memory layer changes:

| Arm | Memory |
|---|---|
| **CogniKernel** | the full system — injected block, hooks, skeleton, supersession |
| **Flat notes** | a hand-maintained `CONTEXT.md` / native auto-memory — what a careful developer already does |
| **No memory** | cold start every session (the control) |

The primary comparison is **CogniKernel vs. flat notes** — it isolates the
value *over what disciplined developers already do by hand*, not over nothing.
Runs execute CogniKernel → flat → cold to keep the evaluator blind to hindsight.

**Four projects**, each overloading one axis:

| Project | Sessions | Stresses |
|---|---|---|
| **Taskflow** | 3 | balanced baseline (small, re-readable) |
| **Relay** | 5 | recall under decision *evolution* — ~18 confusable facts, 3 supersession chains, 3 abandoned-approach probes |
| **Toolbelt** | 5 | cross-package *dependency* contracts — a 22-file public API evolved across sessions, including a breaking change that cascades |
| **Conductor** | 5 | implementation-quality invariants — 12 reliability rules to uphold while building |

---

## The cost model (and why "raw tokens" lie)

Every token is counted from the agent's real API telemetry — not estimated —
across four terms: **injection tokens + all read tokens + output tokens +
cache creation/read.** They should be **price-weighted** using Anthropic's
per-token rates, because the four classes cost wildly different amounts:

| Token class | Relative price |
|---|---|
| Cache read | **0.1×** |
| Input (uncached) | 1× |
| Cache write | 1.25× |
| Output | 5× |

This matters: **~95% of every arm's bill is discounted cache-read.** A raw token
sum is dominated by the cheapest class and *overstates* savings. The ~30–40%
raw reduction on the recall/dependency projects shrinks once price-weighted —
still a real edge there, and roughly a wash on implementation-heavy work. **If a
memory tool advertises a raw-token reduction, ask what it looks like weighted.**

---

## Results

### 1. File reads — the universal win

CogniKernel made the fewest file reads in **every** project. Fewer reads means
fewer tool round-trips and more of the context window left for actual work — the
session runs *longer* before compaction, not just cheaper.

| Project | CogniKernel | Flat notes | No memory |
|---|---|---|---|
| Taskflow | **3** | 29 | 14 |
| Relay | **23** | 63 | 0 † |
| Toolbelt | **16** | 47 | 53 |
| Conductor | **40** | 89 | 83 |

† No-memory Relay did *0* reads not because it was efficient but because it
**collapsed** — with no memory of the evolving decisions it rebuilt from scratch
and produced incoherent output (0% correctness). Zero reads there is the failure
mode, not a win; the meaningful comparison is CogniKernel 23 vs. flat notes 63.

The Toolbelt case is the cleanest illustration of the mechanism: its ground
truth is a ~22-file public API contract. CogniKernel's injected skeleton let the
agent name and update the affected surface **~90% unaided**; both flat arms
recovered the same contract only by **re-opening 47–53 files** (~10–15% unaided).

### 2. Token cost

**Direction, not a precise table** — see the vintage/model note at the top for
why we're not publishing exact per-project token figures yet (the runs span two
agent models with different token footprints).

What holds across the runs we have:

- On the **recall (Relay) and dependency (Toolbelt) projects** — where memory
  earns its keep — CogniKernel used roughly **30–40% fewer raw tokens than the
  auto-memory arm**, and came out ahead price-weighted as well (the honest
  price-weighted edge is smaller than the raw figure, because ~95% of every
  session's tokens are discounted cache-read — see the cost model above).
- On **small / implementation-heavy work (Taskflow, Conductor)** it's roughly a
  **wash**: little needs recalling, so CogniKernel's injected block and caching
  add cost without a recall payoff — on one project a no-memory run was actually
  a touch cheaper. We report that honestly.

Exact per-project numbers will be published after a **single-model
re-benchmark** so every arm is measured on the same agent.

### 3. Answer fidelity / correctness

Percentage of demanded decisions carried faithfully into the produced code,
re-baselined to each arm's *own* recorded decisions (see methodology).
**Current stack, freshly measured:**

| Project | CogniKernel | Flat notes | No memory |
|---|---|---|---|
| Relay (evolving decisions) | **92** | 70 | 0 |
| Toolbelt (dependency) | **97** (90 unaided) | 95 (~10 unaided) | 98 (~15 unaided) |
| Taskflow (small/static) | 96 | **97** | 95 |
| Conductor (impl-heavy) | 98 | **100** | 89 |
| **Genuine mistakes (total)** | **1** (+1 partial) | 4 | 2 (+1 total failure) |

Two things to read here:

- **Relay is the discriminator: 0 → 70 → 92** (no memory → flat → CogniKernel).
  On a project whose decisions *evolve and supersede* across five sessions, the
  no-memory arm scored **zero** (rebuilt from scratch, got the evolved values
  wrong), flat notes recovered most by re-reading their own code, and CogniKernel
  led by a wide margin. This is where structured memory is decisive.
- **On Taskflow and Conductor the arms essentially tie** (96 vs. 97; 98 vs.
  100 — all with zero or near-zero mistakes). Where the working set is small or
  the invariants are re-derivable, *the code is the memory*, so flat notes are
  competitive on correctness. CogniKernel's edge on those projects is not higher
  fidelity — it's the **fewest mistakes of any arm (1 vs. 4)**, the fewest reads,
  and zero manual upkeep.

### 4. Memory-to-action (the causal signal)

Beyond recall, the honor suite (259 cells) toggles a fact PRESENT / ABSENT /
CORRUPTED on frozen state and grades the produced code:

- **Honor rate 1.00** — when a fact reaches the agent through the block, it's
  obeyed in code.
- **Chain-latest lift +0.81** — superseded-value chains (the critical class) are
  the facts memory most changes: with the latest value present the agent uses it,
  absent it does not. Flat notes served a *stale* chain value that was followed
  into code.
- Only a minority of facts are genuinely memory-caused (≈33% load-bearing), but
  where lift exists it tends to be total (0→1) — exactly the facts with no code
  footprint (rationale, superseded values, dead ends).

### 5. What each subsystem measurably contributes

The honor suite and probe replay attribute effects to specific subsystems
(current stack):

| Subsystem | Measured effect | Flat-notes equivalent |
|---|---|---|
| **Supersession / chains** | all 3 chains answered at their *latest* value live; chain-latest lift +0.81 | served a **stale** chain value; needed manual note edits to update |
| **Extraction capture** | 11/11 Taskflow gold decisions captured automatically | hand-notes covered ~¼ ("no record of the cache decision") |
| **Injection (availability)** | the block is injected every session | flat notes violated their *own* recorded rule live — yet honored it 2/2 when the same note was force-fed into context. Flat memory fails at **delivery**, not comprehension |
| **Edit-time binding (K2 / CK-1)** | surfaced the contradicting prior decision *at the moment of the offending edit* on 4/4 (prompt-time) and 3/4 (edit-time) historical mistakes | no equivalent — flat memory can't surface a prohibition at the point of action |
| **Skeleton (structural)** | 22-file contract carried ~90% unaided, read-free | re-read 47–53 files |
| **Always-on (variance)** | structured capture every session | no-memory self-documentation ranged **0→95** across projects — a lottery |
| **Zero maintenance** | automatic | manual upkeep (one no-memory run wrote its CLAUDE.md 51 times in a single session) |

One concrete instance: on Conductor, a reliability invariant CogniKernel had
captured led the agent to **catch a real PII leak** in code it was about to
ship — a memory-caused correction, not a re-derivation.

---

## Where CogniKernel does *not* help (and its live weaknesses)

- **Small, re-readable projects (Taskflow) and implementation-heavy work
  (Conductor).** CogniKernel *ties* the flat arms on fidelity here rather than
  beating them — when the whole working set fits in a few files or the task is
  "uphold these invariants" rather than "recall what we decided," re-reading the
  code recovers the state for free. On Conductor it also costs a touch more in
  weighted tokens than a cold start. Its value on these projects is efficiency
  and zero-maintenance, not correctness.
- **Corruption is followed — supersession correctness is a safety property.**
  In counterfactual tests a *stale* memory line was followed into code ~83% of
  the time (corruption-flip 0.83). A wrong memory is acted on, so keeping the
  store correct (latest-wins supersession) is safety-critical, not cosmetic — a
  stale memory can be worse than none. This is an open hardening area
  (storage-side supersession).
- **THREAD-slot noise** is CogniKernel's one remaining mistake mechanism (the
  single genuine miss on the current stack traces to it) and a source of a
  low-grade "active thread" false-positive rate. It's the top precision fix on
  the roadmap.

**The honest thesis:** structured memory earns its keep when project state is
*too large, too evolving, or too long-lived to re-derive cheaply* — there it
wins decisively (Relay). On small or static work it *ties* on correctness while
still cutting reads and maintenance. The one result that holds everywhere is
**read reduction**.

---

## Methodology — how we keep this honest

The evaluation is built to resist the ways a self-run benchmark flatters itself:

- **Recall ≠ honor.** We grade the *produced artifact* against the decisions
  demanded, not the agent's ability to recite them.
- **Counterfactual attribution.** For load-bearing facts we run
  PRESENT / ABSENT / CORRUPTED on frozen state; *memory lift = honor(present) −
  honor(absent)*. If the agent succeeds with the fact ABSENT, it re-derived it
  and memory earns no credit.
- **Re-baseline to each arm's own decisions**, never a template. On one run this
  choice alone swung the score ~5 points; a legitimate divergence is not a
  mistake.
- **Only genuine mistakes are deducted** — contradictions are classified as
  *demanded* / *useful re-decision* / *mistake*.
- **Price-weight tokens** (above); lead with the robust metric (reads).
- **Tiered grading**: deterministic match first, then a cross-family LLM judge on
  the disputed remainder, then human audit — judge-overturn rate monitored.
- **Adverse findings published**: the Conductor token wash, the fidelity *ties*
  (rather than claiming wins), the corruption-follow rate, and the one remaining
  mistake mechanism. The rubric was pre-committed and never tuned to pass a
  hypothesis.

### Limitations

- Single evaluator; n≈2 repeats per counterfactual cell.
- 3–5 sessions per project — shorter than the memory half-life, so long-horizon
  decay is under-tested.
- All projects are Python-backend-flavored; one agent-model family.
- **Mixed vintage:** QA + honor are current-stack (post-retuning, July 2026);
  reads + raw-token totals are carried from the earlier same-script run (flat
  arms not re-run on the newer scripts).
- The multi-session project fixtures and graded transcripts are **not
  published**, so these exact numbers are **not turn-key reproducible** from this
  repo. The scoring harnesses live in `scripts/` (e.g. `bench_honor.py`,
  `probe_replay.py`), but they read project data kept private.

---

## Reproducing the shape of this

You can't rerun our exact projects, but you can measure the mechanism on your own
multi-session work:

1. Run a real multi-session project **with** CogniKernel and note file-read
   counts and token telemetry (`cognikernel doctor` surfaces cache stats).
2. Run a comparable project **without** it (or with a hand-kept `CONTEXT.md`).
3. Compare **reads first** — the largest, least ambiguous effect — then
   price-weight the tokens (0.1× cache-read, 1.25× cache-write, 5× output) before
   comparing cost.

If your work looks like Relay or Toolbelt (evolving decisions, cross-file
contracts), expect a clear win. If it looks like Taskflow or Conductor (small or
implementation-heavy), expect a tie on correctness and a win on reads and upkeep.
