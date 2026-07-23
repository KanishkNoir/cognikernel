# Benchmarks

This document backs the numbers in the README. It explains **what we measured,
how, and where CogniKernel does and does not help** — including the projects
where it ties or loses, because a benchmark that only reports its wins isn't
worth much.

> **Status: directional, not publication-grade.** These are internal
> measurements from a single evaluator across four multi-session projects
> (3–5 sessions each), one agent-model family. The methodology below is built
> to be bias-resistant and every number ships with its caveat, but this is not
> an independent, peer-reviewed study, and the graded per-session transcripts
> and project fixtures are not published. Read the numbers as *strong
> directional evidence*, not a leaderboard score.

---

## The one-paragraph summary

Across four projects, in a three-arm comparison (CogniKernel vs. flat curated
notes vs. no memory): **file reads are the universal, unambiguous win —
CogniKernel made the fewest reads in every single project, typically 2–4×
fewer.** Token cost is cheaper where memory actually earns its keep (projects
with evolving decisions and cross-file dependencies: **−18% to −23%
price-weighted**) and roughly a wash-to-slightly-worse where the codebase is
small and cheap to re-read. Answer fidelity is decisively better where state
is too large or too evolving to re-derive, and ties or loses where "the code
is the memory."

---

## Why not just report a QA accuracy score?

The standard way to benchmark agent memory is a recall-QA score: ask the agent
questions about earlier sessions, grade the answers. We measure that too — but
it is **not** our headline, because in our own runs it *saturated and
mispredicted behavior*:

- One project scored **92% recall twice** across a real change in the agent
  model — the metric couldn't distinguish the two.
- In another, the agent answered **every** superseded-decision chain correctly
  (7/7) and *then wrote code that violated a graveyarded constraint anyway*.

**Reciting a decision and acting on it are different things.** A high recall
number does not prove the memory changed what the agent *did*. So alongside
recall we measure reads, price-weighted tokens, and a counterfactual
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
Runs execute in the order CogniKernel → flat → cold to keep the evaluator blind
to hindsight.

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
cache creation/read.** They are then **price-weighted** using Anthropic's
per-token rates, because the four token classes cost wildly different amounts:

| Token class | Relative price |
|---|---|
| Cache read | **0.1×** |
| Input (uncached) | 1× |
| Cache write | 1.25× |
| Output | 5× |

This matters enormously: **~95% of every arm's bill is discounted cache-read.**
So a raw token sum is dominated by the cheapest class and *overstates* savings.
In our runs, raw sums made CogniKernel look **28–41% cheaper**; once
price-weighted, the real edge collapsed to the honest numbers below. **We
report the price-weighted number.** If you see a memory tool advertising a
raw-token reduction, ask what it looks like weighted.

---

## Results

### 1. File reads — the universal win

CogniKernel made the fewest file reads in **every** project. Fewer reads means
fewer tool round-trips and more of the context window left for actual work —
the session runs *longer* before compaction, not just cheaper.

| Project | CogniKernel | Flat notes | No memory |
|---|---|---|---|
| Taskflow | **3** | 29 | 14 |
| Relay | **23** | 63 | 63 |
| Toolbelt | **16** | 47 | 53 |
| Conductor | **40** | 89 | 83 |

The Toolbelt case is the cleanest illustration: its ground truth is a ~22-file
public API contract. CogniKernel's injected skeleton let the agent name and
update the affected surface **~90% unaided**; both flat arms recovered the same
contract only by **re-opening 47–53 files** (~10–15% unaided).

### 2. Price-weighted token cost

Cheaper where memory matters; a wash-to-worse where the code is cheap to
re-read. Shown vs. the better flat/cold baseline:

| Project | CogniKernel vs. baseline | Read as |
|---|---|---|
| Relay (recall/evolution) | **−23%** | clear win |
| Toolbelt (dependency) | **−18%** | clear win |
| Taskflow (small/balanced) | −4% | wash |
| Conductor (impl-heavy) | **+13%** | **CogniKernel cost more** |

Conductor is the honest counter-result: on an implementation-heavy project
where little needs recalling, CogniKernel's injected block and its caching add
the dearest token class (cache-write at 1.25×) without a recall payoff — so it
runs *more* expensive than a cold start. We report this because it's true.

### 3. Answer fidelity / correctness

Percentage of demanded decisions carried faithfully into the produced code
(re-baselined to each arm's *own* recorded decisions — see methodology):

| Project | CogniKernel | Flat notes | No memory |
|---|---|---|---|
| Relay | **92** | 77 | 0 |
| Toolbelt | **96** (90 unaided) | 95 (~10 unaided) | 98 (~15 unaided) |
| Taskflow | 83 | **97** | 95 |
| Conductor | 89 | **100** | 89 |

The Relay progression — **0 → 77 → 92** (no memory → flat → CogniKernel) — is
the headline: on a project whose decisions *evolve and supersede* across five
sessions, the no-memory arm scored **zero** (it rebuilt from scratch and got
the evolved values wrong), flat notes recovered most of it by re-reading its
own code, and CogniKernel led. But on **Taskflow and Conductor, CogniKernel
loses** to flat/cold — see below.

---

## Where CogniKernel does *not* help

This is the part most benchmark docs omit.

- **Small, fully re-readable projects (Taskflow).** When the whole working set
  fits in a few files, *the code is the memory* — an agent just re-reads it and
  gets the current state for free. Structured memory adds overhead without a
  recall payoff, and can trail a careful `CONTEXT.md` (83 vs. 97).
- **Implementation-heavy work (Conductor).** When the task is "write code that
  upholds these invariants" rather than "recall what we decided," there's little
  to recall; CogniKernel ties on fidelity (89 = 89 vs. cold) and costs **+13%**
  more in weighted tokens.
- **Acting on memory costs mistakes.** CogniKernel *commits* to recalled
  decisions, so when a recalled fact is wrong it acts on the wrong fact. In the
  suite it carried the most genuine cross-session mistakes (5) precisely because
  it acts on memory instead of re-deriving. This makes **supersession
  correctness a safety property**, not a nicety: in counterfactual tests, a
  stale memory line was followed into the code ~100% of the time — a stale
  memory is worse than no memory.

**The honest thesis:** structured memory earns its keep when project state is
*too large, too evolving, or too long-lived to re-derive cheaply*. On small or
static work, flat notes or plain re-reading are competitive or better. The one
result that holds everywhere is **read reduction**.

---

## Methodology — how we keep this honest

The evaluation is built to resist the ways a self-run benchmark flatters
itself:

- **Recall ≠ honor.** We grade the *produced artifact* against the decisions
  demanded, not the agent's ability to recite them.
- **Counterfactual attribution.** For load-bearing facts we run
  PRESENT / ABSENT / CORRUPTED on frozen state; *memory lift = honor(present) −
  honor(absent)*. If the agent succeeds with the fact ABSENT, it re-derived it
  and memory earns no credit. Result: only a minority of facts are genuinely
  memory-caused, but where lift exists it tends to be total (0→1) — exactly the
  facts with no code footprint (rationale, superseded values, dead ends).
- **Re-baseline to each arm's own decisions**, never a template. On one run this
  choice alone swung the score 5 points (16/27 template-strict vs. 21.25/27
  re-baselined); a legitimate divergence is not a mistake.
- **Only genuine mistakes are deducted** — contradictions are classified as
  *demanded* / *useful re-decision* / *mistake*.
- **Price-weight tokens** (above); lead with the robust metric (reads), not the
  fragile one (weighted tokens, which flips sign across projects).
- **Tiered grading**: deterministic match first, then a cross-family LLM judge
  on the disputed remainder, then human audit — with the judge-overturn rate
  monitored.
- **Adverse findings published** (Conductor +13%, Taskflow fidelity loss,
  CogniKernel's own mistakes) — a rubric was pre-committed and never tuned to
  make a hypothesis pass.

### Limitations

- Single evaluator; n≈2 repeats per counterfactual cell.
- 3–5 sessions per project — shorter than the memory half-life, so long-horizon
  decay is under-tested.
- All projects are Python-backend-flavored; one agent-model family.
- The multi-session project fixtures and graded transcripts are **not
  published**, so these exact numbers are **not turn-key reproducible** from
  this repo. The scoring harnesses live in `scripts/` (e.g. `bench_honor.py`,
  `probe_replay.py`), but they read project data kept private.

---

## Reproducing the shape of this

You can't rerun our exact projects, but you can measure the mechanism on your
own multi-session work:

1. Run a real multi-session project **with** CogniKernel and note file-read
   counts and token telemetry (`cognikernel doctor` surfaces cache stats).
2. Run a comparable project **without** it (or with a hand-kept `CONTEXT.md`).
3. Compare **reads first** — that's where the effect is largest and least
   ambiguous — then price-weight the tokens (0.1× cache-read, 1.25× cache-write,
   5× output) before comparing cost.

If your work looks like Relay or Toolbelt (evolving decisions, cross-file
contracts), expect a clear win. If it looks like Taskflow (small, re-readable),
expect a wash on tokens and a win only on reads.
