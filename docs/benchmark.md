# Benchmarks

What we measured, how, and **where CogniKernel does and does not help.**

> **Status: internal, directional, single-evaluator.** Four multi-session projects,
> one agent model, one run per arm, no repeats and no confidence intervals. The
> method is built to resist self-flattery — per-arm ground truth, adverse findings
> published, every metric carrying its denominator — but this is not an independent
> study. Read it as strong directional evidence, not a leaderboard.
>
> Prior-vintage results (July 2026, two agent models, includes the causal honor
> suite) are archived at `benchmark_2026-07_archived.md`.

---

## Configuration

Everything below is one uniform configuration. This matters: the previous vintage
spanned **two agent models** and therefore had to withhold token figures entirely.
This one does not.

| | |
|---|---|
| **Agent model** | `claude-sonnet-5` — **every arm, every project, every session** |
| Harness | Claude Code CLI 2.1.233 – 2.1.251 |
| CogniKernel | schema v20, `extractor = v2-broad`, `token_budget` 3500, `skeleton_budget` 600 |
| Arms | **CK** (CogniKernel) · **auto** (Claude Code native auto-memory) · **stock** (no memory provided) |
| Projects | Relay 5 sessions · Conductor 5 · Toolbelt 5 · Taskflow 3 |
| Probe scoring | per-arm baselines; median of 3 external judges (Relay v1) or single Opus judge (v2 re-score) |
| Cost model | price-weighted: cache-read 0.1×, input 1×, cache-write 1.25×, output 5× |

The three arms differ **only** in the memory layer. Prompts, order, and model are identical.

---

## 1. Cross-platform: the one structural win

A CogniKernel store is a portable artifact. Native auto-memory is not.

One recall session per project was run in **Codex (gpt-5.5)** against the store
**Claude Code had built**, scored on the same rubric and the same gold facts:

| Project | Probes | Correct |
|---|---|---|
| Taskflow | 4 | **4.0** |
| Relay | 3 | **3.0** |
| Conductor | 4 | 3.5 |
| Toolbelt | 4 | 3.0 |
| **Total** | **15** | **13.5 (0.90)** |

**Cross-platform retention ≈ 0.93** against the same facts scored on Claude Code.

**The flat arms score 0 here by construction** — auto-memory lives under `~/.claude`,
which Codex never reads, and a self-written `CLAUDE.md` is not loaded by Codex. This
is not a close comparison; it is a difference in kind. The entire CK-minus-cold gap
is delivered to a second agent that never saw the project built.

One case where cross-platform recall **beat the origin platform**: Relay's T1, where
Codex's targeted `recall` named the Session-1 router thread *and* distinguished a
newer thread, out-performing the Claude-side session slot.

---

## 2. Cross-session recall — report the vector, never the average

The run sheets require this (`richness_rubric.md` §1: *"Each is reported separately.
Never average them."*), and the suite shows why: on Conductor the CK-vs-auto delta is
**+5.9** across all probes but **−16.7** on chain probes alone. A single number hides
which axis moved.

**CK vs auto, by dimension** (positive = CogniKernel ahead):

| Project | Axis | all probes | chain | **graveyard** | thread |
|---|---|---|---|---|---|
| **Relay** | decisions *evolve* across 5 sessions | **+18.1** | **+16.7** | **+66.7** | 0.0 |
| Conductor | 12 quality invariants | +5.9 | −16.7 | 0.0 | — |
| Toolbelt | self-authored API contract | −9.3 | −10.0 | 0.0 | −66.7 |
| Taskflow | small, re-readable | −3.6 | 0.0 | 0.0 | −100.0 |

Absolute rates:

| Project | CK | auto | stock |
|---|---|---|---|
| Relay | **89%** | 71% | 65% |
| Conductor | **98%** | 92% | — |
| Toolbelt | 91% | **100%** | — |
| Taskflow | 86% | 89% | **93%** |

### What holds

**Relay is the only project where CogniKernel leads on every dimension**, and it is
the only one with **three supersession chains and three graveyard entries**. Its
graveyard margin is the largest single result in the suite: **CK 100% (3/3) vs auto
33%** — auto re-proposed approaches the project had explicitly abandoned.

**The honest boundary:** structured memory repays its overhead when a project
accumulates *multiple* evolving decisions and *multiple* abandoned approaches. With
one of each, it does not. Toolbelt and Taskflow are genuine losses, published as such.

---

## 3. Passive effects — reads and tokens

Mechanical, from API telemetry and transcript parsing. No judge involved.

### Orientation reads — the one efficiency result that holds everywhere

Reads before the first line of code: the cost of working out where you are.

| Project | CK | auto | **delta** |
|---|---|---|---|
| Relay | 27 | 39 | **−30.8%** |
| Taskflow | 17 | 20 | **−15.0%** |
| Conductor | 97 | 104 | **−6.7%** |
| Toolbelt | 90 | 92 | **−2.2%** |

**Lower on 4 of 4.** This is what an injected block is for, and it is the only
efficiency metric that wins on every project.

### Total reads and cost

| Project | reads Δ | raw tokens Δ | **weighted cost Δ** |
|---|---|---|---|
| Relay | **−38.1%** | −17.5% | **−15.1%** |
| Toolbelt | +2.9% | **−17.9%** | **−13.3%** |
| Conductor | +4.5% | +1.1% | +5.3% |
| Taskflow | +6.7% | +9.9% | +32.5% |

Cheaper on Relay and Toolbelt; more expensive on Conductor and Taskflow.

**Raw and weighted can disagree, and only weighted is honest.** On Taskflow CK used
**13% fewer total tokens than stock yet cost 5% more** — its mix carries more
cache-writes and output (priced 1.25× and 5×) and fewer cache-reads (0.1×). Anyone
reporting raw tokens would call that a win. It is not.

### Memory cost, split by direction

| Project | CK authoring | auto authoring | CK recall | auto recall |
|---|---|---|---|---|
| Relay | **0** | 47 | 50,028 | 46,185 |
| Taskflow | **0** | 7 | 10,489 | 1,004 |
| Conductor | **0** | 0 | 21,571 | 25 |
| Toolbelt | 11 | 29 | **25,309** | 28,885 |

CogniKernel spends **zero authoring calls on 3 of 4 projects** — extraction is
automatic. Auto-memory spent up to **47** calls hand-maintaining notes. The trade is
that CK generally pays more to *recall*, because an injected block is charged every
session whether it is used or not. On Toolbelt it is cheaper on both.

---

## 4. Where CogniKernel does not help

- **Small or re-readable projects.** Taskflow (3 sessions, 1 chain): CK is last on
  recall and 32.5% more expensive. When the working set fits in a few files, reading
  them is cheaper than remembering them.
- **Precise self-authored symbols.** Toolbelt: auto scored **100%** by re-reading
  when unsure; CK answered from memory more often (20/27 vs 17/27) and paid for two
  misses. Confidence without verification has a cost.
- **THREAD-slot noise is the dominant defect.** CK loses thread probes on Toolbelt
  (−66.7) and Taskflow (−100), and Conductor's T1 was never captured at all.
  Root-caused: a *user prompt* is captured as a `THREAD_OPEN` carrying `user_stated`
  authority and outranks the genuine thread, which was truncated to a fragment.
  **Toolbelt's entire deficit is this bug** — excluding thread probes, CK scores 98%
  against auto's 100%. It is an extraction-precision defect, not a ranking one.
- **Stale memory is followed.** A wrong stored value gets acted on. Supersession
  correctness is a safety property, not cosmetics.

---

## 5. Method, and its known weaknesses

- **Per-arm ground truth, never a template.** Each arm is graded against the
  decisions *it* made in Session 1. Template scoring penalises faithful recall of a
  different-but-valid choice — worth about 5 points on a prior run.
- **Slots an arm never established are excluded**, not scored 0, so a tester's
  omission is never charged to an arm's memory.
- **Recall counts through prose, code, or tool calls equally.** An earlier
  prose-only pass understated every arm — auto by **21 points**, because it
  demonstrated recall by writing code 14 times out of 28 rather than narrating it.
- **Judge independence is uneven.** Relay v1 used a four-family external panel
  (Kimi K3, DeepSeek V4 Pro, GLM 5.2, Qwen3.8); the v2 re-score and the other three
  projects used same-family judges. Every score carries a `cited_value` for audit,
  but these are not equivalent claims and should not be presented as one number.
- **No causal attribution in this vintage.** Sections 2 and 3 are correlational. The
  counterfactual honor suite (PRESENT / ABSENT / CORRUPTED, executed in an empty
  directory so nothing can be re-derived from code) is in the archived July results
  and was not re-run here.
- Metrics **excluded** rather than reported once the instrument proved wrong:
  `bypass_to_update` (meaningless on greenfield — every first Write counts as one),
  internal-import coupling (missed relative imports), and test-suite ratios on
  projects where the arms wrote no tests.

- **The fixtures and graded transcripts are not published**, so these exact
  numbers are **not turn-key reproducible from this repo**. The four project
  run sheets, the scoring harness, the per-project vectors and the full method
  write-up are kept private alongside the session transcripts they grade, which
  contain project data that is not ours to publish. What is stated above —
  every denominator, every adverse result, and the excluded-metric list — is
  the complete set of findings, not a selection from a larger private set.
