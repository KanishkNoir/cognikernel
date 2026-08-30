# Benchmark method

How the numbers in [`benchmark.md`](benchmark.md) were produced — the criteria, the
validity gates, what a judge decides versus what arithmetic decides, and the rules
that keep the reporting honest.

**Written so the method can be attacked directly.** If you think a result is wrong,
this is the document to argue with.

> **What is not here.** The graded transcripts, project fixtures and run sheets stay
> private — they contain session data that is not ours to publish. This document is
> the complete *method*; `benchmark.md` is the complete *finding set*. Neither is a
> selection from a larger private version.

---

## 1. Design

Arms build the **same system** from the **same prompt sequence**, across multiple
sessions. The arms differ **only** in the memory layer — prompts, order, and agent
model are identical.

| Arm | Memory available |
|---|---|
| **CK** | CogniKernel — injected block, MCP `recall` / `find_related` / `skeleton`, hooks |
| **auto** | The agent's native auto-memory (`~/.claude/projects/<proj>/memory/`) |
| **stock** | None provided — it writes its own notes (`docs/adr/`, `WORKLOG.md`, …) |

Probes in later sessions test whether decisions from earlier sessions survive. Two
probe classes carry most of the signal:

- **Supersession chains** — a decision announced in S1 and *changed* mid-run. Only
  the latest value is correct; recalling the original is a failure, not a partial.
- **Graveyard probes** — re-tempting an approach the project explicitly rejected.

A hand-maintained `CONTEXT.md` arm was replaced by native auto-memory, which is the
stronger and more current comparator.

---

## 2. Validity gates — run **before** any metric

A clean-looking table over incomparable arms is the failure mode that sinks a study.
Every gate is cheap, and all of them precede any inference spend.

| Gate | How | Kill criterion |
|---|---|---|
| Do all arms build and pass? | `pytest` in each project | if one arm is broken, its efficiency numbers are meaningless |
| Same prompts, same order? | parse prompts from transcripts | large divergence = arms not comparable |
| Same probe count? | same script | note any missing probe; report **rates**, never sums |
| Comparable S1 ground truth? | §3 output | an arm with fewer settled slots started weaker |
| Answer-length parity? | max chars per arm | one arm much longer = verbosity advantage |
| Truncation in judge input? | compare max chars to the cap | any truncation invalidates that probe |
| Right tool version? | query the store for a post-fix signature | assumption ≠ verification |
| Identity leakage? | grep the source trees | a codebase judge must not learn which arm it is reading |

> **Trap:** `pytest --collect-only` is authoritative for test counts. Counting
> `def test_` with grep **misses `async def test_`**. That bit one run twice — once
> undercounting by 6×, and once still disagreeing with pytest after a "fix".

---

## 3. Ground truth is derived **per arm**, never from a template

This is the single most important choice here, and it comes from a measured failure.
Scoring against a run sheet's template produced **16/27**; scoring against what the
model *actually decided* produced **21.25/27**. The template assumed decisions the
models never made, so faithful recall of a different-but-valid choice was being
marked wrong.

So for each arm, one deriver model reads **that arm's own Session-1 transcript** and
extracts its decisions per slot as `{value, stated, rationale, evidence}`.
`stated: false` when the transcript never settles a slot — **the extractor is
forbidden from filling gaps with conventional defaults.**

**One deriver for all arms**, so no arm is graded against a differently-derived
standard. Slots an arm never established are **excluded, not scored 0**, so a
tester's omission is never charged to an arm's memory.

**Acknowledged consequence:** this makes grading *not fully blind* — the baseline
comes from the same arm being graded. Unavoidable given per-arm ground truth, and
stated rather than hidden.

---

## 4. Metric taxonomy — what is judged, what is arithmetic

**Roughly half the reported metrics involve no LLM at all.** This matters: the
mechanical half is unfalsifiable by judge bias and recomputable by anyone holding the
transcripts.

### Mechanical — AST, transcript parsing, or arithmetic

| Metric | Source |
|---|---|
| Read-like calls (Read/Grep/Glob), per session | transcript `tool_use` blocks |
| Orientation reads — reads before the first line of code | same, cut at the first Write/Edit |
| Token usage by class (input / cache-write / cache-read / output) | transcript `usage` records |
| Price-weighted cost, and dollar cost | usage × published rates |
| Cost per WHAT-point, per correct chain answer | above ÷ judged score |
| Memory tax (scaffolding / authoring / recall) | file sizes + tool-result payloads |
| Injected-block size per session | SessionStart hook attachment |
| File-operation trace per prompt; churn; re-reads of own output | transcript, ordered |
| Design metrics: duplication/kLOC, function length, complexity, nesting, docstring coverage, test ratio | Python AST |
| Test pass counts | `pytest --collect-only` |

### Judged — LLM

| Metric | Scale |
|---|---|
| WHAT recall | 1 / 0.5 / 0, median of 3 |
| WHY recall | 1 / 0 / null (null when the baseline records no rationale) |
| `cited_value` | free text — **diagnostic, not scored** |
| `answered_wrong_subject` | bool |
| `contradicts_standard` | bool |
| Per-answer engineering quality | 1–5 |
| Whole-codebase pairwise, 6 axes | A / B / tie + confidence 1–5 |
| Redundant-question detection | redundant / partial / legitimate |

### Derived — no judge, computed from judged inputs

Chain-probe accuracy · graveyard accuracy · per-session WHAT rate (the decay curve) ·
inter-judge agreement · per-judge robustness.

**Recall counts through prose, code, *or* tool calls equally.** An earlier prose-only
pass understated every arm — one by 21 points, because it demonstrated recall by
writing the code rather than narrating it.

---

## 5. Judges — assignment and controls

Judges are assigned **per metric**, not all judges on everything. Recall is the
headline number and needs numeric precision plus supersession reasoning, so it gets
three strong reasoners and a **median** vote — with three judges the median *is* the
majority whenever two agree, and one outlier cannot drag a probe. Preference-style
tasks (codebase pairwise) get more model families and both A/B orders instead.

A materially smaller model is kept **out** of the recall panel — it would dilute a
consensus it cannot carry — while remaining useful where diversity beats precision.

| Control | Implementation |
|---|---|
| Inter-judge agreement | reported per arm, never averaged away |
| Disagreement diagnosis | every judgment returns `cited_value`, so a split can be classified as rubric variance vs judges reading *different values* out of the same answer |
| Judge bias | tracked against the median (strict vs lenient), contained by the median |
| Robustness | the full result recomputed **under each judge alone** |
| Position bias | every codebase pair judged in **both A/B orders** |
| Retry integrity | a call counts only on success; failures retried under per-model rate limiting |

**What judges see:** the arm's own baseline for the slot, a supersession note only if
the probe's session ≥ the change session, the probe question, and the answer text.
Judges never see the raw session JSONL.

---

## 6. Traps that produced wrong numbers

Each of these silently produced a plausible result that was false. They are listed
because a replication will hit them too.

| Trap | Wrong result | Fix |
|---|---|---|
| Raw win counts at uneven *n* | a 45–16–0 scoreline that inverted once corrected | **head-to-head rate per pair**, print the n, and check the rate is stable as the sample grows |
| The injected block arrives as a `type: "attachment"` record (hook stdout), not a message | injection cost reported as **0 tokens/session** | parse `attachment` records; the block is in `stdout.hookSpecificOutput.additionalContext` |
| Memory artifacts detected by a name list (`CLAUDE.md`, `docs/adr/`, …) | an arm's hand-maintenance reported as **0 writes, 0 recall** — it had written `WORKLOG.md` | detect **any agent-written `.md`**, excluding install scaffolding |
| "More files/tests = better codebase" | a size artifact read as a quality win | **size-normalised** AST metrics — this inverted a judged verdict |

> **Rule:** when a metric returns exactly zero for one arm and non-zero for others,
> verify against the filesystem or the raw transcript before reporting it. A clean
> zero is more often a detector gap than a real finding.

---

## 7. Reporting rules

- **Rates, not raw counts**, whenever sampling can be uneven. Print the n.
- **Report the vector, never the average.** Overall, chain, graveyard and thread
  probes move independently — a single number hides which axis moved.
- **Price-weighted cost, not raw tokens.** They can disagree, and only weighted is
  honest: one arm used ~13% *fewer* total tokens yet cost ~5% *more*, because its mix
  carried more cache-writes and output.
- **Report adverse findings**, including the tool losing an axis outright.
- **Separate "measured" from "inferred."** Verify judge claims in source before
  repeating them.
- **State what is correlational.** Without a counterfactual arm there is no causal
  claim.
- **Exclude a broken instrument rather than reporting it.** Metrics dropped for this
  reason are named in `benchmark.md` §5.
- **Correct the record in place** when a number turns out wrong, and say it was wrong.

---

## 8. Known weaknesses

1. **Not fully blind on recall** — the baseline is derived from the arm being graded (§3).
2. **Single-run, no repeats, no confidence intervals.**
3. **Judge independence is uneven** across vintages — an external multi-family panel
   in one, same-family judges in another. These are not equivalent claims and should
   not be presented as one number.
4. **Quality scores come from a single judge** — treat as an ordinal comparison
   between arms, not an absolute grade.
5. **`stock` is not a clean no-memory control.** It writes its own notes and
   interrupts the engineer; it is better described as *unassisted human-in-the-loop*.
6. **Projects stress different things.** Nothing here transfers to another axis
   without re-running.
