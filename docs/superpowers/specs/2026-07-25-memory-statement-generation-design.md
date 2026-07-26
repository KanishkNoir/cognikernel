# Memory statement generation — strengthening a measured weakness

**Status:** design approved, not yet implemented
**Date:** 2026-07-25
**Framing:** this branch targets where CogniKernel *underperforms*. Section 0 is
the weakness measurement; everything after it is remediation.

---

## 0. The weakness

CogniKernel has no generation anywhere. `extraction/windowing.py` stores
**verbatim transcript spans** as the description and rationale of every event,
and `injection/template.py:363` — `generate_summary()`, docstring *"Deterministic
NL summary — no LLM call, zero latency"* — assembles the block by templating
those spans together. Encoders classify and extract well; nothing writes a
sentence. The stored memory is therefore only ever as well-formed as whatever
sentence happened to appear in the transcript.

### Preliminary audit (scaffolding — must be redone properly)

A heuristic sweep of **6,434 active statements** (`DECISION`, `CONSTRAINT_HARD`,
`CONSTRAINT_SOFT`, `APPROACH_ABANDONED`) drawn from **45 contributing project
stores**. 163 store files were scanned in total; 118 of them hold no typed
statements at all — mostly ephemeral `ck-honor-exec-*` benchmark projects. All
163 opened cleanly and none were skipped, so the sweep is complete rather than
undercounted.

| Defect class (surface heuristics) | Rate |
|---|---|
| `DANGLING_REFERENCE` — anaphora / deictic / discourse opener | 4.5% |
| `NOT_DURABLE` — workflow narration, status chatter | 3.8% |
| `NOT_A_STATEMENT` — fragment, table row, markdown debris | 2.5% |
| `META_TALK` — about CogniKernel itself, not the host project | 1.5% |
| **any surface flag** | **12.1%** (778 / 6,434) |

By type: `APPROACH_ABANDONED` 18.6%, `DECISION` 15.7%, `CONSTRAINT_SOFT` 10.9%,
`CONSTRAINT_HARD` 6.2%.

*(Figures are from the productionised `scripts/audit_statement_quality.py`,
`research/statement_audit/heuristic_20260725-193802.json`. They supersede an
earlier scratchpad estimate of 11.9% / "163 stores" — the regex set gained three
patterns, and the store count now reports contributing stores rather than files
scanned.)*

**Those heuristics measure a lower bound and must not be quoted as the result.**
A manual read of 11 statements from the bucket the heuristics called *clean*
found roughly 4 clearly defective and 3 borderline — defects that are
surface-clean and semantically wrong, which no regex can detect:

- **Type-mismatch** — `CONSTRAINT_HARD`: *"Non-zero temperature means the caller
  explicitly wants non-deterministic output."* A definition, not a constraint.
- **Prompt fragment as decision** — *"Memory/encoder scientist: the five
  highest-stakes science decisions, a concrete typed-event schema…"* is a
  subagent prompt.
- **Compound** — one `APPROACH_ABANDONED` carrying four distinct abandonments.
- **Dangling subject** — *"Fails the build if any file other than
  app/auth/jwt.py calls .get_secret_value()."* What fails the build?
- **Discourse opener** — a statement beginning *"But the corpus also carries…"*.

If that ratio holds, the true defect rate is plausibly **35–50%**. n=11 is far
too small to assert it.

**Also measured:** `rationale` is populated in only **9.9%** of events. The field
`windowing.py` exists to fill is empty nine times in ten, so the *why* behind a
stored fact is almost never captured.

---

## Phases and the go/no-go gate

The audit and the generator are **different commitments** and must not be
planned as one project. The audit is ~400 human labels over data already on
disk. The generator arm is 526-session window recovery, teacher spend, a LoRA
run, and a stratified human preference study.

| | Phase A | Phase B |
|---|---|---|
| Work | the audit (§0) | eval + generator (§1–§4) |
| Cost | ~400 human labels, no spend | window recovery, teacher spend, LoRA run, preference study |
| Gate | — | **conditional on Phase A** |

**Pre-registered continuation threshold**, fixed now, before the number is seen:

> Phase B proceeds only if the human-labeled defect rate on `DECISION` +
> `APPROACH_ABANDONED_DO_NOT_RETRY` (the two worst types in the preliminary
> sweep) is **≥ 20%**. Below that, the generator is not worth building and
> Phase A publishes alone.

Phase A is worth doing on its own terms regardless of the outcome: a
statement-quality audit of 6,434 real stored statements across 45 stores, with
two-labeler agreement, is a contribution in itself and fits this branch's
framing — measuring where CogniKernel underperforms — without needing the
generator to justify it.

Setting the threshold in advance matters because by the time Phase A returns a
number, Phase B will already be specced, and any number will read as
justification to continue.

### Phase A0 — pilot first (the immediate next action)

The 35–50% projection in §0 rests on 4-of-11 clean-bucket items judged in a
single pass, and it is currently the *sole* justification for the entire branch.
Before committing to the full 400:

> Label ~60 clean-bucket statements against the defect taxonomy. If the rate
> holds near 40%, Phase A is clearly worth the full effort. If it comes back
> near 15%, an hour's work has saved the branch's entire cost.

This is the highest-value action available and nothing else should start before
it.

### A0 interim outcome — LLM pre-labels only (human verification pending)

**Status: the gate is NOT yet decided.** These are model labels. The
pre-registered rule requires human-anchored labels, and the human verification
of a 20-statement subset has not been done. Recorded here so the evidence exists
before the decision, not as the decision.

Four hosted models (`DeepSeek-V4-Pro`, `Kimi-K2.6`, `gpt-oss-120b`, `GLM-5.2`)
independently labeled the 60-statement pilot pool, drawn entirely from the
bucket the surface heuristics called **clean**. 58 statements were labeled by
all four.

| Reading | Rate | 95% CI (Wilson) |
|---|---|---|
| DeepSeek-V4-Pro | 40/60 = 66.7% | 54.1 – 77.3 |
| Kimi-K2.6 | 40/59 = 67.8% | 55.1 – 78.3 |
| GLM-5.2 | 40/59 = 67.8% | 55.1 – 78.3 |
| gpt-oss-120b | 34/60 = 56.7% | 44.1 – 68.4 |
| **majority (≥3 of 4)** | **35/58 = 60.3%** | **47.5 – 71.9** |
| most conservative (unanimous 4/4 only) | 23/58 = 39.7% | 28.1 – 52.5 |

**Every reading clears the pre-registered 25% threshold with its CI excluding
it** — including the deliberately pessimistic unanimous-only reading, whose
lower bound is 28.1%. If human verification agrees even approximately, Phase A
proceeds.

This also confirms the §0 thesis quantitatively: the surface heuristics flag
12.1% of the corpus, but 57–68% of what they call *clean* is judged defective.
The heuristics are a severe undercount, exactly as predicted, and the n=11
hand-read estimate of 35–50% was if anything conservative.

**The load-bearing caveat.** Inter-model agreement is only *moderate*: binary
defective/clean Cohen's κ across the six pairs is 0.30, 0.35, 0.42, 0.45, 0.49,
0.77 (mean ≈0.46); per-code mean κ is 0.34–0.58. Models were unanimous on 32 of
58 statements and split on 26. Per-code counts diverge sharply —
`DANGLING_REFERENCE` 11–26, `WRONG_TYPE` 8–19, `COMPOUND` 3–12.

Two conclusions follow, and they must not be collapsed:

1. **The aggregate rate is robust.** Four architecturally distinct models that
   disagree about *which* statements are broken still converge on most of them
   being broken.
2. **Per-item and per-code labels are NOT reliable enough to serve as Phase B
   gold.** At κ 0.3–0.5 they cannot define the target a generator is trained or
   evaluated against. Phase B gold needs human labels or a materially better
   instrument.

This mirrors what `salience_v2`'s own evaluation already established: the
underlying judgment is genuinely ambiguous at the margin, which is why the
project holds models to a human-agreement ceiling rather than to 100%.

**Method caveat, stated for the record.** The original design had a human label
all 60. The human elected LLM pre-labeling with verification of a subset, so the
reported rate is a model construct anchored to human judgment only through that
subset and only as strongly as the measured agreement. A first run at
`max_tokens=1024` had to be discarded: 167 of 341 responses truncated with empty
content, and because harder statements consume more reasoning tokens, the
surviving labels skewed toward `CLEAN` — a non-random loss that would have
biased the rate *downward*. Re-run at 8192, completion went from 11/60 to 59/60
on the worst-affected model. Artifacts: `manifest_20260726-004114.json`
(superseded 1024-cap run kept as evidence at `manifest_20260725-234313.json`).

---

### Phase A deliverable — the audit, done properly

The preliminary sweep is scaffolding. The real thing is a deliverable:

- Stratified sample across event type, project, and register; n large enough for
  useful CIs (target ~400).
- **Human labels**, not heuristics, against the defect taxonomy above plus an
  open "other" category.
- **Two independent labelers on an overlapping subset**, reporting inter-labeler
  agreement — following the precedent already set for `salience_v2`, where the
  model is held to the human-agreement ceiling rather than to 100%.
- The heuristic sweep is reported alongside as a *cheap proxy*, with its measured
  miss rate against the human labels.

Productionise `scripts/audit_statement_quality.py` (currently a scratchpad
script) so the number is repeatable and can be tracked after remediation.

---

## 1. The eval (Phase B; the long pole — build this first within it)

**There is no gold data for this task.** Building it is Phase B's critical path,
not an appendix.

**Input unit is the transcript window, not the stored event.** Because
`rationale` is empty 90% of the time, an event alone does not contain the
context needed to write a good statement — the generator would have to invent
it. Pairs are therefore built from the **526 session JSONLs**, reusing
`extraction/jsonl_converter.py` and `extraction/windowing.py` to recover the
window each event came from.

**Gold statements are teacher-written with human spot-check.** A strong teacher
writes the canonical statement from the window; a stratified sample of ~100 is
hand-verified. This follows established practice in this repo —
`scripts/build_cot_sft.py` already calls a hosted teacher to build training data.

**Teacher provider: Together AI.** The OpenAI key has expired.
`scripts/build_cot_sft.py:40` hardcodes
`https://api.openai.com/v1/chat/completions` and reads `OPENAI_API_KEY` from
`.env`; the new data-building script must not inherit that. Requirements:

- Read base URL and key from config/env (`TOGETHER_API_KEY`, base
  `https://api.together.xyz/v1`) rather than hardcoding a host. Together's
  endpoint is OpenAI-compatible, so the request/response handling in
  `build_cot_sft.py` is reusable as-is — only the URL, key name, and model id
  change.
- Keep the on-disk response cache (`cot_cache/`-style, hashed by input) so a
  provider or key change never forces a full re-spend.
- Record the teacher model id in the dataset metadata. Gold built by different
  teachers must not be silently mixed.
- `.env` is gitignored (`.gitignore:84`) and untracked — verified. The key
  belongs there, never in a committed file or a script default.

> **The no-LLM promise is a *runtime* constraint, not an eval-construction one.**
> CogniKernel's claim is that nothing leaves your machine during a session and no
> API key is required to use it. Building training data offline with a teacher
> does not touch that claim, and the precedent is already in the repo. The spec
> states this explicitly so the study does not read as violating the project's
> headline promise.

**Known ceiling:** gold quality is bounded by the teacher, and any statement the
teacher writes badly becomes wrong gold. The human spot-check measures that rate
rather than assuming it away; if teacher error is high, fall back to
hand-writing a smaller set.

This ceiling matters *more* on Together than it did on gpt-4o-mini: the model
menu is open-weights and the quality spread across them is wide, so the choice
of teacher model is a real experimental variable rather than an implementation
detail. Pick a large instruct model, and treat the spot-check as gating the
teacher — if it fails, change teacher before generating the full set, not after.

**Decontamination:** eval windows must come from sessions disjoint from any used
for training data, checked by session ID *and* by near-duplicate text (5-gram
Jaccard > 0.9), since the same project produces near-identical sentences across
sessions.

### Where ABSTAIN gold comes from

§2 makes abstention a required output and §3 scores its precision and recall —
but gold built only from windows that *produced* stored events contains **no
negatives**, because the classifier already accepted every one of them. The eval
would have nothing to abstain on.

The negatives come from Phase A: the audit's **meta-talk, narration, and
prompt-fragment** items are precisely the population of windows that were
classified as memory but should not have been. Route them into the eval as
abstain-gold, stratified alongside the positives, and record how many are
available — if that population is thin, abstention metrics will be
underpowered and must be reported as such rather than quietly averaged in.

---

## 2. The task

```
input :  transcript window (the span + surrounding context) + event type
output:  a single canonical memory statement   OR   ABSTAIN
```

**Scope is deliberately narrow.** Not in phase 1: compound-splitting (emitting N
statements for N facts) and type correction. Type correction in particular would
put the generator in competition with `salience_v2` rather than beside it, and
the division of labor — **encoders classify and extract; a small local decoder
writes** — is the point. The encoder is not a competitor here; it structurally
cannot generate.

**A good statement is:**

1. **Grounded** — asserts only what the window supports.
2. **Standalone** — readable with no access to the transcript (this is what the
   anaphora and dangling-subject defects violate).
3. **Adequate** — captures the fact, not a fragment of it.

**ABSTAIN is a required output, not a nice-to-have.** Several audit defects
(subagent prompt as decision, narration as decision) are *classification*
failures upstream. A generator that always writes will rewrite those into
confident, fluent, well-formed garbage — strictly worse than today's ugly output,
because the current text at least *looks* broken. Abstention is what prevents the
generator from amplifying upstream errors, and it gets its own metric.

---

## 3. Metrics

**Baseline to beat is the current system: the verbatim span.** This is the honest
comparison and it may lose — for the ~50% of statements the audit finds already
clean, a generator can only break what works.

- **Groundedness** — human-labeled on a sample: does the statement assert
  anything the window does not support? Reported as a hallucination rate, since
  a hallucinated "decision" injected as authoritative context corrupts every
  later session. Automatable proxy (NLI entailment) reported with its measured
  agreement against the human labels.
- **Standalone-ness** — human-labeled: comprehensible without the transcript.
- **Adequacy** — human-labeled: captures the fact.
- **Abstention precision / recall** — against gold "no durable fact here".
- **Head-to-head vs verbatim span** — forced-choice human preference on matched
  windows, reported separately for the audit's *defective* and *already-clean*
  strata. **A generator that improves defective statements while degrading clean
  ones is a net loss**, and the stratified split is the only way to see that.
- **Regression guard** — the head-to-head loss rate on the already-clean stratum
  is the number that decides shippability, not the average.

All human-labeled metrics carry two-labeler agreement on an overlapping subset.

### Two confounds the head-to-head must control

**Blinding.** Stratum assignment comes from the audit's own labels, so raters
must not see them, must not know which side is generated, and left/right
presentation must be randomized per item. Otherwise "this one is from the
defective stratum" leaks straight into the preference judgment.

**Fluency.** A generated statement will read better than a raw transcript span
whether or not it is *more accurate*, so forced-choice preference favors
generation almost by construction. Preference must therefore not be allowed to
absorb correctness: **groundedness is asked separately and per-item** ("does this
assert anything the window does not support?"), and a fluent hallucination that
wins on preference is still counted as a groundedness failure. Report the two
side by side; preference alone is not evidence of improvement.

---

## 4. The generator

**Candidate:** LoRA-SFT `models/lfm2.5-230m-base` on (window → statement) pairs,
merged and exported to ONNX for torch-free CPU inference — the same contract the
existing heads already ship under, running off-hot-path in the Stop hook/worker.

**Capability gate, checked early.** Grounded generation is materially harder than
classification, and 230M is small. Before committing to the full study, hand-run
the teacher-written pairs through the base model and a LoRA'd checkpoint on ~50
windows and measure the hallucination rate. If a 230M model cannot stay grounded,
the finding is *"faithful memory-statement generation needs more than 230M on
CPU"* — a real and useful negative result that redirects the architecture, and
worth reporting rather than burying. Larger local candidates are then a separate
question.

---

## 5. Non-goals

- Compound-splitting and type correction (see §2).
- Any change to `salience_v2`, `supersession_xenc`, or the taxonomy.
- Any runtime API dependency. The deliverable runs locally or it does not ship.
- Replacing `generate_summary()`'s block assembly. This study is about statement
  quality at capture time; block-level summarisation is downstream and separate.
- Fixing the non-generation defects the audit surfaced in passing — truncated
  paths (`xtraction/`, `rc/memlora/`) and `.uv-cache` entries in the skeleton are
  string-slicing and scope **bugs**. They should be filed separately; a generator
  fixes neither, and citing them as motivation would weaken the argument.

---

## 6. Risks

| Risk | Handling |
|---|---|
| 230M too small to stay grounded | §4 capability gate; negative result is reportable |
| Generator degrades already-clean statements | Stratified head-to-head; clean-stratum loss rate gates shipping |
| Generator amplifies upstream misclassification | ABSTAIN as required output with its own metric |
| Teacher-written gold inherits teacher errors | Human spot-check measures the rate; fall back to hand-written |
| True defect rate is nearer 12% than 40% | Phase A0 pilot (n≈60) tests it in an hour; Phase A's pre-registered ≥20% threshold gates Phase B. A low rate ends the branch, and that is an acceptable outcome |
| Preference study rewards fluency over accuracy | Groundedness asked separately per-item; blinded, randomized presentation |
| No negatives in the eval, so ABSTAIN is unscorable | Audit's meta-talk/narration/prompt-fragment items routed in as abstain-gold |
| Eval windows contaminated by training sessions | Session-ID disjointness + 5-gram Jaccard near-dup check |
| Reads as violating the no-LLM promise | §1 states the runtime/eval-construction distinction explicitly |

---

## Relationship to the CoT-faithfulness spec

`2026-07-25-cot-faithfulness-design.md` measures whether a **classifier's**
emitted `<thought>` is load-bearing. It is a valid, cheaper, separate question —
the model and eval already exist — but it does not feed this study: its eval
contains no gold statements, and its competence gate and encoder comparison were
built for classification. Neither transfers. Both specs stand; this one is the
priority.
