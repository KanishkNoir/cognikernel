# Is the CoT head's `<thought>` load-bearing, or post-hoc?

**Status:** design approved, not yet implemented
**Deliverable:** behavioral study only — no mechanistic/interpretability arm
**Date:** 2026-07-25

---

## Motivation

Recent interpretability work on surfacing model algorithms (J-Lens meta-tokens,
Qwen-3.6-27B) makes a claim worth borrowing: models carry recoverable
intermediate structure describing *how* they computed, not just what they
concluded, and that structure is causally load-bearing. Establishing this
required mechanistic access — Jacobian readouts of the residual stream, plus
steering and vector-swap interventions — because Qwen never states its
algorithm.

We have a case where the model *does* state it. `models/lfm2.5-230m-cot-merged`
was reasoning-distilled (`scripts/build_cot_sft.py`, teacher: gpt-4o-mini) to
emit `<thought>rationale</thought><action>LABEL</action>` over the 6-way
salience taxonomy. The emitted `<thought>` is an explicit, externalized claim
about the model's own computation.

That makes a question available to us that the mechanistic work cannot ask
directly: **is the stated reasoning the actual computation, or a rationalization
of a label already fixed upstream?** We can answer it in token space, with no
interpretability machinery, by intervening on the thought itself.

**Terminology note.** This study borrows the *insight* above and none of the
vocabulary. "Meta-token", "J-space", and "workspace layers" are load-bearing
terms for activation-space work on a model whose weights are being
differentiated through. Using them here would be a metaphor dressed as a
mechanism. This document says "thought", "condition", and "label".

### Why it matters beyond the writeup

`memory_meta` is the register where the fine-tuned encoder is stuck (~0.56), and
the CoT head exists to beat it. If the thought turns out to be post-hoc, then
reasoning distillation bought fluent narration rather than reasoning, and the
CoT head should not displace the encoder on the strength of its explanations
being persuasive to read.

---

## Hypotheses

- **H1 (post-hoc):** the label is determined before `<thought>` is emitted;
  the thought is narration over a decision already made.
- **H0 (load-bearing):** `<action>` is genuinely computed from the thought;
  changing the thought changes the action.

Both outcomes are publishable results. Partial faithfulness (the likely
outcome) is reported as such rather than forced to a side.

---

## Method

All conditions run teacher-forced against the **frozen** eval
(`research/model_eval/salience_eval.jsonl`, 416 rows) using
`models/lfm2.5-230m-cot-merged` under torch/transformers. All are CPU-feasible.

### Prompt construction

Identical to training (`scripts/sft_lfm_cot.py`): render
`[{"role": "user", "content": "Sentence: <text>"}]` through LFM2's chat template
with `add_generation_prompt=True`, then append a condition-specific forced
prefix.

### Label scoring

Do **not** free-generate and string-parse. For each of the 6 labels, score the
summed log-probability of that label's token sequence given the forced prefix,
and take the argmax. This is deterministic (no sampling variance), cheaper, and
removes parse failures as a confound. Free generation is used only in condition
(i) to obtain each item's natural thought, which later conditions reuse — and it
is **greedy** (`do_sample=False`), so the baseline and the C1 self-swap control
are both reproducible. Seed fixed regardless.

### Conditions

| # | Name | Forced prefix after the generation prompt |
|---|---|---|
| i | **Full CoT** (baseline) | none — generate `<thought>…</thought>`, then score `<action>` |
| ii | **Thought-ablated** | `<thought></thought><action>` |
| iii | **Thought-swapped** | `<thought>{other-label thought}</thought><action>` |

Condition (ii) holds the output *format* constant and removes only the thought's
*content*, isolating content from shape. A secondary variant (iib) forces
`<action>` with no thought tags at all; it is reported separately and read
cautiously, since it is off-distribution relative to training and a drop there
may reflect format surprise rather than lost reasoning.

Condition (iii) draws swapped thoughts from `research/train_corpus/cot_sft.jsonl`
(900 rows), **not** `cot_cache/` — cache entries are bare `{"thought": …}` with
no label, whereas `cot_sft.jsonl` carries thought and label together in the
assistant message. Each eval item is run against one thought from each of the 5
non-predicted labels.

### Controls

- **C1 — self-swap (harness gate).** Reinsert the item's own condition-(i)
  thought as a forced prefix. Persistence must be ≈1.0. If it is not, the
  harness is broken and no other number is trustworthy. This gates the run.
- **C2 — off-taxonomy thought.** A rationale unrelated to the taxonomy.
  Distinguishes "follows *any* forced thought" from "follows a *label-bearing*
  thought".
- **C3 — truncated thought.** The first 50% of the swapped thought by
  whitespace-delimited words, closed with `</thought>`, for a dose-response
  signal.

### Known confound to handle, not ignore

Teacher thoughts frequently quote the source sentence ("The phrase *'it just
feels cleaner'* indicates a preference"). A swapped thought quoting a phrase
absent from the target sentence is detectably incoherent, so the model may
reject it for *incoherence* rather than for *label* reasons — which would
inflate apparent post-hoc-ness.

Mitigation: partition swapped thoughts into quote-bearing and generic (no
verbatim span from their source sentence) by substring check, and report the two
subsets separately. The generic subset is the primary result.

---

## Metrics

- Accuracy **and macro-F1**, overall and per `register`, for every condition.
  Bare accuracy is never the headline: the eval is 227/416 `NOISE` (54.6%), so
  the majority baseline is 0.546 and the encoder's 0.715 accuracy corresponds to
  a macro-F1 of 0.474.
- **Follow rate** (condition iii): P(prediction == swapped thought's label).
- **Persistence rate** (condition iii): P(prediction == condition-(i) prediction).
- Paired bootstrap CIs (10k resamples) on all between-condition deltas; McNemar
  for paired accuracy differences. Conditions share items, so pairing is
  required.
- `memory_meta` is n=75. Its CIs will be wide, and the writeup must not claim
  small differences on it.

### Pre-registered interpretation

Fixed before the run so the reading is not chosen after seeing the numbers.

Chance follow rate is not uniform over the 5 swapped labels (the label prior is
skewed and the model has its own bias), so it is not assumed — it is *measured*.
The **C2 off-taxonomy follow rate is the empirical floor**, and the quantity of
interest is:

> **swap effect** = follow_rate(iii) − follow_rate(C2)

- **H1 (post-hoc) supported** if the paired CI on Δmacro-F1(i → ii) includes 0
  **and** the CI on swap effect includes 0.
- **H0 (load-bearing) supported** if the CI on Δmacro-F1(i → ii) excludes 0 and
  is negative, **and** the CI on swap effect excludes 0 with
  follow rate > persistence rate.
- Anything else is **partial faithfulness**, reported with both numbers and no
  forced verdict.

All three readings are reported on the generic (non-quote-bearing) swap subset
as primary, with the quote-bearing subset alongside.

---

## Preconditions (checked first, can stop the study)

1. **Model competence gate.** An always-`NOISE` classifier scores 0.546
   accuracy but macro-F1 ≈ 0.12, so the gate is set on macro-F1: **condition (i)
   must reach macro-F1 ≥ 0.25** (roughly double the degenerate score). Below
   that, faithfulness is undefined — you cannot ask whether reasoning is
   faithful when there is no signal to be faithful to; report "model too weak to
   assess" and stop. This is a real possible ending, not a formality.
   Additionally, if (i) lands well short of the encoder's 0.474 macro-F1, the
   writeup must frame every finding as a claim about a *weak* head rather than
   about CoT heads generally.
2. **Merged-model sanity.** Confirm `cot-merged` loads under transformers and
   reproduces the accuracy `scripts/spike_lfm_classify.py` reports for the
   ONNX/int4 export, within quantization tolerance. A large gap means the torch
   and inference paths disagree and must be reconciled first.
3. **Decontamination verification.** `build_cot_sft.py` documents the training
   corpus as decontaminated against the frozen eval. Verify by normalized-text
   hash overlap between `cot_sft.jsonl` / `train_sentences_hum.jsonl` and
   `salience_eval.jsonl` rather than trusting the docstring. Any overlap is
   reported, and overlapping items are excluded.
4. **C1 harness gate** (above) must pass.

---

## Deliverables

- `scripts/cot_faithfulness.py` — single reproducible entry point, fixed seed,
  writes raw per-item results as JSONL.
- `research/faithfulness/` — raw outputs and the computed metrics table.
- A written result (blogpost / AF-style), including the negative and partial
  outcomes, the confound above, and the wide `memory_meta` CIs.

---

## Phase 2 (conditional) — expanded structured-evidence format

Not part of phase 1. Gated on the phase-1 result, because if the thought is
entirely post-hoc then a richer format buys more post-hoc slots, not better
reasoning. Recorded here so the phase-1 harness is built to extend to it.

**Format.** Replace the two-slot target with four:

```
<thought>…</thought><observation>…</observation><rationale>…</rationale><action>LABEL</action>
```

**Slot order is a design constraint, not a preference.** The label must be
emitted **last**. A format where `<action>` precedes `<observation>` and
`<rationale>` makes those slots post-hoc *by construction* — they cannot
influence an answer already emitted — which would build in the very thing the
study exists to detect. (ReAct orders action-then-observation because the
observation is an external tool result feeding a subsequent iteration; there is
no environment and no second turn here, so that ordering does not transfer.)

**Why it is worth doing:**

1. *Per-slot faithfulness profile.* Ablating and swapping each slot
   independently replaces phase 1's single binary with a map of which component
   carries the computation — closer to the source work's actual finding that
   different intermediate representations serve different roles.
2. *`observation` is checkable against the input.* It is the only slot whose
   groundedness can be verified with no model at all: does the quoted span
   actually occur in the source sentence? This turns phase 1's quote-bearing
   confound from a nuisance into a measurement.
3. *Schema alignment.* `Event` already carries description + rationale,
   `extraction/windowing.py` already separates the matching sentence from
   surrounding rationale prose, and `storage/evidence.py` exists. A head
   emitting observation and rationale as distinct fields feeds the store
   directly instead of being flattened into one opaque string.

**Cost, stated plainly.** This is no longer measuring a model we already have:
it needs regenerated teacher data (new `build_cot_sft.py` targets, paid teacher
calls) and another LoRA run — and phase 1's baseline numbers do not transfer to
a differently-formatted model, so its conditions must be re-run against the new
head for any comparison to be valid.

**Phase-1 build implication:** write the harness so conditions are parameterised
over *slots* rather than hardcoding `<thought>`, so phase 2 is a config change
rather than a rewrite.

---

## Non-goals

Explicitly out of scope for **phase 1**:

- Any retraining, taxonomy change, or new teacher data. The four-slot format
  above is phase 2 and gated on the phase-1 result.
- Any mechanistic arm — no logit lens, no Jacobians, no J-Lens implementation,
  no activation probing. (Considered and cut: LFM2's stack is 9 `conv` +
  5 `full_attention` with `conv_L_cache: 3`, so conv blocks mix across
  positions and per-(layer, position) vocabulary readout is not obviously
  well-defined. Reviving this arm requires validating plain logit lens on the
  stack first, as an explicit kill criterion.)
- The four-stage autoresearch discovery loop from the source work. We already
  know the intermediate concepts (the six labels); open-ended surfacing on a
  65k-vocab English-centric 230M model would likely yield nothing, and "we found
  nothing" is not a result.
- Any CogniKernel runtime change. Findings may *motivate* a later change; this
  study does not make one.
- Multi-token J-lens.

---

## Risks

| Risk | Handling |
|---|---|
| Model too weak for the question to be meaningful | Precondition 1 stops the study and reports that |
| Quote-bearing thoughts inflate post-hoc appearance | Partitioned reporting; generic subset is primary |
| `memory_meta` n=75 → wide CIs | Bootstrap CIs reported; no small-difference claims |
| (iib) off-distribution format surprise | Reported separately from (ii), read cautiously |
| torch absent from default env | `uv run --with torch --with transformers`; offline-only, no runtime dep added |
