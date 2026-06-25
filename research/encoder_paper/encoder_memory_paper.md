# Encoder-Based Memory Extraction for AI Coding Agents — A Local-First, Deterministic Approach

> **Status:** LIVING DRAFT + METRICS LEDGER. Updated incrementally as each sprint
> experiment lands. The full prose write-up is to be expanded by a dedicated
> sub-agent AFTER B2/C1/D2 complete; until then this file is the source of truth for
> every measured number so nothing is reconstructed from memory later.
> **Scope:** CogniKernel / MemLoRA Stage-2 (extraction) and Stage-5 (supersession).

---

## Abstract (draft)

CogniKernel preserves durable project memory (decisions, constraints, rejected
approaches, open threads) across AI-coding sessions. Its Stage-2 extraction began as
a deterministic keyword/regex/Aho-Corasick pipeline (≈83 trigger phrases + ≈28 marker
tokens). On a recall-heavy held-out session it scored **38% precision / 19% clean
recall / 42% noise**. We reframe extraction as a set of *discriminative* sub-tasks
(salience, typing, supersession) served by a single small encoder backbone, paid for
once offline and run at inference as a deterministic numpy/ONNX forward pass — keeping
the system **local-first, free, and cache-stable**. We report an incremental
progression of methods and a key negative result: a *frozen* embedding backbone with
a linear head saturates well below the ceiling, and hard-negative training data only
becomes useful once the backbone itself is contrastively fine-tuned.

---

## 1. The thesis

"Extraction" decomposes into sub-tasks split by model class:
- **Discriminative / encoder-shaped:** salience (fact vs noise), typing (6 classes),
  supersession/contradiction. → BERT-family encoder + cheap heads.
- **Generative / decoder-shaped:** canonicalization, list aggregation. → deferred (v3).

A single shared encoder backbone populates multiple stores from cheap forward passes:
pooled vector → recall/embeddings DB; class logits → events DB; (cross-encoder) pair
score → supersession edges. Intelligence is paid once, offline; online inference stays
tiny, deterministic, numpy/ONNX, off the interactive hot path (runs in the Stop hook).

**Backbone:** `BAAI/bge-small-en-v1.5` — BERT, 12 layers / 384 hidden / 33.4M params,
512-token limit, CLS pooling, MIT license. Already resident for recall (zero marginal
model cost). MTEB profile shows latent strength on the two hard tasks (Pair
Classification 84.9, STS 81.6) that a linear probe over the *frozen* vector cannot
reach — motivating fine-tuning.

---

## 2. Methodology (held-out, leakage-disciplined, version-agnostic)

- **Stage-2 gate:** `tests/fixtures/relay_s1_gold.json` + `scripts/eval_extraction.py`.
  Token+type scoring → identical yardstick for the regex extractor and any encoder
  head. The Relay S1 session is **held out** (never trained on).
- **Stage-5 gate (new, this sprint):** `tests/fixtures/supersession_pairs_gold.json`
  (21 relation-labeled pairs incl. same-area *guard* negatives) +
  `scripts/eval_supersession.py`. Text-only, version-agnostic, pluggable predicate.
- **Synthetic training data, global + diverse:** generated via OpenAI (offline,
  paid-once) over 32 project domains × 22 software facets so the model learns the
  *task*, not a stack. Every generator excludes the held-out fixtures (leakage guard).
- **Determinism:** closed-form ridge / fixed-weight ONNX → byte-stable logits → stable
  `content_hash` and prompt-cache prefix.

---

## 3. Metrics Ledger — old → new (APPEND AS EXPERIMENTS LAND)

### 3.1 Stage-2 Extraction — Relay S1 held-out gold

| Approach | Precision | Present-recall | Clean-recall | Noise | hard_miss | Source |
|---|---|---|---|---|---|---|
| Legacy regex/Aho-Corasick (baseline) | 38% | ~58% | 19% | 42% | 7 | hand-scored analysis |
| + Segmenter fix (A: keep_whole_fact, list tags) | 78% | — | — | 22% | — | analysis (truncated 12→0) |
| + Learned salience head, **broad mode** (B, frozen+linear) | **93%** | **95%** | **57%** | **7%** | **1** | Relay S1 gold |
| Fine-tuned backbone (B2, SetFit) | _pending_ | _pending_ | _target ≥75%_ | _≤6%_ | _≤1_ | — |
| + Decoder span repair (v3) | _future_ | — | _ceiling ~88–90%_ | — | — | — |

**Typing (6-way):**
| Setting | Accuracy | Source |
|---|---|---|
| Frozen+linear CV (seeds+generated, 6-way) | ~91% | KT (documented) |
| Frozen+linear, in-dist held-out (n=222) | **90.5%** | B1 A/B (this session) |
| Frozen+linear + hard-neg twins (in-dist held-out) | **84.2%** ⬇ | B1 A/B (this session) |
| **SetFit fine-tuned + twins (in-dist held-out)** | **92.3%** ⬆ | B2 (this session) — the F2 flip |
| Real transcript (Taskflow/MOB_C) typing | 8/11 (~73%) | KT (documented) |
| Fine-tuned backbone, real-transcript typing | _pending D2 wiring_ | _target ≥88%_ |

### 3.2 Stage-5 Supersession — held-out pair gold (21 pairs)

| Approach | Precision | Recall | guard_fp | Source |
|---|---|---|---|---|
| Lexical (Jaccard∪Levenshtein) + subject-keying | **100%** | **38%** | **0** | D1 (this session) |
| + bi-encoder cosine @0.75 | ~100% | ~+0 (semantically blind) | 0 | supersede.py:289 finding |
| Cross-encoder (C1, bge-small base, 3 ep CPU) @ natural threshold ≈0.963 | **100%** | **85%** | **0** | C1 (this session) — meets all targets |
| Cross-encoder (C1) @ synthetic-val-selected threshold (0.05) | 62% | 100% | 5 | C1 — calibration failure, see F5 |

**C1 per-pair separability (held-out gold):** classic same-area guards are cleanly
rejected — `composite-PK vs UUID-PK` P=**0.058**, `Postgres-store vs Redis-counters`
0.064, `SHA256-keys vs argon2id-passwords` 0.279 — vs true relations at **0.96–0.97**.
The cross-encoder does exactly what cosine could not (supersede.py:289). The only
near-positive negatives are "complementary same-area" pairs (priority-router vs
round-robin), one of which is a debatable gold label.

---

### 3.3 D2 — deployable runtime (local, torch-free)

| Check | Result |
|---|---|
| ONNX export parity vs SetFit `encode()` (CLS+L2norm) | **max_abs 8.9e-08, cosine 1.000000** |
| ONNX-runtime v2 head vs torch SetFit (60 real Relay texts) | **60/60 labels match** |
| `MEMLORA_EXTRACTOR=v2`/`v2-broad` wired into pipeline | done; end-to-end extract OK |
| Extraction regression suite | **314 passed, 1 xfailed** |

Runtime path is onnxruntime + `tokenizers` + numpy (no torch) — deterministic forward
pass, consistent with v1's content-hash/cache contract. **Decoupled design:** the v2
fine-tuned body serves *classification only*; recall embeddings stay on stock fastembed,
so no re-embed/backfill is required. Confidence-floor is intentionally NOT enabled for v2
(F6 shows v2 already errs conservative; a floor would cut recall further).

## 4. Key findings so far

**F1 — Detection is solved; typing is the gap.** Present-recall 95% vs clean-recall
57% means fact/noise detection is essentially done by a linear probe; the remaining
loss is *type* correctness (and faithfulness). Clean-recall ≈ typing-bound.

**F2 — The frozen-backbone ceiling is real (negative result).** On a fixed
in-distribution held-out, the frozen+linear head scores 90.5%; adding 1,840
hard-negative deontic/polarity twins *lowers* it to 84.2%. Hard negatives sit on top
of each other in the frozen space (different labels, near-identical vectors) → label
noise to a linear head. They become signal only after the backbone is contrastively
fine-tuned. **Implication:** the twins are *banked for B2 (SetFit)*, not shipped in the
frozen head — and the real typing gap on transcripts is an out-of-distribution /
backbone-capacity problem, not a labeling problem.

> **F2 CONFIRMED (B2):** SetFit fine-tuning flipped the twins from harmful to helpful
> on the *same* fixed held-out: frozen 90.5% → frozen+twins 84.2% → **fine-tuned+twins
> 92.3%** (+8.1 vs frozen+twins, +1.8 vs frozen). The geometry — not the labels or the
> architecture — was the bottleneck, exactly as the thesis predicted. (In-distribution
> synthetic gain is modest because that held-out is easy; the larger expected gain is
> out-of-distribution on real transcripts, pending D2 runtime wiring to measure.)

**F3 — Cosine cannot do supersession; the relation needs joint encoding.** Lexical +
subject-keying is precise (100%) but blind to paraphrased corrections, contradictions,
and subsumption (recall 38%). Bi-encoder cosine adds ~nothing at the safe threshold
because same-domain corrections and unrelated same-domain decisions are non-separable
in vector space. This is the textbook cross-encoder case (C1).

**F4 — Data diversity is the gating resource.** Both corpora span 32 domains × 22
facets to generalize globally; the corpus, not the architecture, is the bottleneck
(the panel's consensus, now backed by F2).

**F5 — The cross-encoder works; synthetic-val threshold *calibration* does not.** The
C1 cross-encoder separates the same-area guards cosine cannot (0.06 vs 0.96) and meets
every gold target (P100/R85/guard0) at its natural confidence edge (~0.963). But the
operating threshold cannot be *selected* from the synthetic validation set: synthetic
negatives are too easy, so val-based selection lands at ~0.05 (P62/guard5 on real
guards). This is the same "easy-negative" methodology trap recorded in
embedding_architecture.md §5.2. **Implication / next step:** before C2 integration,
(a) harden the negative corpus with "complementary same-area" pairs and (b) apply
temperature calibration, so a deployable threshold is selectable without gold-peeking.
Shipping a fragile threshold is unacceptable here — a false supersession deletes a
valid decision.

---

**F6 — In-distribution typing gain ≠ OOD clean-recall gain (a caution).** Re-typing
the legacy extractor's 60 real Relay S1 candidates (OOD text) with each head:
v1 frozen → P66/clean-recall33/noise34; v2 SetFit → **P81 (+16) / clean-recall24 (−10)
/ noise19 (−16)**. The fine-tune made the head markedly more precise and less noisy but
more conservative, *lowering* clean-recall — the opposite of the in-distribution result
(F2). Two caveats: this is FILTER mode over a candidate set whose own present-recall is
only 52% (so recall is candidate-capped, not head-capped), and the drop looks like a
threshold/conservatism effect. **Implication:** a fair v2 verdict needs BROAD-mode
evaluation over the raw transcript (no fixture yet) plus confidence-floor tuning; the
synthetic→real typing transfer is partial and must not be assumed from in-dist numbers.
This is the same lesson as F5, on the typing side.

## 5. Corpora built (this sprint)

| Corpus | Size | Distribution | Leakage | Feeds |
|---|---|---|---|---|
| Pairwise supersession (A2) | 2,200 | unrel 720 / sup 470 / ref 370 / sub 320 / con 320 | 0 | C1 |
| Hard-negative twins (A1) | 1,840 | DEC 480 / HARD 470 / NOISE 390 / SOFT 200 / GRAVE 180 / THREAD 120 | 0 | B2 |
| Prior type corpus | 1,111 | 6-class (seeds 161 + generated 950) | 0 | B1/B2 |

---

## 6. Open / pending (paper completion checklist for the sub-agent)

- [ ] B2 results: fine-tuned backbone typing acc + Relay/Taskflow clean-recall delta.
- [ ] C1 results: cross-encoder P/R/guard_fp on the gold gate; the ROC/threshold curve.
- [ ] D2: end-to-end `v2` mode numbers; confidence-floor calibration on the real gate.
- [ ] Ablations: twins on/off under fine-tuning (mirror of the F2 A/B, now expecting +).
- [ ] Latency/footprint table (CPU inference ms, model MB) — the local-first claim.
- [ ] Related work: mem0, Zep (temporal KG), GLiNER/GLiNER2, SetFit, BGE.
- [ ] Threats to validity: synthetic-data realism, single held-out session, domain mix.

---

## 7. Sub-agent handoff note

When B2/C1/D2 are complete, a sub-agent expands §1–4 into full prose, generates the
final tables from this ledger, and produces figures (metric-progression bar charts per
stage). It must treat this ledger as ground truth and must NOT invent numbers — every
value here is traceable to a script run recorded in the sprint task history.
