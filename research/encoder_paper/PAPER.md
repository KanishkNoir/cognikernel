# Encoder-Based Memory Extraction for AI Coding Agents: A Local-First, Deterministic Approach

> Companion data appendix: `research/encoder_paper/encoder_memory_paper.md` (the metrics ledger). Every quantitative claim in this paper is traceable to that ledger; numbers not yet measured are explicitly marked *pending* or *future work*.

## Abstract

CogniKernel preserves durable project memory — decisions, constraints, rejected approaches, and open threads — across AI-coding sessions. Its Stage-2 extraction began as a deterministic keyword/regex/Aho-Corasick pipeline (≈83 trigger phrases + ≈28 marker tokens). On a recall-heavy held-out session it scored 38% precision, 19% clean recall, and 42% noise. We reframe extraction as a set of *discriminative* sub-tasks — salience (fact vs noise), typing (six classes), and supersession (does a newer memory replace an older one) — served by a single small encoder backbone (`BAAI/bge-small-en-v1.5`, 33.4M parameters). Intelligence is paid for once, offline; online inference is a deterministic numpy/ONNX forward pass with no PyTorch, keeping the system local-first, free, and cache-stable. We report an incremental method progression with several honest negative results. A *frozen* backbone with a linear head saturates below the ceiling, and hard-negative training data only becomes useful after the backbone is contrastively fine-tuned (SetFit): the geometry, not the labels, was the bottleneck. A cross-encoder solves supersession separability that cosine cannot, but its operating threshold cannot be calibrated from synthetic validation data. In-distribution typing gains did not transfer cleanly to out-of-distribution real transcripts. We document the runtime parity and the open calibration gap.

## 1. Introduction

AI coding agents are stateless across sessions. Each new session begins without the architectural decisions, hard constraints, rejected approaches, and open work items established in prior sessions. The agent re-asks settled questions, re-proposes abandoned approaches, and silently violates rules it was never told about. CogniKernel addresses this by capturing durable project memory from session transcripts and re-injecting a compact, typed view of it at the start of each new session.

The quality of that memory is gated by extraction: the step that turns an unstructured session transcript into typed events. If extraction is wrong, every downstream stage — ranking, compression, injection — optimizes noise.

The original extraction pipeline (Stage 2) was deterministic and ML-free by design: a custom sentence splitter, an Aho-Corasick scan over a dictionary of ≈83 trigger phrases and ≈28 marker tokens, content-aware sliding windows, and a heuristic classifier. This was a deliberate v1 choice — developer language is conventional ("we decided to", "we can't use"), so keyword matching is a defensible starting point. But on a recall-heavy held-out session ("Relay S1", an architecture discussion) the deterministic pipeline produced **60 events at 38% precision, 19% clean recall, and 42% noise**, with **7 hard misses** (ground-truth facts with no covering event). The keyword surface over-fires on echoes and meta-narration while missing facts phrased outside the dictionary, and — most importantly — it cannot tell a hard constraint from a soft one, or a decision from a rejection, because those distinctions are semantic, not lexical.

**Thesis.** Extraction is not one task; it is a set of *discriminative* sub-tasks, each shaped like a classification problem an encoder solves well:

- **Salience** — is this sentence a durable fact or noise?
- **Typing** — which of six event types is it?
- **Supersession** — does a newer memory replace an older one, or are they unrelated?

These are encoder-shaped problems. A single small BERT-family backbone can serve all of them from cheap forward passes: the pooled vector feeds the recall/embeddings store, class logits feed the events store, and a cross-encoder pair score feeds supersession edges. Crucially, the intelligence is paid *once, offline* — synthetic data generation and fine-tuning happen on a developer's CPU before deployment. At inference, the system runs a deterministic numpy/ONNX forward pass with no PyTorch, no GPU, and no network call. This is the **local-first, free, deterministic** moat: the same input always produces the same logits, which keeps the `content_hash` and the prompt-cache prefix byte-stable, and nothing ever leaves the machine.

Generative sub-tasks — canonicalization, list aggregation, span repair — are deferred to a future decoder (v3); this paper covers the discriminative half.

## 2. Background: The CogniKernel System

CogniKernel is a local, SQLite-backed memory and code-orientation layer for AI coding sessions. It is event-sourced: every extracted decision is an immutable row in an append-only `events` table, and the "current state" is a *projection* derived by replaying active (non-superseded, non-archived) events. Event sourcing is a deliberate architectural commitment — it preserves rationale and history for time-travel queries, enables non-destructive bug recovery, and structurally produces the labeled data future training stages need.

Each project maps to one SQLite database. The system runs through Claude Code hooks:

- **SessionStart** renders a compact memory block (decisions, constraints, graveyard, open threads, codebase skeleton) into the agent's context.
- **PreToolUse** gates redundant file reads against an AST-derived symbol skeleton.
- **Stop** is where extraction runs — off the interactive hot path, after the session closes.

The write path converges on `session_end()`: it stores raw evidence, extracts candidate events, and merges them atomically. The **merge** (Stage 5) deduplicates by content hash, links provenance, finds superseded events, applies cross-type dedup, decays weights, and archives stale low-weight events. Supersession is intentionally conservative, with structured gates — temporal direction (newer supersedes older), authority precedence (lower-trust never overrides higher-trust), and provenance (same-evidence matches are restatements, not evolutions) — applied *before* any similarity check.

A key property: no network LLM call ever runs during extraction, merge, render, or any hook hot path. The optional semantic layer is a local fastembed/ONNX model that already loads `bge-small` for recall. This is the context for the encoder rework: the backbone is *already resident* for recall, so reusing it for extraction carries zero marginal model cost.

### 2.1 The deterministic Stage-2 pipeline being replaced

The baseline pipeline has six stages: (1) a custom sentence splitter that treats fenced code blocks as atomic and respects abbreviations and markdown bullets; (2) an Aho-Corasick trie scan over the signal dictionary (≈83 trigger phrases + ≈28 marker tokens), with word-boundary post-filtering; (3) content-aware variable-width windowing to capture rationale around each match; (4) a heuristic hard/soft constraint classifier driven by signal strength, repetition, authority, and domain markers against a tuned 0.85 threshold; (5) git-diff augmentation producing `COMPONENT_STATUS` events; and (6) SHA-256 content hashing over a normalized key phrase for idempotent dedup.

Its documented v1 ambition was 80% precision / 60% recall. The Relay S1 measurement shows where the keyword approach actually lands on a hard, recall-heavy session, and why the hard/soft and decision/rejection distinctions — the semantic boundaries — are where it fails.

## 3. Method

### 3.1 The shared backbone

The backbone is `BAAI/bge-small-en-v1.5`: a BERT encoder with 12 layers, 384 hidden dimensions, 33.4M parameters, a 512-token limit, CLS pooling, and an MIT license. It is already loaded for recall embeddings, so it is free to reuse. Its MTEB profile shows latent strength on exactly the two hard tasks at issue — Pair Classification 84.9 and STS 81.6 — that a linear probe over the *frozen* pooled vector cannot reach. That gap is the motivation for fine-tuning rather than only probing.

### 3.2 Salience and typing heads

**v1 — frozen linear head.** The first head is a linear classifier over the frozen, L2-normalized pooled embedding — a cosine-prototype classifier. It is fit in closed form via ridge regression (`W = (XᵀX + λI)⁻¹XᵀY`), not SGD, so identical seeds produce byte-identical weights and therefore deterministic logits. The label vocabulary is six classes: `NOISE`, `DECISION`, `CONSTRAINT_HARD`, `CONSTRAINT_SOFT`, `APPROACH_ABANDONED_DO_NOT_RETRY`, `THREAD`. `NOISE` is a first-class label the pipeline drops. At inference the head is a single numpy matmul over a bias-augmented 385-dimensional vector (`heads/salience_v1.npz`, a `(385, C)` weight matrix where row 384 is the bias), with an argmax over six classes. If the head file is missing or the embedding model is not resident, `classify()` returns `None` and the caller falls back to the legacy keyword path — the head never blocks extraction.

**v2 — SetFit contrastive fine-tune.** The frozen head's ceiling (Section 6) motivated fine-tuning the *body*. We use SetFit: phase 1 contrastively fine-tunes the encoder so semantically adjacent classes are pulled apart in vector space; phase 2 fits the classifier head on the fine-tuned body's embeddings. Training is on CPU with `BAAI/bge-small-en-v1.5` as the base, default 800 max steps, batch size 16, oversampling sampling strategy, seed 0. The held-out split is the *same* fixed 20% of seeds+generated used by the frozen-head A/B, so the numbers are directly comparable. The fine-tuned head is exported to the same `(385, C)` numpy format (`heads/salience_v2.npz`).

### 3.3 Cross-encoder for supersession

Bi-encoder cosine cannot separate "this corrects that" from "this is a different decision in the same area," because same-domain corrections and unrelated same-domain decisions cluster together in embedding space (Section 6, F3). A cross-encoder reads both sentences jointly with full attention and scores the relation directly. The training script re-heads `BAAI/bge-small-en-v1.5` to two classes (binary `should_supersede = relation != "unrelated"`, matching the eval gate), staying on-thesis with the shared backbone family and CPU-light. Default training is 3 epochs, batch size 16, 15% validation split, seed 0, with 10% warmup. The bias is toward **precision** — a false supersession deletes a still-valid decision — so the operating threshold is chosen from the validation split as the upper edge of the maximum-F1 plateau, just below the lowest validation positive.

### 3.4 Deterministic ONNX runtime

The v2 head was fit on the *fine-tuned* body's embeddings, so the runtime must reproduce that body exactly while staying torch-free. The export script wraps the fine-tuned encoder as `BERT → CLS pooling → L2 normalize` (the bge/SetFit `encode()` pipeline) and exports it to ONNX (opset 14, legacy TorchScript exporter to avoid the onnxscript dependency). It validates the exported graph against `SetFitModel.encode()` before saving. At inference, `salience_v2` runs `onnxruntime` (CPU execution provider) over `body.onnx` plus the `tokenizers` fast tokenizer (already a fastembed dependency) plus the numpy linear head — a deterministic forward pass, no PyTorch. The decoupled design is deliberate: the v2 fine-tuned body serves *classification only*; recall embeddings stay on stock fastembed, so no re-embed or backfill is required. The runtime path is wired behind a `MEMLORA_EXTRACTOR` environment variable supporting `legacy`, `v1`, `v1-broad`, `v2`, and `v2-broad` modes. Plain modes (`v1`/`v2`) filter and re-type the legacy candidate set; `-broad` modes classify every prose sentence.

### 3.5 Synthetic data generation

All training data is synthetic, generated once offline via the OpenAI API (a paid-once, offline cost — not an inference-time dependency). Generation spans a broad global pool of **32 project domains** (across languages, platforms, and business domains) × **22 software facets** (data modeling, auth, API design, error handling, observability, security, concurrency, deployment, and so on), so the model learns the *task*, not a stack. Two purpose-built corpora support the two hard problems:

- **Hard-negative twins (A1).** Minimal pairs that share a topic's vocabulary and differ *only* along a confused axis: hard vs soft constraint (deontic strength), decision vs rejection (polarity), decision vs constraint (modality), and fact vs noise (salience). Independent per-class examples cannot teach these boundaries; minimal pairs can.
- **Pairwise supersession (A2).** Pairs labeled with the relation the newer statement bears to the older one — `supersedes`, `refines`, `subsumes`, `contradicts`, `unrelated` — with `unrelated` over-weighted and deliberately seeded with *same-area* hard negatives (two different decisions both about the database, both about auth, both about caching).

Every generator enforces leakage discipline: any generated item matching a held-out gold fixture (both orderings, for pairs) is dropped.

## 4. Experimental Setup

**Held-out gates, version-agnostic scoring.** Both stages are scored against frozen gold fixtures the models never train on. Scoring is text-based and version-agnostic so the legacy regex extractor and any future encoder head are measured on the identical yardstick.

- **Stage-2 (extraction):** `tests/fixtures/relay_s1_gold.json` + `scripts/eval_extraction.py`. The harness scores token-overlap *present-recall*, token+type *clean-recall*, precision, hard-miss count, and a noise-rate breakdown (echo / meta / dup / truncated / fragment / mistyped). The gold fixture has 21 ground-truth facts. The Relay S1 session is held out and never trained on. The v1 targets encoded in the fixture are: precision ≥ 85%, clean-recall ≥ 70%, hard-miss ≤ 3, noise-rate ≤ 10%, with zero echo/meta/dup/truncated events and an event count in [18, 28].
- **Stage-5 (supersession):** `tests/fixtures/supersession_pairs_gold.json` + `scripts/eval_supersession.py`. The fixture has **21 relation-labeled pairs** (8 unrelated, 6 supersedes, 3 refines, 2 subsumes, 2 contradicts), including **5 same-area GUARD negatives**. The scorer reports precision, recall, and `guard_fp` (guard negatives wrongly flagged — the dangerous class, since a false supersession hides a valid decision). The predicate is pluggable, so the lexical baseline and the cross-encoder are scored on the same 21 pairs. Targets: precision ≥ 95%, recall ≥ 70%, `guard_fp` = 0.

**Corpora.** The corpora built this sprint (with measured zero leakage against the held-out fixtures):

| Corpus | Size | Distribution | Feeds |
|---|---|---|---|
| Pairwise supersession (A2) | 2,200 | unrel 720 / sup 470 / ref 370 / sub 320 / con 320 | C1 |
| Hard-negative twins (A1) | 1,840 | DEC 480 / HARD 470 / NOISE 390 / SOFT 200 / GRAVE 180 / THREAD 120 | B2 |
| Prior type corpus | 1,111 | 6-class (seeds 161 + generated 950) | B1/B2 |

**Training compute.** All fine-tuning is CPU-only, consistent with the local-first claim. SetFit typing: ≤800 steps, batch 16, oversampling, seed 0. Cross-encoder: 3 epochs, batch 16, 15% val, 10% warmup, seed 0.

A latency/footprint table (CPU inference milliseconds, model MB) is *pending* — it is on the ledger's completion checklist but no measured numbers are recorded yet.

## 5. Results

All numbers below are reproduced verbatim from the ledger. Cells marked *pending* / *future* are not yet measured.

### 5.1 Stage-2 extraction — Relay S1 held-out gold

| Approach | Precision | Present-recall | Clean-recall | Noise | Hard-miss |
|---|---|---|---|---|---|
| Legacy regex/Aho-Corasick (baseline, hand-scored) | 38% | ~58% | 19% | 42% | 7 |
| + Segmenter fix (keep-whole-fact, list tags) | 78% | — | — | 22% | — |
| + Learned salience head, **broad mode** (frozen+linear) | **93%** | **95%** | **57%** | **7%** | **1** |
| Fine-tuned backbone (B2, SetFit) | *pending* | *pending* | target ≥75% | ≤6% | ≤1 |
| + Decoder span repair (v3) | *future* | — | ceiling ~88–90% | — | — |

The frozen+linear head in broad mode is a large jump over the keyword baseline on precision (38% → 93%), present-recall (~58% → 95%), clean-recall (19% → 57%), noise (42% → 7%), and hard-miss (7 → 1). The B2 fine-tuned-backbone extraction numbers on this gate are *pending* (the listed values are targets, not measurements).

### 5.2 Typing (6-way)

| Setting | Accuracy |
|---|---|
| Frozen+linear CV (seeds+generated, 6-way) | ~91% |
| Frozen+linear, in-dist held-out (n=222) | **90.5%** |
| Frozen+linear + hard-neg twins (in-dist held-out) | **84.2%** (down) |
| **SetFit fine-tuned + twins (in-dist held-out)** | **92.3%** (up) |
| Real transcript (Taskflow / MOB_C) typing | 8/11 (~73%) |
| Fine-tuned backbone, real-transcript typing | *pending D2 wiring* (target ≥88%) |

The frozen-then-fine-tuned arc — 90.5% → 84.2% (twins added) → 92.3% (twins under fine-tuning) — is the central evidence for F2 (Section 6). Fine-tuned real-transcript typing is *pending*.

### 5.3 Stage-5 supersession — held-out pair gold (21 pairs)

| Approach | Precision | Recall | guard_fp |
|---|---|---|---|
| Lexical (Jaccard ∪ Levenshtein) + subject-keying | **100%** | **38%** | **0** |
| + bi-encoder cosine @0.75 | ~100% | ~+0 (semantically blind) | 0 |
| Cross-encoder (C1) @ natural threshold ≈0.963 | **100%** | **85%** | **0** |
| Cross-encoder (C1) @ synthetic-val-selected threshold (0.05) | 62% | 100% | 5 |

At its natural confidence edge (~0.963) the cross-encoder meets every gold target: precision 100%, recall 85% (up from the lexical baseline's 38%), and zero guard false positives. Per-pair, the classic same-area guards are cleanly rejected — `composite-PK vs UUID-PK` scores 0.058, `Postgres-store vs Redis-counters` 0.064, `SHA256-keys vs argon2id-passwords` 0.279 — against true relations at 0.96–0.97. The catch is the threshold *selected from synthetic validation*: it lands at ~0.05, collapsing to 62% precision and 5 guard false positives on the real guards (F5).

### 5.4 D2 — deployable runtime (local, torch-free)

| Check | Result |
|---|---|
| ONNX export parity vs SetFit `encode()` (CLS+L2norm) | **max_abs 8.9e-08, cosine 1.000000** |
| ONNX-runtime v2 head vs torch SetFit (60 real Relay texts) | **60/60 labels match** |
| `MEMLORA_EXTRACTOR=v2` / `v2-broad` wired into pipeline | done; end-to-end extract OK |
| Extraction regression suite | **314 passed, 1 xfailed** |

The torch-free runtime reproduces the fine-tuned body to within `8.9e-08` max absolute error (cosine 1.000000) and matches torch labels on all 60 real Relay texts, confirming the determinism the content-hash/cache contract depends on. The confidence-floor is intentionally *not* enabled for v2, because F6 shows v2 already errs conservative and a floor would cut recall further.

## 6. Discussion: Findings F1–F6

**F1 — Detection is solved; typing is the gap.** Present-recall of 95% against clean-recall of 57% means fact/noise *detection* is essentially done by a linear probe; the remaining loss is *type* correctness (and faithfulness). Clean-recall is typing-bound. This reframes the problem: the encoder rework is, at its core, a typing problem.

**F2 — The frozen-backbone ceiling is real, and geometry was the bottleneck.** On a fixed in-distribution held-out, the frozen+linear head scores 90.5%; adding 1,840 hard-negative deontic/polarity twins *lowers* it to 84.2%. The twins sit on top of each other in the frozen space — different labels, near-identical vectors — so to a linear head they read as label noise. They become signal only after the backbone is contrastively fine-tuned. SetFit confirms this on the *same* held-out: 90.5% (frozen) → 84.2% (frozen + twins) → **92.3%** (fine-tuned + twins), i.e. +8.1 over frozen+twins and +1.8 over frozen. The geometry — not the labels, not the head architecture — was the bottleneck, exactly as the thesis predicted. This is the paper's most important methodological result: hard negatives are worthless, even harmful, until the representation can place them apart.

**F3 — Cosine cannot do supersession; the relation needs joint encoding.** Lexical overlap plus subject-keying is perfectly precise (100%) but blind to paraphrased corrections, contradictions, and subsumption — recall 38%. A bi-encoder cosine axis adds essentially nothing at a safe threshold, because same-domain corrections and unrelated same-domain decisions are non-separable in vector space (a real revalidation found a genuine correction at cosine 0.658 sitting between unrelated decisions at 0.654 and 0.633). This is the textbook cross-encoder case, and the cross-encoder delivers (Section 5.3).

**F4 — Data diversity is the gating resource.** Both corpora span 32 domains × 22 facets to generalize globally. The corpus, not the architecture, is the bottleneck — a conclusion now backed by F2 (the architecture was capable once given separable geometry and the right data).

**F5 — The cross-encoder works; synthetic-val threshold *calibration* does not.** The C1 cross-encoder separates the same-area guards that cosine cannot (0.06 vs 0.96) and meets every gold target at its natural confidence edge (~0.963). But the operating threshold cannot be *selected* from the synthetic validation set: synthetic negatives are too easy, so a max-F1 selection lands at ~0.05, which scores 62% precision and 5 guard false positives on real guards. This is the same "easy-negative" methodology trap recorded elsewhere in the project's embedding analysis. Shipping a fragile threshold is unacceptable here, because a false supersession silently deletes a valid decision. The fix, before integration, is two-fold: (a) harden the negative corpus with "complementary same-area" pairs, and (b) apply temperature calibration so a deployable threshold is selectable without gold-peeking.

**F6 — In-distribution typing gain ≠ OOD clean-recall gain (a caution).** Re-typing the legacy extractor's 60 real Relay S1 candidates (out-of-distribution text) with each head: v1 frozen gives precision 66 / clean-recall 33 / noise 34; v2 SetFit gives **precision 81 (+16) / clean-recall 24 (−10) / noise 19 (−16)**. The fine-tune made the head markedly more precise and less noisy but more conservative, *lowering* clean-recall — the opposite of the in-distribution result in F2. Two caveats temper this: it is FILTER mode over a candidate set whose own present-recall is only 52% (so recall is candidate-capped, not head-capped), and the drop looks like a conservatism/threshold effect. A fair v2 verdict needs a BROAD-mode evaluation over the raw transcript (no fixture exists yet) plus confidence-floor tuning. The synthetic→real typing transfer is partial and must not be assumed from in-distribution numbers — the same lesson as F5, on the typing side.

Together, F2, F5, and F6 form the spine of an honest narrative: the architecture works when the representation is right (F2), but generalization and calibration to *real, hard* distributions are unsolved (F5, F6), and we mark them as such rather than overclaiming.

## 7. Related Work

**mem0.** mem0 provides a memory layer for LLM agents with extraction and retrieval of salient facts, typically backed by a hosted vector store and an LLM-in-the-loop extraction step. CogniKernel differs in being local-first and LLM-free at inference: extraction is a deterministic encoder forward pass on the developer's machine, and the store is per-project SQLite with no network dependency.

**Zep (temporal knowledge graph).** Zep maintains a temporal knowledge graph of conversation memory, tracking how facts change over time with validity intervals. CogniKernel shares the temporal-evolution concern — its supersession stage is precisely about a newer memory replacing an older one — but resolves it with structured gates (temporal/authority/provenance) plus a small cross-encoder, rather than a hosted graph service, and keeps full provenance in an event-sourced log for audit and replay.

**GLiNER / GLiNER2 and GLiREL.** GLiNER is a generalist, zero-shot NER model using a BERT-family encoder to extract arbitrary entity types from a label prompt; GLiNER2 extends to richer schema extraction, and GLiREL targets zero-shot relation extraction. These are close cousins of the thesis — small encoders repurposed for flexible discriminative extraction. CogniKernel's sub-tasks (salience, six-way typing, pairwise supersession) are a fixed, domain-tuned instantiation of the same idea, optimized for deterministic, offline-trained, on-device serving rather than zero-shot flexibility.

**SetFit.** SetFit is a sample-efficient method for fine-tuning sentence-transformer encoders via contrastive pretraining followed by a lightweight classifier head, achieving strong few-shot text classification without prompts. It is the engine of our v2 typing head, and our F2 result is a concrete demonstration of *why* it helps: it fixes the geometry so hard negatives become separable.

**BGE and bge-reranker / cross-encoders.** The BGE family provides strong open embedding models (we use `bge-small-en-v1.5` as the shared backbone), and bge-reranker exemplifies the cross-encoder pattern for sentence-pair scoring. Cross-encoders trade the bi-encoder's precompute-and-cache efficiency for joint attention over both inputs, which is exactly what supersession requires (F3). CogniKernel applies a cross-encoder narrowly — only to candidate pairs surfaced by the merge — keeping the cost bounded.

CogniKernel's positioning across all of these is consistent: local, deterministic, free at inference, and provenance-aware. The novelty is not any single model but the engineering discipline — one shared offline-trained backbone, numpy/ONNX serving with byte-stable logits, event-sourced provenance, and held-out version-agnostic gates — applied to cross-session coding-agent memory.

## 8. Limitations and Threats to Validity

- **Single held-out session.** The extraction gate is one session (Relay S1, an architecture discussion). It is recall-heavy and representative of the failure mode we care about, but a single session cannot establish generalization. Broader held-out coverage is needed.
- **Synthetic-data realism.** All training data is LLM-generated. F5 and F6 are both, at root, failures of synthetic data to model the *hard* tail of the real distribution — synthetic negatives are too easy to calibrate a threshold (F5), and synthetic typing gains do not fully transfer to real transcripts (F6).
- **No raw-transcript broad-mode OOD evaluation yet.** The fair test of v2 typing — BROAD mode over a raw transcript rather than FILTER mode over a capped candidate set — has no fixture and is unmeasured.
- **The C1 calibration gap.** The cross-encoder meets all targets at its natural threshold but the threshold cannot be selected without gold-peeking. Until calibration is hardened, the cross-encoder is not deployable for automatic supersession.
- **Debatable gold labels.** The supersession gold includes a near-positive "complementary same-area" pair (U7: a priority-ordered router vs round-robin within a tier, labeled `unrelated`/guard) whose correct relation is genuinely debatable. A small number of such labels can move a 21-pair score noticeably.
- **Pending measurements.** Several headline cells (B2 extraction P/clean-recall on Relay; fine-tuned real-transcript typing; latency/footprint) are not yet measured and are marked pending throughout.

## 9. Conclusion and Future Work

We reframed deterministic keyword extraction as a set of discriminative sub-tasks served by one small, offline-trained, deterministically-served encoder backbone — preserving CogniKernel's local-first, free, cache-stable contract. A frozen linear head already lifts extraction far above the keyword baseline (38% → 93% precision, 7 → 1 hard miss on the held-out session) and essentially solves detection (95% present-recall). Fine-tuning the backbone resolves the typing bottleneck on the in-distribution gate (90.5% → 92.3% with hard negatives that previously hurt), confirming that geometry, not labels, was the limit. A cross-encoder solves the supersession separability that cosine cannot (recall 38% → 85% at 100% precision on the gold gate). We deploy the fine-tuned body torch-free with byte-level ONNX parity.

The honest gaps define the roadmap:

1. **C1 calibration hardening → C2 gated integration.** Harden the negative corpus with complementary same-area pairs and apply temperature calibration so a deployable supersession threshold is selectable without gold-peeking; only then wire the cross-encoder into the merge behind the existing structured gates.
2. **Broad-mode OOD evaluation and v2 verdict.** Build a raw-transcript broad-mode fixture and tune the confidence floor, to settle whether v2's in-distribution typing gain transfers (F6).
3. **List aggregation and a v3 grammar-locked decoder.** Add the deferred generative sub-tasks — canonicalization and list aggregation — and a grammar-constrained decoder for span repair, targeting the ~88–90% clean-recall ceiling the discriminative heads alone cannot reach.

The discipline throughout is the same one this paper tries to model: report the negative and mixed results (F2's harmful-then-helpful twins, F5's calibration failure, F6's non-transfer) as first-class findings, and mark every unmeasured number as pending.
