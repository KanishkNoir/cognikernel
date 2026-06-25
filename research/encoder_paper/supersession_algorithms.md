# Beyond lexical + cross-encoder: the algorithmic space for memory supersession

> Research note prompted by the R5 result. Status: survey + conclusions. The goal is to
> establish, from the ML/maths literature, why the lexical+cross-encoder framing is a
> *ceiling* and what algorithmic families can lift CogniKernel's memory-evolution quality.

## 1. The category error (why the ceiling exists)

Lexical overlap (Jaccard/Levenshtein) and a learned cross-encoder both compute a **similarity
metric over a holistic pair** `sim(a, b) ∈ [0,1]`. Empirically (R5), this **conflates topical
relatedness with supersession**: on the real Relay store the cross-encoder scored ~0.97 for
almost any same-topic pair, "superseding" 48 unrelated decisions; no threshold separated true
supersessions from complementary same-topic facts (both ≈0.97).

This is a **category error**, statable in maths:

- **Similarity** is a *symmetric metric on an embedding manifold*: `d(a,b)=d(b,a)`, governed by
  topical proximity. Nearest-neighbour retrieval lives here.
- **Supersession** is an *asymmetric, structured relation*: `a ⊳ b` iff `key(a)=key(b)` ∧
  `value(a)≠value(b)` ∧ `t(a) > t(b)`. It is a **partial order over (key, value, time)**, not a
  point distance.

A metric cannot express a key-equality-plus-value-change-plus-temporal-order predicate. So
*any* similarity model (lexical or cross-encoder) is the wrong hypothesis class — it will always
trade recall (catch paraphrases) against precision (reject same-topic), because it never sees
the **key** or the **value** as separate objects. The R5 hybrid (similarity AND lexical co-fire)
is a patch on the symptom, not the structure.

## 2. The algorithm families that model the structure

### (a) Decomposition: (subject/key, predicate, value) triples
Represent each event as a relational triple `⟨subject, predicate, object/value⟩`
([knowledge-graph triplets](https://www.emergentmind.com/topics/knowledge-graph-triplets);
[CasRel cascade tagging](https://arxiv.org/pdf/1909.03227)). Supersession then becomes
**key-equality (`subject,predicate` match) + value-difference** — a structured comparison, not a
similarity. The cross-encoder failed *because* it never extracted the key; explicit extraction
(GLiNER/GLiREL-class span+relation models, CPU-cheap, Apache-licensed) removes the conflation.
CogniKernel already has a primitive of this (`derive_subject` regexes); the expansion is a
*learned* subject/relation extractor.

### (b) NLI: the value-comparison as entailment/contradiction
Once two events share a key, deciding supersession is asking whether the new **contradicts** or
**refines/subsumes** the old — i.e. **Natural Language Inference** (entailment / contradiction /
neutral), not similarity ([NLI](https://www.emergentmind.com/topics/natural-language-inference-nli);
[Knowledge Conflicts for LLMs: a Survey, EMNLP 2024](https://github.com/pillowsofwind/Knowledge-Conflicts-Survey)).
NLI's `contradiction` ≈ value-flip (D2/C-class), `entailment` ≈ subsumption (B-class), `neutral`
≈ complementary same-topic (the exact U7/U8 negatives the cross-encoder false-fired on). A 3-way
NLI head is a *better-aligned hypothesis class* than a 2-way supersede/not similarity head — it is
trained to separate contradiction from neutral, which is precisely our failure mode. Caveat from
the literature: NLI ignores temporal context unless dates/order are canonicalised — so NLI is the
*value* axis, still gated by the temporal axis.

### (c) Bi-temporal knowledge graph (the lossless, structured frame)
[Zep / Graphiti](https://arxiv.org/abs/2501.13956) model memory as a graph whose **edges carry
validity intervals** (`t_valid`, `t_invalid`) plus system times (`t_created`, `t_expired`). A new
edge on the same `(subject, predicate)` **invalidates** the old edge's valid-interval rather than
deleting it. This is the decisive match for CogniKernel:
- supersession = **edge invalidation on a shared key**, made first-class (not inferred from
  similarity);
- it is **lossless** (the old edge is retained, time-bounded) — exactly the project's principle;
- "what is the value *today*?" (the Relay S5 chain probe) becomes a trivial valid-time query, not a
  ranking guess.

### (d) Belief revision / truth maintenance (the logical grounding)
Classical AGM belief revision and systems like
[SNePS](https://arxiv.org/pdf/cs/0003011) / [abductive KB update](https://arxiv.org/pdf/cs/0405076)
formalise *consistent* update under new conflicting information. They supply the invariants
(minimal change, consistency) that a production supersession policy should satisfy, and motivate
"invalidate, don't delete."

## 3. Conclusions (evidence-backed)

1. **The ceiling is structural, not a tuning problem.** Similarity models (lexical, cross-encoder)
   are the wrong hypothesis class for an asymmetric (key, value, time) relation. The R5 over-
   supersession is the empirical proof; more data/threshold-tuning cannot fix a category error.
2. **The expansion is to STRUCTURE the memory, not to grow the pair-similarity model.** The
   target architecture decomposes the problem along its true axes:
   - **Key axis** — learned subject/relation extraction (triples), replacing `derive_subject`.
   - **Value axis** — an **NLI** (contradiction/entailment/neutral) head over same-key pairs,
     replacing the supersede/not similarity head. This directly separates the U7/U8 "neutral"
     complementary negatives from real contradictions.
   - **Time axis** — a **bi-temporal valid-interval** model (Zep-style) where supersession =
     invalidate-on-shared-key. Lossless by construction.
3. **It composes with what exists and the project's principles.** The shared bge-small backbone can
   host the NLI head (3-class) and a span/relation extractor; the temporal/authority/provenance
   gates remain the always-on floor; bi-temporal invalidation *is* the lossless "down-sample at
   read" model (old edges retained, time-filtered). The encoder thesis holds — only the *head's
   objective* changes from similarity to NLI, and the *store* gains a key+valid-time index.
4. **Near-term, ship the safe hybrid; do not over-invest in the cross-encoder.** The R5 hybrid
   (`lexical OR (xenc≥0.97 ∧ jaccard≥0.3)`) is precision-safe and a modest gain — a fine interim.
   But the next real step is the **NLI + key-extraction + bi-temporal** redesign, which the maths
   says is the only path past the similarity ceiling.

## 4. Recommended next experiments (to confirm, not assume)
- Train/borrow a small **NLI head** on the bge-small backbone; re-run the gold + real-store
  supersession test as a 3-class (contradiction/entailment/neutral) problem; measure whether
  `neutral` correctly captures the complementary same-topic negatives the cross-encoder missed.
- Prototype **key extraction** (GLiNER/GLiREL or a learned subject head) and re-measure
  supersession as same-key + NLI-contradiction; compare precision/recall to the hybrid.
- Spike a **bi-temporal `event_edges`** table (valid_from/valid_to) and re-answer the Relay S5
  chain probes by valid-time query; compare to the ranking-based block.
- Decision rule throughout: **precision-first** (a false invalidation hides a valid decision) and
  **lossless** (invalidate, never delete).
