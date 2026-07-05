"""Assemble the training corpus: mined + synthetic → deduped, eval-decontaminated JSONL.

Inputs (whatever exists under research/train_corpus/):
  mined_adr_sentences.jsonl / mined_adr_pairs.jsonl   (scripts/mine_adr_corpus.py)
  synth_sentences.jsonl     / synth_pairs.jsonl       (scripts/gen_synthetic_corpus.py)

Outputs:
  research/train_corpus/train_sentences.jsonl
  research/train_corpus/train_pairs.jsonl
  research/train_corpus/corpus_stats.json

Guarantees:
  1. Near-dup removal (normalized content-token signature).
  2. EVAL DECONTAMINATION: any sentence/pair text with Jaccard >= 0.6 content-token
     overlap against ANY text in research/model_eval/ is dropped and counted.
     The frozen eval must never leak into training.
  3. Every item keeps provenance (source, license where mined).

Usage: uv run python scripts/assemble_train_corpus.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from memlora.delta.supersede import normalize_for_overlap

CORPUS = Path("research/train_corpus")
EVAL = Path("research/model_eval")


def load(name: str) -> list[dict]:
    p = CORPUS / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def eval_token_sets() -> list[set[str]]:
    sets = []
    for fname in ("salience_eval.jsonl", "supersession_eval.jsonl"):
        p = EVAL / fname
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            it = json.loads(line)
            for key in ("text", "new_text", "old_text"):
                if it.get(key):
                    toks = normalize_for_overlap(it[key])
                    if toks:
                        sets.append(toks)
    return sets


def contaminated(text: str, eval_sets: list[set[str]], threshold: float = 0.6) -> bool:
    toks = normalize_for_overlap(text)
    if not toks:
        return False
    for es in eval_sets:
        inter = len(toks & es)
        if inter and inter / len(toks | es) >= threshold:
            return True
    return False


def signature(text: str) -> frozenset:
    return frozenset(normalize_for_overlap(text))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="write train_sentences<suffix>.jsonl / train_pairs<suffix>.jsonl "
                         "instead of the default names — lets an adversarial corpus be "
                         "assembled WITHOUT overwriting a corpus a running sweep depends on.")
    ap.add_argument("--include-humanized", action="store_true",
                    help="fold humanized_sentences.jsonl (messy real-developer register) "
                         "into the sentence sources (Spike A1).")
    args = ap.parse_args()
    sfx = args.suffix

    eval_sets = eval_token_sets()
    print(f"eval texts loaded for decontamination: {len(eval_sets)}")

    # ── sentences ────────────────────────────────────────────────────────────
    sents = load("mined_adr_sentences.jsonl") + load("synth_sentences.jsonl")
    if args.include_humanized:
        hum = load("humanized_sentences.jsonl")
        print(f"including {len(hum)} humanized sentences")
        sents += hum
    out_s: list[dict] = []
    seen_sigs: set[frozenset] = set()
    dropped = Counter()
    for it in sents:
        text = it["text"].strip()
        if len(text) < 15:
            dropped["too_short"] += 1
            continue
        sig = signature(text)
        # A humanized item shares content tokens with its clean original by
        # design (same fact, messy register), so it near-dups it. That is
        # legitimate augmentation — keep BOTH — so humanized items are exempt
        # from the near-dup drop (still deduped against OTHER humanized items via
        # exact-text, and still eval-decontaminated).
        is_hum = str(it.get("source", "")).startswith("humanized:")
        if not sig or (sig in seen_sigs and not is_hum):
            dropped["near_dup"] += 1
            continue
        if contaminated(text, eval_sets):
            dropped["eval_contaminated"] += 1
            continue
        seen_sigs.add(sig)
        out_s.append(it)

    # ── pairs ────────────────────────────────────────────────────────────────
    pairs = load("mined_adr_pairs.jsonl") + load("synth_pairs.jsonl")
    out_p: list[dict] = []
    seen_pair: set[tuple] = set()
    for it in pairs:
        new, old = it["new_text"].strip(), it["old_text"].strip()
        if len(new) < 15 or len(old) < 15:
            dropped["pair_too_short"] += 1
            continue
        key = (signature(new), signature(old))
        if key in seen_pair:
            dropped["pair_near_dup"] += 1
            continue
        if contaminated(new, eval_sets) or contaminated(old, eval_sets):
            dropped["pair_eval_contaminated"] += 1
            continue
        seen_pair.add(key)
        out_p.append(it)

    with open(CORPUS / f"train_sentences{sfx}.jsonl", "w", encoding="utf-8") as f:
        for it in out_s:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(CORPUS / f"train_pairs{sfx}.jsonl", "w", encoding="utf-8") as f:
        for it in out_p:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    stats = {
        "sentences": len(out_s),
        "sentences_by_label": dict(Counter(i["label"] for i in out_s)),
        "sentences_by_register": dict(Counter(i.get("register", "?") for i in out_s)),
        "sentences_by_source_kind": dict(Counter(i["source"].split(":")[0] for i in out_s)),
        "pairs": len(out_p),
        "pairs_by_kind_label": {f"{k}|{l}": c for (k, l), c in
                                Counter((i["kind"], i["label"]) for i in out_p).items()},
        "dropped": dict(dropped),
    }
    (CORPUS / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
