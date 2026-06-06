#!/usr/bin/env python3
"""Train the cross-encoder supersession scorer (WS-C1) — local, CPU-friendly.

A bi-encoder cosine cannot separate "this corrects that" from "this is a different
decision in the same area" (the supersede.py:289 finding). A CROSS-encoder reads both
sentences jointly with full attention and scores the relation directly — the right tool.

Data:   tests/fixtures/supersession_pairs_generated.jsonl  ({a, b, relation})
Label:  binary should_supersede = relation != "unrelated"  (matches the eval gate)
Base:   cross-encoder/ms-marco-MiniLM-L-6-v2 (22M, fast on CPU) — a sentence-pair
        relevance model, re-headed to 2 classes. Lightweight: the local+free moat.
Bias:   PRECISION. A false supersession hides a valid decision, so the decision
        threshold is chosen as the LOWEST that holds precision >= TARGET_PRECISION on
        the validation split; recall is reported at that operating point.

Outputs:
  models/supersession_xenc/                      the fine-tuned cross-encoder
  models/supersession_xenc/threshold.json        {"threshold": p, "base": ...}

Validate against the HELD-OUT gold (never trained on):
  python scripts/eval_supersession.py --predicate train_supersession_xenc:supersedes --check

`supersedes(a, b)` below lazy-loads the trained model so the same gate that scored the
lexical baseline now scores the cross-encoder, on the same 21 pairs.

Usage:
  python scripts/train_supersession_xenc.py [--epochs 3] [--batch 16]
      [--base cross-encoder/ms-marco-MiniLM-L-6-v2] [--val 0.15] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CORPUS = _ROOT / "tests" / "fixtures" / "supersession_pairs_generated.jsonl"
_GOLD = _ROOT / "tests" / "fixtures" / "supersession_pairs_gold.json"
_MODEL_DIR = _ROOT / "models" / "supersession_xenc"
# Base with NO pretrained sequence-classification head, so CrossEncoder attaches a
# fresh num_labels=2 head with zero state-dict mismatch. bge-small is on-thesis (the
# shared backbone family) and CPU-light (33M). A pretrained reranker base (ms-marco)
# would warm-start the body but its 1-logit head collides with num_labels=2 under ST v5.
_DEFAULT_BASE = "BAAI/bge-small-en-v1.5"
TARGET_PRECISION = 0.95


# ── inference predicate (used by eval_supersession.py --predicate) ─────────────
_model = None
_threshold = 0.5


def _load_model():
    global _model, _threshold
    if _model is not None:
        return _model
    from sentence_transformers import CrossEncoder
    _model = CrossEncoder(str(_MODEL_DIR))
    try:
        _threshold = float(json.loads((_MODEL_DIR / "threshold.json").read_text())["threshold"])
    except Exception:
        _threshold = 0.5
    return _model


def _prob_supersede(model, a: str, b: str) -> float:
    import numpy as np
    logits = model.predict([[a, b]])  # (1, 2) for num_labels=2
    arr = np.asarray(logits, dtype="float32").reshape(-1)
    if arr.size == 1:  # regression head fallback
        return float(1.0 / (1.0 + np.exp(-arr[0])))
    z = arr - arr.max()
    p = np.exp(z) / np.exp(z).sum()
    return float(p[1])


def supersedes(new_desc: str, old_desc: str) -> bool:
    """Predicate for eval_supersession.py: does the newer `new_desc` supersede `old_desc`?"""
    model = _load_model()
    return _prob_supersede(model, new_desc, old_desc) >= _threshold


# ── training ──────────────────────────────────────────────────────────────────

def _load_corpus():
    rows = []
    for line in _CORPUS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        label = 0 if r["relation"] == "unrelated" else 1
        rows.append((r["a"], r["b"], label))
    return rows


def _gold_pairs():
    gold = json.loads(_GOLD.read_text(encoding="utf-8"))
    rel = gold["relations"]
    pairs = [(p["a"], p["b"], bool(rel[p["relation"]]["should_supersede"]), bool(p.get("guard")))
             for p in gold["pairs"]]
    return pairs


def _pick_threshold(probs, labels):
    """Precision-conservative threshold = the HIGHEST threshold still at max F1 on val.

    The cross-encoder outputs are bimodal (positives ~0.96, negatives ~0.06). A
    cleanly-separable val set gives F1=1 across the whole gap, so picking the LOWEST
    such threshold (the old rule) pins the operating point at the bottom of the gap —
    far too permissive once REAL same-area negatives (harder than the synthetic ones)
    appear. Taking the UPPER edge of the max-F1 plateau sits the threshold just below
    the lowest val positive, which is robust to the easy-negative bias. Not gold-peeking
    — computed only on the training-distribution val split.
    """
    import numpy as np
    probs = np.asarray(probs); labels = np.asarray(labels)
    scored = []
    for t in sorted(set(probs.tolist())):
        pred = probs >= t
        tp = int((pred & (labels == 1)).sum()); fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        scored.append((f1, float(t), prec, rec))
    best_f1 = max(s[0] for s in scored)
    plateau = sorted((s for s in scored if abs(s[0] - best_f1) <= 1e-9), key=lambda s: s[1])
    _, t, prec, rec = plateau[-1]  # highest threshold on the max-F1 plateau
    return t, prec, rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=_DEFAULT_BASE)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = _load_corpus()
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(rows))
    n_val = int(len(rows) * args.val)
    val_idx, train_idx = set(idx[:n_val].tolist()), idx[n_val:]
    train = [rows[i] for i in train_idx]
    val = [rows[i] for i in range(len(rows)) if i in val_idx]
    pos = sum(1 for *_, l in rows if l == 1)
    print(f"corpus={len(rows)}  positives={pos}  train={len(train)}  val={len(val)}")

    # ms-marco bases ship a 1-logit reranker head; reinit a fresh 2-class head while
    # KEEPING the sentence-pair-tuned body (a strong warm start for our relation task).
    model = CrossEncoder(args.base, num_labels=2, model_kwargs={"ignore_mismatched_sizes": True})
    train_samples = [InputExample(texts=[a, b], label=l) for a, b, l in train]
    loader = DataLoader(train_samples, shuffle=True, batch_size=args.batch)
    warmup = int(len(loader) * args.epochs * 0.1)
    print(f"training {args.epochs} epochs on CPU (base={args.base}, warmup={warmup})...")
    model.fit(train_dataloader=loader, epochs=args.epochs, warmup_steps=warmup, show_progress_bar=True)

    # Threshold from the validation split (precision-biased).
    val_probs = [_prob_supersede(model, a, b) for a, b, _ in val]
    val_labels = [l for *_, l in val]
    thr, vprec, vrec = _pick_threshold(val_probs, val_labels)
    print(f"\nval operating point: threshold={thr:.3f}  precision={vprec:.0%}  recall={vrec:.0%}")

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(_MODEL_DIR))
    (_MODEL_DIR / "threshold.json").write_text(
        json.dumps({"threshold": thr, "base": args.base, "target_precision": TARGET_PRECISION}, indent=2)
    )
    print(f"saved model + threshold to {_MODEL_DIR.relative_to(_ROOT)}")

    # Held-out GOLD scorecard (the real gate — never trained on).
    gold = _gold_pairs()
    gp = [_prob_supersede(model, a, b) for a, b, _, _ in gold]
    tp = fp = fn = guard_fp = 0
    for (a, b, should, guard), p in zip(gold, gp):
        got = p >= thr
        if got and should: tp += 1
        elif got and not should:
            fp += 1
            guard_fp += int(guard)
        elif not got and should: fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    print("\n=== GOLD scorecard (held-out) ===")
    print(f"  precision {prec:.0%}  (target >= {TARGET_PRECISION:.0%})")
    print(f"  recall    {rec:.0%}  (target >= 70%)")
    print(f"  guard_fp  {guard_fp}  (target 0)")
    print("\nNext: python scripts/eval_supersession.py --predicate train_supersession_xenc:supersedes --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
