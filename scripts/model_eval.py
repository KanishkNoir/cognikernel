"""Run the current learned components against the frozen model-eval dataset.

Evaluates:
  - salience heads v1 (frozen bge-small + linear) and v2 (SetFit/ONNX):
    per-label precision/recall/F1, macro-F1, accuracy, per-register accuracy,
    confusion matrix, coverage (None returns).
  - supersession: cross-encoder (deployed hybrid = prob >= 0.97 AND jaccard >= 0.3;
    plus raw-threshold sweep for best F1 and rank-AUC), lexical baseline
    (delta.supersede.supersedes), cosine baseline (bge-small at the deployed 0.75).

Writes a timestamped results JSON with model-artifact fingerprints so a retrain
can be compared like-for-like:  research/model_eval/results_<stamp>.json

Usage: uv run --extra embedding python scripts/model_eval.py [--tag baseline]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path("research/model_eval")
LABELS = ("NOISE", "DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
          "APPROACH_ABANDONED_DO_NOT_RETRY", "THREAD")

# Deployed operating points (mirror src defaults; see supersede.py)
XENC_THRESHOLD = 0.97
XENC_JACCARD_COFIRE = 0.3
COSINE_THRESHOLD = 0.75


def _sha(path: Path) -> str:
    if not path.exists():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def artifact_fingerprints() -> dict:
    import memlora.extraction.salience as s1
    import memlora.extraction.salience_v2 as s2
    home = Path.home() / ".memlora" / "models"
    return {
        "salience_v1_head": _sha(Path(s1.__file__).parent / "heads" / "salience_v1.npz"),
        "salience_v2_head": _sha(Path(s2.__file__).parent / "heads" / "salience_v2.npz"),
        "salience_v2_body": _sha(home / "salience_v2" / "body.onnx"),
        "supersession_xenc_body": _sha(home / "supersession_xenc" / "body.onnx"),
    }


def load_jsonl(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA_DIR / name).read_text(encoding="utf-8").splitlines()]


# ── salience ─────────────────────────────────────────────────────────────────

def eval_salience(items: list[dict], head, head_name: str) -> dict:
    if not head.is_available():
        return {"available": False}
    y_true, y_pred, confs = [], [], []
    per_register = defaultdict(lambda: [0, 0])
    confusion: Counter = Counter()
    none_returns = 0
    for it in items:
        scored = head.classify_scored(it["text"])
        if scored is None:
            none_returns += 1
            continue
        label, conf = scored
        y_true.append(it["label"]); y_pred.append(label); confs.append(conf)
        confusion[(it["label"], label)] += 1
        reg = per_register[it["register"]]
        reg[1] += 1
        if label == it["label"]:
            reg[0] += 1
    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n if n else 0.0
    per_label = {}
    f1s = []
    for lab in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = tp + fn
        per_label[lab] = {"precision": round(prec, 3), "recall": round(rec, 3),
                          "f1": round(f1, 3), "support": support}
        if support:
            f1s.append(f1)
    return {
        "available": True,
        "n_scored": n,
        "none_returns": none_returns,
        "accuracy": round(acc, 3),
        "macro_f1": round(sum(f1s) / len(f1s), 3) if f1s else 0.0,
        "mean_confidence": round(sum(confs) / n, 3) if n else 0.0,
        "per_label": per_label,
        "per_register_accuracy": {
            r: {"acc": round(c / t, 3), "n": t} for r, (c, t) in sorted(per_register.items())
        },
        "confusion_top_errors": [
            {"true": t, "pred": p, "n": c}
            for (t, p), c in confusion.most_common(30) if t != p
        ][:12],
    }


# ── supersession ─────────────────────────────────────────────────────────────

def _pr_f1(preds: list[int], labels: list[int]) -> dict:
    tp = sum(1 for p, l in zip(preds, labels) if p and l)
    fp = sum(1 for p, l in zip(preds, labels) if p and not l)
    fn = sum(1 for p, l in zip(preds, labels) if not p and l)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def _rank_auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return 0.0
    wins = sum(1 for p in pos for q in neg if p > q) + 0.5 * sum(
        1 for p in pos for q in neg if p == q)
    return round(wins / (len(pos) * len(neg)), 3)


def eval_supersession(pairs: list[dict]) -> dict:
    from memlora.delta import supersede_xenc
    from memlora.delta.supersede import jaccard_similarity, supersedes
    from memlora.embedding.input import embedding_input
    from memlora.embedding.model import embed_text, ensure_ready

    labels = [p["label"] for p in pairs]
    out: dict = {"n_pairs": len(pairs), "by_kind": dict(Counter(p["kind"] for p in pairs))}

    # Lexical baseline (the always-on candidate axis)
    lex_preds = [1 if supersedes(p["new_text"], p["old_text"]) else 0 for p in pairs]
    out["lexical_baseline"] = _pr_f1(lex_preds, labels)

    # Cosine baseline at the deployed threshold
    if ensure_ready(timeout=120):
        import numpy as np
        cos_scores = []
        for p in pairs:
            va = embed_text(embedding_input({"description": p["new_text"]}, "DECISION"))
            vb = embed_text(embedding_input({"description": p["old_text"]}, "DECISION"))
            cos_scores.append(float(np.dot(va, vb)) if va is not None and vb is not None else 0.0)
        out["cosine"] = {
            "auc": _rank_auc(cos_scores, labels),
            f"at_{COSINE_THRESHOLD}": _pr_f1([1 if s >= COSINE_THRESHOLD else 0 for s in cos_scores], labels),
        }

    # Cross-encoder
    if supersede_xenc.is_available():
        xs = [supersede_xenc.prob_supersedes(p["new_text"], p["old_text"]) or 0.0 for p in pairs]
        jac = [jaccard_similarity(p["new_text"], p["old_text"]) for p in pairs]
        deployed = [1 if (s >= XENC_THRESHOLD and j >= XENC_JACCARD_COFIRE) else 0
                    for s, j in zip(xs, jac)]
        raw_at_t = [1 if s >= XENC_THRESHOLD else 0 for s in xs]
        best = {"f1": -1.0}
        for t in [i / 100 for i in range(50, 100)]:
            m = _pr_f1([1 if s >= t else 0 for s in xs], labels)
            if m["f1"] > best["f1"]:
                best = {**m, "threshold": t}
        # per-kind recall/FP at the deployed operating point
        per_kind: dict = {}
        for kind in out["by_kind"]:
            idx = [i for i, p in enumerate(pairs) if p["kind"] == kind]
            k_lab = [labels[i] for i in idx]
            k_pred = [deployed[i] for i in idx]
            if any(k_lab):
                per_kind[kind] = {"recall": round(sum(p and l for p, l in zip(k_pred, k_lab)) / sum(k_lab), 3), "n": len(idx)}
            else:
                per_kind[kind] = {"fp_rate": round(sum(k_pred) / len(idx), 3) if idx else 0.0, "n": len(idx)}
        out["xenc"] = {
            "available": True,
            "auc": _rank_auc(xs, labels),
            "deployed_hybrid (>=0.97 AND jac>=0.3)": _pr_f1(deployed, labels),
            "raw_at_0.97": _pr_f1(raw_at_t, labels),
            "best_f1_threshold": best,
            "deployed_per_kind": per_kind,
        }
    else:
        out["xenc"] = {"available": False}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    import os
    os.environ.setdefault("MEMLORA_V2_BODY_DIR", str(Path.home() / ".memlora" / "models" / "salience_v2"))

    import memlora.extraction.salience as head_v1
    import memlora.extraction.salience_v2 as head_v2

    sal_items = load_jsonl("salience_eval.jsonl")
    sup_pairs = load_jsonl("supersession_eval.jsonl")

    report = {
        "tag": args.tag,
        "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "salience_items": len(sal_items),
            "salience_by_label": dict(Counter(i["label"] for i in sal_items)),
            "salience_sha": _sha(DATA_DIR / "salience_eval.jsonl"),
            "supersession_pairs": len(sup_pairs),
            "supersession_sha": _sha(DATA_DIR / "supersession_eval.jsonl"),
        },
        "artifacts": artifact_fingerprints(),
        "salience_v1": eval_salience(sal_items, head_v1, "v1"),
        "salience_v2": eval_salience(sal_items, head_v2, "v2"),
        "supersession": eval_supersession(sup_pairs),
    }

    out = DATA_DIR / f"results_{args.tag}_{time.strftime('%Y%m%d')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name in ("salience_v1", "salience_v2"):
        r = report[name]
        if not r.get("available"):
            print(f"{name}: UNAVAILABLE")
            continue
        print(f"\n{name}: acc={r['accuracy']} macro_f1={r['macro_f1']} "
              f"(n={r['n_scored']}, none={r['none_returns']})")
        for lab, m in r["per_label"].items():
            print(f"    {lab:34} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} n={m['support']}")
    sup = report["supersession"]
    print(f"\nsupersession ({sup['n_pairs']} pairs): lexical F1={sup['lexical_baseline']['f1']}")
    if "cosine" in sup:
        print(f"    cosine: AUC={sup['cosine']['auc']} @0.75={sup['cosine'][f'at_{COSINE_THRESHOLD}']}")
    if sup["xenc"].get("available"):
        x = sup["xenc"]
        print(f"    xenc:   AUC={x['auc']} deployed={x['deployed_hybrid (>=0.97 AND jac>=0.3)']} "
              f"best_f1={x['best_f1_threshold']}")
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
