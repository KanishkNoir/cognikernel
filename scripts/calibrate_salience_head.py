"""P1 — temperature-calibrate the salience head (fold T into W, zero runtime change).

The head's confidence IS the event weight downstream (broad mode sets
weight = conf), and it is measurably overconfident: errors average ~0.78
confidence vs ~0.88 on correct. Temperature scaling is the standard fix:
softmax(logits / T) with one scalar T fit by NLL on a held-out slice.

Because logits = [emb, 1] @ W, dividing logits by T is exactly W' = W / T —
so the calibrated head ships as the same .npz and classify_scored is untouched.
Argmax is unchanged (accuracy identical); only the confidence becomes honest.

Fit data: the TRAINING held-out (the same seed-0 20% split the trainer prints),
never the frozen eval — that stays a pure yardstick. Reported on both.

Usage: uv run --extra embedding python scripts/calibrate_salience_head.py
           [--corpus research/train_corpus/train_sentences_p2.jsonl] [--apply]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

HEAD = Path("src/cognikernel/extraction/heads/salience_v2.npz")
LABELS = ("NOISE", "DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
          "APPROACH_ABANDONED_DO_NOT_RETRY", "THREAD")
IX = {l: i for i, l in enumerate(LABELS)}


def _load_rows(path: Path) -> list[tuple[str, int]]:
    """Replicates the trainer's _load exactly (order matters for the split)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        r = json.loads(line)
        if r.get("label") in IX:
            rows.append((r["text"], IX[r["label"]]))
    return rows


def _ece(probs, labels, bins: int = 10) -> float:
    import numpy as np
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype("float64")
    e = 0.0
    for lo in [i / bins for i in range(bins)]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.sum():
            e += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="research/train_corpus/train_sentences_p2.jsonl")
    ap.add_argument("--apply", action="store_true",
                    help="write the calibrated W back to the .npz (default: report only)")
    args = ap.parse_args()

    import numpy as np
    import cognikernel.extraction.salience_v2 as h

    if not h.is_available():
        sys.exit("v2 head/body not available")
    # Reach the raw W + embedder for logit computation.
    from cognikernel.extraction import salience_v2 as sv

    rows = _load_rows(Path(args.corpus))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    n_hold = int(len(rows) * 0.2)
    hold = [rows[i] for i in idx[:n_hold].tolist()]
    print(f"corpus={len(rows)}  calibration holdout={len(hold)} (trainer's seed-0 split)")

    # Embed holdout, compute raw logits.
    X, y = [], []
    for i, (text, lab) in enumerate(hold):
        v = sv._embed(text)
        X.append(np.concatenate([v, np.ones(1, dtype="float32")]))
        y.append(lab)
        if (i + 1) % 400 == 0:
            print(f"  embedded {i+1}/{len(hold)}", flush=True)
    X = np.stack(X)
    y = np.asarray(y)
    logits = X @ sv._W

    def stats(T: float):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=1, keepdims=True)
        nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
        return nll, _ece(p, y), p

    # Grid-search T by NLL.
    best_T, best_nll = 1.0, stats(1.0)[0]
    for T in np.arange(0.5, 5.01, 0.05):
        nll = stats(float(T))[0]
        if nll < best_nll:
            best_T, best_nll = float(T), nll

    nll0, ece0, p0 = stats(1.0)
    nll1, ece1, p1 = stats(best_T)
    pred = p0.argmax(axis=1)  # argmax identical at any T
    acc = float((pred == y).mean())
    conf_ok0 = float(p0.max(axis=1)[pred == y].mean())
    conf_err0 = float(p0.max(axis=1)[pred != y].mean())
    conf_ok1 = float(p1.max(axis=1)[pred == y].mean())
    conf_err1 = float(p1.max(axis=1)[pred != y].mean())

    print(f"\nfitted temperature T = {best_T:.2f}   (holdout acc {acc:.3f}, unchanged by T)")
    print(f"  NLL : {nll0:.4f} -> {nll1:.4f}")
    print(f"  ECE : {ece0:.4f} -> {ece1:.4f}")
    print(f"  mean conf on correct: {conf_ok0:.3f} -> {conf_ok1:.3f}")
    print(f"  mean conf on errors : {conf_err0:.3f} -> {conf_err1:.3f}")

    if args.apply:
        d = np.load(HEAD, allow_pickle=False)
        extras = {k: d[k] for k in d.files if k != "W"}
        np.savez(HEAD, W=(d["W"] / best_T).astype("float32"), **extras)
        print(f"\napplied: W/T written to {HEAD} (labels/context_input preserved)")
        print("re-deploy is not needed for the head (.npz is read from the package); "
              "the ONNX body is unchanged.")
    else:
        print("\n(report only — rerun with --apply to fold T into the head)")


if __name__ == "__main__":
    main()
