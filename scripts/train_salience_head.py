#!/usr/bin/env python3
"""Train the v1 salience+type head — closed-form ridge over bge-small embeddings.

Reads a JSONL of {text, label} seed examples, embeds each with the SAME model the
runtime uses (memlora.embedding.model.embed_text, L2-normalized 384-d), fits a
linear classifier in closed form, and exports heads/salience_v1.npz.

Closed-form ridge (no SGD → deterministic, no sklearn):
    Xb = [X | 1]                       bias-augmented, (N, 385)
    Y  = one-hot(labels)               (N, C)
    W  = (Xbᵀ Xb + λI)⁻¹ Xbᵀ Y         (385, C)
    predict = argmax(xb @ W)

Usage:
    python scripts/train_salience_head.py [--seeds tests/fixtures/salience_seed.jsonl]
                                          [--lam 1.0] [--holdout 0.2] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SEEDS = _ROOT / "tests" / "fixtures" / "salience_seed.jsonl"
_GENERATED = _ROOT / "tests" / "fixtures" / "salience_train_generated.jsonl"
_TWINS = _ROOT / "tests" / "fixtures" / "salience_twins_generated.jsonl"
_OUT = _ROOT / "src" / "memlora" / "extraction" / "heads" / "salience_v1.npz"

# Must match salience.LABELS order.
LABELS = (
    "NOISE",
    "DECISION",
    "CONSTRAINT_HARD",
    "CONSTRAINT_SOFT",
    "APPROACH_ABANDONED_DO_NOT_RETRY",
    "THREAD",
)
_LABEL_IX = {lbl: i for i, lbl in enumerate(LABELS)}


def load_seeds(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        r = json.loads(line)
        lbl = r["label"]
        if lbl not in _LABEL_IX:
            raise SystemExit(f"unknown label {lbl!r} (allowed: {LABELS})")
        rows.append((r["text"], lbl))
    return rows


def embed_all(texts: list[str]) -> np.ndarray:
    from memlora.embedding.model import embed_text, ensure_ready

    if not ensure_ready(timeout=180):
        raise SystemExit("embedding model not available — cannot train")
    vecs = []
    for t in texts:
        v = embed_text(t)
        if v is None:
            raise SystemExit(f"failed to embed: {t!r}")
        vecs.append(np.asarray(v, dtype="float32"))
    return np.vstack(vecs)


def ridge_fit(X: np.ndarray, y_ix: np.ndarray, n_classes: int, lam: float) -> np.ndarray:
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1), dtype="float32")])  # (n, d+1)
    Y = np.zeros((n, n_classes), dtype="float32")
    Y[np.arange(n), y_ix] = 1.0
    reg = lam * np.eye(d + 1, dtype="float32")
    reg[-1, -1] = 0.0  # do not regularize the bias term
    W = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)  # (d+1, C)
    return W.astype("float32")


def predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype="float32")])
    return np.argmax(Xb @ W, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=None,
                    help="seed JSONL files (default: hand seeds + generated + twins if present)")
    ap.add_argument("--out", default=str(_OUT),
                    help="output .npz path (default: the shipped salience_v1.npz)")
    ap.add_argument("--no-twins", action="store_true",
                    help="exclude the hard-negative twin corpus (for A/B against the legacy recipe)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.seeds:
        paths = [Path(p) for p in args.seeds]
    else:
        paths = [_DEFAULT_SEEDS] + ([_GENERATED] if _GENERATED.exists() else [])
        if not args.no_twins and _TWINS.exists():
            paths.append(_TWINS)
    rows: list[tuple[str, str]] = []
    for p in paths:
        n0 = len(rows)
        rows.extend(load_seeds(p))
        print(f"loaded {len(rows) - n0} from {p.name}")
    texts = [t for t, _ in rows]
    y = np.array([_LABEL_IX[l] for _, l in rows], dtype="int64")
    print(f"seeds: {len(rows)}")
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:<32} {int((y == i).sum())}")

    X = embed_all(texts)

    # Stratified-ish held-out split for an honest generalization read.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(rows))
    n_hold = int(len(rows) * args.holdout)
    hold, train = idx[:n_hold], idx[n_hold:]

    W_tr = ridge_fit(X[train], y[train], len(LABELS), args.lam)
    tr_acc = float((predict(W_tr, X[train]) == y[train]).mean())
    ho_acc = float((predict(W_tr, X[hold]) == y[hold]).mean()) if n_hold else float("nan")
    print(f"\ntrain acc: {tr_acc:.1%}   held-out acc ({n_hold}): {ho_acc:.1%}")

    # Final model: fit on ALL seeds (we want every label in the shipped head).
    W = ridge_fit(X, y, len(LABELS), args.lam)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, labels=np.array(LABELS), W=W)
    try:
        disp = out.relative_to(_ROOT)
    except ValueError:
        disp = out
    print(f"wrote {disp}  (W shape {W.shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
