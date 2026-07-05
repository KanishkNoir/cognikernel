#!/usr/bin/env python3
"""Fine-tune the bge-small backbone for typing via SetFit (WS-B2) — local CPU.

B1 proved a FROZEN backbone + linear head saturates at ~90.5% in-dist and that
hard-negative twins HURT it (84.2%) because the twins are non-separable in the frozen
space. SetFit fixes the geometry: phase 1 contrastively fine-tunes the BODY so the
deontic/polarity twins get pulled apart; phase 2 fits the classifier head. The twins
should now flip from harmful to helpful.

Backbone: BAAI/bge-small-en-v1.5 (the same model the runtime already loads).
Corpus:   seeds + generated + twins ({text, label}, 6 classes).
Eval:     the SAME fixed original-distribution held-out as scripts/_b1_twin_ab.py
          (seed=0, 20% of seeds+generated), so the number is directly comparable to
          B1's frozen 90.5% / +twins 84.2%.

Outputs:
  models/salience_setfit/                    the fine-tuned SetFit model (body + head)
  src/memlora/extraction/heads/salience_v2.npz   the head as a (385, C) W matrix,
          column order == salience.LABELS, for the existing numpy inference path.
          (Runtime body-ONNX export + re-embed wiring is WS-D2.)

Determinism note: training uses SGD (not byte-identical across runs), but the EXPORTED
weights are fixed, so inference stays deterministic — which is what content_hash needs.

Usage:
  python scripts/train_salience_setfit.py [--max-steps 800] [--batch 16]
      [--no-twins]            # ablation: reproduce the B1 comparison under fine-tuning
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
FIX = _ROOT / "tests" / "fixtures"
_BASE = "BAAI/bge-small-en-v1.5"
_MODEL_DIR = _ROOT / "models" / "salience_setfit"
_HEAD_OUT = _ROOT / "src" / "memlora" / "extraction" / "heads" / "salience_v2.npz"

LABELS = ("NOISE", "DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
          "APPROACH_ABANDONED_DO_NOT_RETRY", "THREAD")
IX = {l: i for i, l in enumerate(LABELS)}


def _load(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        r = json.loads(line)
        if r["label"] in IX:
            rows.append((r["text"], IX[r["label"]]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=_BASE)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--no-twins", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-corpus", action="append", default=[], metavar="JSONL",
                    help="additional {text,label} JSONL corpora (e.g. "
                         "research/train_corpus/train_sentences.jsonl — the "
                         "universal ADR+synthetic corpus). Joins the TRAIN split "
                         "only; the fixed held-out stays untouched for "
                         "comparability, and the real gate is the frozen "
                         "research/model_eval suite (never trained on).")
    args = ap.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    orig = _load(FIX / "salience_seed.jsonl") + _load(FIX / "salience_train_generated.jsonl")
    twins = [] if args.no_twins else _load(FIX / "salience_twins_generated.jsonl")
    extra = []
    for p in args.extra_corpus:
        rows = _load(Path(p))
        print(f"extra corpus {p}: {len(rows)} rows")
        extra += rows

    # FIXED original-distribution held-out (identical to _b1_twin_ab.py) for comparability.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(orig))
    n_hold = int(len(orig) * 0.2)
    hold, train = idx[:n_hold].tolist(), idx[n_hold:].tolist()
    train_rows = [orig[i] for i in train] + twins + extra
    hold_rows = [orig[i] for i in hold]
    print(f"orig={len(orig)} twins={len(twins)} extra={len(extra)} | "
          f"train={len(train_rows)} held-out={len(hold_rows)}")

    ds_train = Dataset.from_dict({"text": [t for t, _ in train_rows],
                                  "label": [l for _, l in train_rows]})

    model = SetFitModel.from_pretrained(args.base)
    targs = TrainingArguments(
        batch_size=args.batch,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        sampling_strategy="oversampling",
        seed=args.seed,
        report_to="none",
        show_progress_bar=True,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds_train,
        column_mapping={"text": "text", "label": "label"},
    )
    print(f"SetFit fine-tune: base={args.base} max_steps={args.max_steps} batch={args.batch}")
    trainer.train()

    # Held-out 6-way accuracy (comparable to B1: frozen 90.5% / +twins 84.2%).
    preds = np.asarray(model.predict([t for t, _ in hold_rows]))
    # model.predict may return label strings or ints depending on head; normalize to int.
    if preds.dtype.kind in ("U", "S", "O"):
        preds = np.asarray([IX.get(str(p), -1) for p in preds])
    yh = np.asarray([l for _, l in hold_rows])
    acc = float((preds == yh).mean())
    print(f"\nheld-out 6-way acc: {acc:.1%}   (B1 frozen 90.5% / +twins 84.2%)")

    def conf(a, b):
        m = yh == IX[a]
        return (float((preds[m] == IX[b]).mean()) if m.any() else float("nan")), int(m.sum())
    print("directional confusions (lower better):")
    for a, b in [("CONSTRAINT_HARD", "CONSTRAINT_SOFT"), ("CONSTRAINT_SOFT", "CONSTRAINT_HARD"),
                 ("DECISION", "APPROACH_ABANDONED_DO_NOT_RETRY")]:
        r, n = conf(a, b)
        print(f"  {a:>32} -> {b:<32} {r:.0%} (n={n})")

    # Save the SetFit model.
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(_MODEL_DIR))
    print(f"saved SetFit model to {_MODEL_DIR.relative_to(_ROOT)}")

    # Export the head as a (385, C) W matrix for the existing numpy inference path,
    # when the head is the default sklearn LogisticRegression.
    try:
        head = model.model_head
        coef = np.asarray(head.coef_, dtype="float32")          # (C, 384)
        intercept = np.asarray(head.intercept_, dtype="float32")  # (C,)
        classes = list(getattr(head, "classes_", range(len(LABELS))))
        # reorder rows so columns of W match LABELS order
        order = [list(classes).index(i) for i in range(len(LABELS))]
        W = np.vstack([coef[order].T, intercept[order][None, :]]).astype("float32")  # (385, C)
        _HEAD_OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez(_HEAD_OUT, labels=np.array(LABELS), W=W)
        print(f"exported head -> {_HEAD_OUT.relative_to(_ROOT)}  (W {W.shape})")
    except Exception as exc:  # noqa: BLE001
        print(f"head export skipped ({type(exc).__name__}: {exc}); body+head saved in model dir")
    print("\nNote: runtime body-ONNX export + re-embed wiring is WS-D2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
