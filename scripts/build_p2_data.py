"""Assemble the P2 (role+context) training corpus and eval — PRE-COMPOSED.

Every training/eval item's text is passed through compose_head_input(text, role,
prev) at build time, so the retrain and the eval both see the composed string and
the format is anchored to that ONE function (later inference calls it too). This
is why the train/eval scripts need no P2 flag — the composition lives in the data.

Inputs:
  fixtures (seed/generated/twins)        -> compose([assistant], no prev)  [bare-ish]
  train_sentences_boost.jsonl (register) -> compose([role_from_register], no prev)
  context_examples.jsonl (role, prev)    -> compose([role], prev)   [the P2 signal]

Outputs (research/train_corpus + research/model_eval):
  train_sentences_p2.jsonl   full composed training set (use --no-fixtures)
  salience_eval_p2.jsonl     the FROZEN eval, composed (role_from_register, no prev)
                             — SAME items as boost800's eval, so macro-F1 is
                             directly comparable; measures the role signal + no regression
  salience_eval_p2ctx.jsonl  a held-out slice of context_examples, composed with
                             real prev — the mechanism test (does the head USE prev?)

Usage: uv run python scripts/build_p2_data.py [--ctx-holdout 0.15]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")
random.seed(1106)

from cognikernel.extraction.head_input import compose_head_input, role_for_register

CORPUS = Path("research/train_corpus")
EVAL = Path("research/model_eval")
FIX = Path("tests/fixtures")
LABELS = {"NOISE", "DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
          "APPROACH_ABANDONED_DO_NOT_RETRY", "THREAD"}


def _load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "//")):
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-holdout", type=float, default=0.15)
    args = ap.parse_args()

    # ── training set ──────────────────────────────────────────────────────────
    train: list[dict] = []

    # fixtures: no register/role -> assistant default, no prev
    for name in ("salience_seed.jsonl", "salience_train_generated.jsonl", "salience_twins_generated.jsonl"):
        for r in _load(FIX / name):
            if r.get("label") in LABELS and r.get("text"):
                train.append({"text": compose_head_input(r["text"], "assistant", ""), "label": r["label"]})

    # boost corpus: register -> role, no prev
    for r in _load(CORPUS / "train_sentences_boost.jsonl"):
        if r.get("label") in LABELS and r.get("text"):
            role = role_for_register(r.get("register", ""))
            train.append({"text": compose_head_input(r["text"], role, ""), "label": r["label"]})

    # context examples: real role + prev — split into train + a held-out ctx eval
    ctx = [r for r in _load(CORPUS / "context_examples.jsonl") if r.get("label") in LABELS]
    random.shuffle(ctx)
    n_hold = int(len(ctx) * args.ctx_holdout)
    ctx_eval, ctx_train = ctx[:n_hold], ctx[n_hold:]
    for r in ctx_train:
        train.append({"text": compose_head_input(r["text"], r.get("role", "assistant"), r.get("prev", "")),
                      "label": r["label"]})

    (CORPUS / "train_sentences_p2.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in train) + "\n", encoding="utf-8")

    # ── eval 1: the FROZEN eval, composed (role_from_register, no prev) ─────────
    frozen = _load(EVAL / "salience_eval.jsonl")
    eval_main = [{"text": compose_head_input(r["text"], role_for_register(r.get("register", "")), ""),
                  "label": r["label"], "register": r.get("register", "")} for r in frozen]
    (EVAL / "salience_eval_p2.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in eval_main) + "\n", encoding="utf-8")

    # ── eval 2: context held-out, composed WITH prev (the mechanism test) ───────
    eval_ctx = [{"text": compose_head_input(r["text"], r.get("role", "assistant"), r.get("prev", "")),
                 "label": r["label"], "register": r.get("register", "")} for r in ctx_eval]
    (EVAL / "salience_eval_p2ctx.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in eval_ctx) + "\n", encoding="utf-8")

    from collections import Counter
    print(f"train_sentences_p2.jsonl: {len(train)}  ({dict(Counter(t['label'] for t in train))})")
    print(f"salience_eval_p2.jsonl:   {len(eval_main)} (frozen eval, composed — the comparison)")
    print(f"salience_eval_p2ctx.jsonl:{len(eval_ctx)} (context holdout — the mechanism test)")
    print("\nsample composed training item:")
    print(" ", next(t['text'] for t in train if t['text'].startswith('[')  and '||' in t['text'])[:160])


if __name__ == "__main__":
    main()
