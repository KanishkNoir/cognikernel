"""Spike B — zero-shot LFM2.5-230M salience classification via onnxruntime-genai.

Runs a sub-500M local LLM (LFM2.5-230M, int4 CPU) through onnxruntime-genai —
pip-installable, torch-free, NO external service (not Ollama), rides the
onnxruntime dep the package already ships. Zero session tokens (off-hot-path in
the Stop hook/worker, not the agent). Classifies the FROZEN eval
(research/model_eval/salience_eval.jsonl) zero-shot and reports the SAME metrics
as scripts/model_eval.py, so it's directly comparable to the encoder
(current best: acc 0.715, macro-F1 0.474). The decisive slice is memory_meta,
where the encoder is stuck ~0.56 (the meta-framing test).

Model dir = the ort-genai builder output (has genai_config.json):
  python -m onnxruntime_genai.models.builder -m LiquidAI/LFM2.5-230M \
      -o models/lfm2.5-230m-genai -p int4 -e cpu

Usage: uv run --with onnxruntime-genai python scripts/spike_lfm_classify.py \
           --model-dir models/lfm2.5-230m-genai [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path("research/model_eval/salience_eval.jsonl")
LABELS = ("NOISE", "DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
          "APPROACH_ABANDONED_DO_NOT_RETRY", "THREAD")

_SYSTEM = (
    "You classify ONE sentence from an AI-coding-assistant session into exactly one label.\n"
    "- DECISION: a concrete choice made for the project (tool, value, design).\n"
    "- CONSTRAINT_HARD: an inviolable rule/invariant (must/never/always, security/correctness).\n"
    "- CONSTRAINT_SOFT: a convention or preference (naming, style, layout).\n"
    "- APPROACH_ABANDONED_DO_NOT_RETRY: an approach explicitly rejected/ruled out.\n"
    "- THREAD: open/ongoing work — in progress or planned next.\n"
    "- NOISE: everything else — narration, questions, explanations, acknowledgments, and\n"
    "  META references that merely QUOTE or mention a prior decision/memory rather than\n"
    "  state a new project fact (e.g. 'the recall surfaces the earlier decision to use X').\n"
    'Answer with ONLY the label word, nothing else.'
)

_ALIAS = {
    "ABANDONED": "APPROACH_ABANDONED_DO_NOT_RETRY", "APPROACH_ABANDONED": "APPROACH_ABANDONED_DO_NOT_RETRY",
    "DO_NOT_RETRY": "APPROACH_ABANDONED_DO_NOT_RETRY", "APPROACH": "APPROACH_ABANDONED_DO_NOT_RETRY",
    "THREAD_OPEN": "THREAD", "THREAD_CLOSE": "THREAD",
    "HARD": "CONSTRAINT_HARD", "SOFT": "CONSTRAINT_SOFT", "CONSTRAINT": "CONSTRAINT_HARD",
}


import re as _re
_ACTION_RE = _re.compile(r"<action>\s*(.*?)\s*</action>", _re.I | _re.S)


def _norm(raw: str, cot: bool = False) -> str:
    # CoT models emit "<thought>…</thought><action>LABEL</action>" — parse the
    # ACTION tag, never scan the whole string (the <thought> text mentions other
    # label words and a naive scan collapses everything to the first match).
    if cot:
        m = _ACTION_RE.search(raw)
        seg = m.group(1) if m else raw.split("</thought>")[-1]
        up = seg.strip().upper()
    else:
        up = raw.strip().upper()
    for lab in LABELS:
        if lab in up:
            return lab
    tok = up.split()[0].replace(",", "").replace(".", "") if up.split() else ""
    tok = _ALIAS.get(tok, tok)
    return tok if tok in LABELS else "NOISE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/lfm2.5-230m-genai")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cot", action="store_true",
                    help="model was CoT-instruction-tuned: prompt with the bare "
                         "'Sentence: X' (training format, no system) and parse "
                         "<action>LABEL</action> from the thought/action output.")
    args = ap.parse_args()

    import onnxruntime_genai as og
    model = og.Model(args.model_dir)
    tok = og.Tokenizer(model)

    def classify(text: str) -> str:
        if args.cot:
            # Exactly the shape cot_sft.jsonl trained on: user-only, no system.
            prompt = (f"<|startoftext|><|im_start|>user\nSentence: {text}<|im_end|>\n"
                      f"<|im_start|>assistant\n")
            extra = 80  # room for <thought>… + <action>LABEL</action>
        else:
            prompt = (f"<|startoftext|><|im_start|>system\n{_SYSTEM}<|im_end|>\n"
                      f"<|im_start|>user\nSentence: {text}<|im_end|>\n<|im_start|>assistant\n")
            extra = 24
        ids = tok.encode(prompt)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(ids) + extra, do_sample=False)
        gen = og.Generator(model, params)
        gen.append_tokens(ids)
        while not gen.is_done():
            gen.generate_next_token()
        out = gen.get_sequence(0)[len(ids):]
        return _norm(tok.decode(out), cot=args.cot)

    items = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        items = items[: args.limit]

    y_true, y_pred = [], []
    per_register = defaultdict(lambda: [0, 0])
    confusion: Counter = Counter()
    t0 = time.time()
    for i, it in enumerate(items):
        pred = classify(it["text"])
        y_true.append(it["label"]); y_pred.append(pred)
        confusion[(it["label"], pred)] += 1
        reg = per_register[it["register"]]; reg[1] += 1
        if pred == it["label"]:
            reg[0] += 1
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(items)}  ({(time.time()-t0)/(i+1):.2f}s/item)", flush=True)

    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n if n else 0.0
    f1s = []
    print(f"\nLFM2.5-230M zero-shot (onnxruntime-genai, int4 CPU) on {n} eval items:")
    for lab in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if tp + fn:
            f1s.append(f1)
        print(f"    {lab:34} P={prec:.2f} R={rec:.2f} F1={f1:.2f} n={tp+fn}")
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    sig_t = sum(1 for t in y_true if t != "NOISE") or 1
    sig_cap = sum(1 for t, p in zip(y_true, y_pred) if t != "NOISE" and p != "NOISE")
    noise_t = sum(1 for t in y_true if t == "NOISE") or 1
    noise_cap = sum(1 for t, p in zip(y_true, y_pred) if t == "NOISE" and p != "NOISE")
    mm = per_register["memory_meta"]
    print(f"\n  acc={acc:.3f}  macro_f1={macro:.3f}   (encoder best: acc 0.715 / macro_f1 0.474)")
    print(f"  deployment: capture_recall={sig_cap/sig_t:.3f} false_capture={noise_cap/noise_t:.3f}")
    print(f"  memory_meta acc: {mm[0]}/{mm[1]} = {mm[0]/max(1,mm[1]):.3f}  (encoder stuck ~0.56)")
    print(f"  avg latency: {(time.time()-t0)/max(1,n):.2f}s/item  (~{n/(time.time()-t0):.1f} sent/s)")
    print("  top confusions:", [f"{a}->{b}:{c}" for (a, b), c in confusion.most_common(8) if a != b][:6])

    out = Path("research/model_eval") / "results_lfm2.5-230m.json"
    out.write_text(json.dumps({
        "model": "LFM2.5-230M int4 ort-genai", "n": n, "acc": round(acc, 3),
        "macro_f1": round(macro, 3), "capture_recall": round(sig_cap / sig_t, 3),
        "false_capture": round(noise_cap / noise_t, 3),
        "per_register": {r: {"acc": round(c / t, 3), "n": t} for r, (c, t) in sorted(per_register.items())},
        "sec_per_item": round((time.time() - t0) / max(1, n), 3),
    }, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
