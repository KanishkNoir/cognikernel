#!/usr/bin/env python3
"""Export the supersession cross-encoder to ONNX with parity (R5 Phase 3).

The runtime must score sentence PAIRS torch-free (onnxruntime + tokenizers), per the
local/deterministic moat. This exports the fine-tuned cross-encoder (a 2-class
BertForSequenceClassification) as a graph: (input_ids, attention_mask, token_type_ids)
-> logits, and VALIDATES it against the torch model before saving.

Input:  models/supersession_xenc/            (the trained cross-encoder, HF format)
Output: models/supersession_xenc/onnx/body.onnx + tokenizer.json
Deploy: copy to ~/.memlora/models/supersession_xenc/ (or set MEMLORA_XENC_BODY_DIR);
        threshold.json (next to the model) carries the calibrated decision threshold.

Usage:  python scripts/export_xenc_onnx.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "supersession_xenc"
OUT_DIR = MODEL_DIR / "onnx"
# (newer, older) pairs — a supersedes b, and a same-area complementary non-pair.
SAMPLES = [
    ("Switch the password hashing algorithm from bcrypt to argon2id",
     "We hash passwords with bcrypt"),
    ("Use a composite primary key for the events table",
     "Use a UUID primary key for the users table"),
]


def _prob(logits):
    import numpy as np
    a = np.asarray(logits, dtype="float32").reshape(-1)
    if a.size == 1:
        return float(1.0 / (1.0 + np.exp(-a[0])))
    z = a - a.max(); p = np.exp(z) / np.exp(z).sum()
    return float(p[1])


def main() -> int:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits

    wrap = Wrap(model)
    enc = tok([a for a, _ in SAMPLES], [b for _, b in SAMPLES],
              padding=True, truncation=True, max_length=256, return_tensors="pt")
    inputs = (enc["input_ids"], enc["attention_mask"],
              enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / "body.onnx"
    torch.onnx.export(
        wrap, inputs, str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in
                      ("input_ids", "attention_mask", "token_type_ids")} | {"logits": {0: "batch"}},
        opset_version=14, dynamo=False,
    )
    tok.save_pretrained(str(OUT_DIR))
    print(f"exported {onnx_path.relative_to(ROOT)}")

    # parity: onnxruntime vs torch
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {k: v.cpu().numpy() for k, v in zip(
        ("input_ids", "attention_mask", "token_type_ids"), inputs)}
    onnx_logits = sess.run(["logits"], feeds)[0]
    with torch.no_grad():
        torch_logits = wrap(*inputs).numpy()
    max_abs = float(np.abs(onnx_logits - torch_logits).max())
    print(f"\nPARITY: max_abs_diff={max_abs:.2e}  (want < 1e-3)")
    for (a, b), ol, tl in zip(SAMPLES, onnx_logits, torch_logits):
        print(f"  onnx P(sup)={_prob(ol):.3f}  torch={_prob(tl):.3f}  :: {a[:40]!r} vs {b[:30]!r}")
    print("PASS" if max_abs < 1e-3 else "FAIL")
    return 0 if max_abs < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
