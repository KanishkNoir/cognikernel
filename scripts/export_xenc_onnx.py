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
    # QUARANTINE: export to a temp name and promote only on parity PASS, so a
    # failed export can never be picked up by install-heads (the rollback lesson —
    # a FAIL that leaves body.onnx in place gets deployed by the next chained step).
    tmp_path = OUT_DIR / "body.onnx.tmp"
    torch.onnx.export(
        wrap, inputs, str(tmp_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in
                      ("input_ids", "attention_mask", "token_type_ids")} | {"logits": {0: "batch"}},
        opset_version=14, dynamo=False,
    )
    tok.save_pretrained(str(OUT_DIR))

    import onnxruntime as ort
    sess = ort.InferenceSession(str(tmp_path), providers=["CPUExecutionProvider"])

    # GATE — batch-1 parity, the runtime contract: supersede_xenc.prob_supersedes
    # scores one unpadded pair at a time, so this is what production executes.
    worst_b1 = 0.0
    for a, b in SAMPLES:
        e1 = tok([a], [b], padding=True, truncation=True, max_length=256, return_tensors="pt")
        t1 = (e1["input_ids"], e1["attention_mask"],
              e1.get("token_type_ids", torch.zeros_like(e1["input_ids"])))
        o = sess.run(["logits"], {k: v.cpu().numpy() for k, v in zip(
            ("input_ids", "attention_mask", "token_type_ids"), t1)})[0]
        with torch.no_grad():
            t = wrap(*t1).numpy()
        worst_b1 = max(worst_b1, float(np.abs(o - t).max()))
        print(f"  batch-1 parity {float(np.abs(o - t).max()):.2e}  P(sup) onnx={_prob(o[0]):.3f} "
              f"torch={_prob(t[0]):.3f} :: {a[:40]!r}")

    # REPORT — padded-batch parity. The legacy exporter mishandles attention over
    # padding on some torch versions (measured 0.56 divergence); nothing in the
    # runtime batches, so this is a warning, not the gate. Do NOT batch pairs
    # through this graph without fixing the export path first.
    feeds = {k: v.cpu().numpy() for k, v in zip(
        ("input_ids", "attention_mask", "token_type_ids"), inputs)}
    onnx_logits = sess.run(["logits"], feeds)[0]
    with torch.no_grad():
        torch_logits = wrap(*inputs).numpy()
    max_padded = float(np.abs(onnx_logits - torch_logits).max())
    print(f"\nPARITY gate (batch-1, runtime contract): max_abs={worst_b1:.2e}  (want < 1e-3)")
    print(f"PARITY info (padded batch)             : max_abs={max_padded:.2e}"
          + ("  — WARNING: do not batch through this graph" if max_padded >= 1e-3 else ""))

    if worst_b1 < 1e-3:
        tmp_path.replace(onnx_path)
        print(f"PASS — promoted {onnx_path.relative_to(ROOT)}")
        return 0
    tmp_path.unlink(missing_ok=True)
    print("FAIL — quarantined (no body.onnx written)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
