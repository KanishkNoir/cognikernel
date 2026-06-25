#!/usr/bin/env python3
"""Export the fine-tuned SetFit body to ONNX with exact parity to encode() (WS-D2).

The salience_v2 head was fit on the FINE-TUNED body's embeddings, so the runtime must
reproduce that body exactly — and stay torch-free (numpy/onnxruntime), per the local
moat. This exports a graph that does: BERT -> CLS pooling -> L2 normalize (the bge /
SetFit encode() pipeline) and VALIDATES it against model_body.encode() before saving.

Outputs (to models/salience_setfit/onnx/):
  body.onnx            the fine-tuned encoder (input_ids, attention_mask, token_type_ids)
  tokenizer files      saved alongside for the runtime tokenizer (no torch needed)

Usage:  python scripts/export_setfit_onnx.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "salience_setfit"
OUT_DIR = MODEL_DIR / "onnx"
SAMPLES = [
    "We decided to use Postgres for the primary datastore.",
    "Secrets must never be written to logs.",
    "Still need to wire up the retry policy.",
    "What if we cached the responses for an hour?",
]


def main() -> int:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(str(MODEL_DIR))
    st = model.model_body                      # SentenceTransformer
    transformer = st[0]                        # models.Transformer
    hf = transformer.auto_model               # HF BertModel
    tok = transformer.tokenizer

    # Confirm the pooling really is CLS (bge default) — the head depends on it.
    pool_cfg = st[1].get_config_dict() if len(st) > 1 else {}
    cls_pool = (pool_cfg.get("pooling_mode") == "cls") or bool(pool_cfg.get("pooling_mode_cls_token"))
    print(f"pooling config: {pool_cfg}  -> CLS={cls_pool}")
    if not cls_pool:
        print("WARNING: body is not CLS-pooled; export wrapper assumes CLS.", file=sys.stderr)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
            cls = out.last_hidden_state[:, 0]      # CLS pooling
            return F.normalize(cls, p=2, dim=1)    # L2 normalize

    wrap = Wrap(hf).eval()
    enc = tok(SAMPLES, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = (enc["input_ids"], enc["attention_mask"],
              enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / "body.onnx"
    torch.onnx.export(
        wrap, inputs, str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["embedding"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask", "token_type_ids")}
        | {"embedding": {0: "batch"}},
        opset_version=14,
        dynamo=False,   # legacy TorchScript exporter — avoids the onnxscript dependency
    )
    tok.save_pretrained(str(OUT_DIR))
    print(f"exported {onnx_path.relative_to(ROOT)}")

    # ── parity validation: onnxruntime vs SetFit encode() ──
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {k: v.cpu().numpy() for k, v in zip(
        ("input_ids", "attention_mask", "token_type_ids"), inputs)}
    onnx_emb = sess.run(["embedding"], feeds)[0]
    ref_emb = st.encode(SAMPLES, normalize_embeddings=True)
    ref_emb = np.asarray(ref_emb, dtype="float32")

    max_abs = float(np.abs(onnx_emb - ref_emb).max())
    cos = float((onnx_emb * ref_emb).sum(axis=1).mean())
    print(f"\nPARITY: max_abs_diff={max_abs:.2e}  mean_cosine={cos:.6f}  (want diff<1e-3, cos~1.0)")
    print("PASS" if max_abs < 1e-3 else "FAIL — investigate pooling/normalize mismatch")
    return 0 if max_abs < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
