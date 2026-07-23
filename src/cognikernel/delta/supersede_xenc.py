"""Torch-free cross-encoder supersession scorer (R5 Phase 4 — runtime).

A bi-encoder cosine cannot separate "this corrects that" from "this is a different
same-area decision" (the F5 finding). The cross-encoder reads both descriptions jointly
and scores P(supersede). This module runs it WITHOUT torch — onnxruntime over the
exported ONNX (parity-validated by scripts/export_xenc_onnx.py) + the `tokenizers` fast
tokenizer (already a fastembed dependency) — a deterministic forward pass off the hot
path (the Stop-hook merge).

Artifacts (located via COGNIKERNEL_XENC_BODY_DIR, else ~/.cognikernel/models/supersession_xenc,
else the repo dev path):
  body.onnx + tokenizer.json   the fine-tuned cross-encoder
  threshold.json               {"threshold": p}  the precision-calibrated decision point

GRACEFUL FALLBACK: any missing artifact -> prob_supersedes returns None and
find_superseded degrades to the gated lexical+cosine baseline. The cross-encoder only
ever ADDS supersessions the structured gates already permit, at this high threshold, so
it is precision-safe by construction. It never deletes — supersession marks superseded_by.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

_MAX_TOKENS = 256
_DEFAULT_THRESHOLD = 0.9  # conservative fallback if threshold.json is absent

_lock = threading.Lock()
_loaded = False
_sess = None
_tok = None
_threshold = _DEFAULT_THRESHOLD
_deployed_min: Optional[float] = None


def _body_dir() -> Path:
    env = os.environ.get("COGNIKERNEL_XENC_BODY_DIR")
    if env:
        return Path(env)
    home = Path(os.environ.get("COGNIKERNEL_DIR") or (Path.home() / ".cognikernel")) / "models" / "supersession_xenc"
    if (home / "body.onnx").exists():
        return home
    return Path(__file__).resolve().parents[3] / "models" / "supersession_xenc" / "onnx"


def _load() -> bool:
    global _loaded, _sess, _tok, _threshold, _deployed_min
    with _lock:
        if _loaded:
            return _sess is not None
        _loaded = True
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            bd = _body_dir()
            onnx_path, tok_path = bd / "body.onnx", bd / "tokenizer.json"
            if not (onnx_path.exists() and tok_path.exists()):
                return False
            _sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            _tok = Tokenizer.from_file(str(tok_path))
            for cand in (bd / "threshold.json", bd.parent / "threshold.json"):
                if cand.exists():
                    data = json.loads(cand.read_text())
                    _threshold = float(data["threshold"])
                    if "deployed_min" in data:
                        _deployed_min = float(data["deployed_min"])
                    break
            return True
        except Exception:
            _sess = None
            return False


def is_available() -> bool:
    return _load()


def threshold() -> float:
    _load()
    return _threshold


def deployed_min() -> Optional[float]:
    """The model's re-validated DEPLOYED operating point, if it ships one.

    threshold.json's `threshold` is the VAL-calibrated point (target-precision on
    the training distribution) — it historically does NOT transfer to real stores
    (baseline: val says 0.136; the real-store bleed-stop is 0.97). So the merge
    gate only trusts an explicit `deployed_min` field, which a model earns by
    re-validation on the frozen real-store eval. Absent -> caller falls back to
    the hardcoded, store-validated constant.
    """
    _load()
    return _deployed_min


def prob_supersedes(new_desc: str, old_desc: str) -> Optional[float]:
    """P(new_desc supersedes old_desc) in [0,1], or None if the model is unavailable."""
    if not new_desc or not old_desc or not _load():
        return None
    try:
        import numpy as np

        enc = _tok.encode(new_desc, old_desc)
        ids = enc.ids[:_MAX_TOKENS]
        tti = (enc.type_ids[:_MAX_TOKENS] if enc.type_ids else [0] * len(ids))
        input_ids = np.asarray([ids], dtype="int64")
        attn = np.ones_like(input_ids)
        token_type = np.asarray([tti], dtype="int64")
        logits = _sess.run(["logits"], {
            "input_ids": input_ids, "attention_mask": attn, "token_type_ids": token_type,
        })[0].reshape(-1)
        if logits.size == 1:
            return float(1.0 / (1.0 + np.exp(-logits[0])))
        z = logits - logits.max()
        p = np.exp(z) / np.exp(z).sum()
        return float(p[1])
    except Exception:
        return None


def supersedes(new_desc: str, old_desc: str) -> bool:
    """Bool predicate at the calibrated threshold (used by eval_supersession.py --predicate)."""
    p = prob_supersedes(new_desc, old_desc)
    return p is not None and p >= threshold()
