"""v2 salience+type head over the FINE-TUNED bge-small body (WS-D2 runtime path).

B2 (SetFit) fine-tuned the backbone so the deontic/polarity twins separate; the head
was fit on that fine-tuned body's embeddings. This module runs that body at inference
WITHOUT torch — onnxruntime over the exported ONNX (byte-parity validated by
scripts/export_setfit_onnx.py) + the `tokenizers` fast tokenizer (already a fastembed
dependency) + the numpy linear head in heads/salience_v2.npz. Deterministic forward
pass → stable content_hash, consistent with the v1 path's contract.

Artifacts:
  heads/salience_v2.npz                       labels + (385, C) W   (shipped, small)
  <body dir>/body.onnx + tokenizer.json       the fine-tuned encoder (large; located via
      MEMLORA_V2_BODY_DIR, else ~/.memlora/models/salience_v2, else repo models/ dir)

GRACEFUL FALLBACK: any missing artifact -> classify_scored returns None and the caller
keeps the v1 / legacy path. The head never blocks extraction.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

LABELS: tuple[str, ...] = (
    "NOISE", "DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
    "APPROACH_ABANDONED_DO_NOT_RETRY", "THREAD",
)

_HEAD_PATH = Path(__file__).resolve().parent / "heads" / "salience_v2.npz"
_MAX_TOKENS = 512

# R2 — confidence floor. A non-NOISE prediction must clear this softmax probability,
# else the head is too unsure and the candidate is dropped as NOISE. Primary lever
# against v2-broad over-capture (Relay: 379 active for ~18 facts). Coarse de-noiser,
# not a calibrated probability — temperature calibration is the follow-on. Env-tunable
# via MEMLORA_SIFT_FLOOR; 0.0 disables it.
_FLOOR = float(os.environ.get("MEMLORA_SIFT_FLOOR", "0.5"))

_lock = threading.Lock()
_loaded = False
_sess = None
_tok = None
_W = None
_labels: tuple[str, ...] | None = None


def _body_dir() -> Path:
    env = os.environ.get("MEMLORA_V2_BODY_DIR")
    if env:
        return Path(env)
    home = Path(os.environ.get("MEMLORA_DIR") or (Path.home() / ".memlora")) / "models" / "salience_v2"
    if (home / "body.onnx").exists():
        return home
    # dev fallback: the export script's default output location in the repo
    return Path(__file__).resolve().parents[3] / "models" / "salience_setfit" / "onnx"


def _load() -> bool:
    global _loaded, _sess, _tok, _W, _labels
    with _lock:
        if _loaded:
            return _sess is not None and _W is not None
        _loaded = True
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer

            bd = _body_dir()
            onnx_path, tok_path = bd / "body.onnx", bd / "tokenizer.json"
            if not (_HEAD_PATH.exists() and onnx_path.exists() and tok_path.exists()):
                return False
            data = np.load(_HEAD_PATH, allow_pickle=False)
            _labels = tuple(str(x) for x in data["labels"])
            _W = data["W"].astype("float32")
            _sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            _tok = Tokenizer.from_file(str(tok_path))
            return True
        except Exception:
            _sess = None
            return False


def is_available() -> bool:
    """True if the v2 head + ONNX body + tokenizer all load."""
    return _load()


def _embed(text: str):
    import numpy as np

    enc = _tok.encode(text)
    ids = enc.ids[:_MAX_TOKENS]
    input_ids = np.asarray([ids], dtype="int64")
    attn = np.ones_like(input_ids)
    tti = np.zeros_like(input_ids)
    out = _sess.run(["embedding"], {
        "input_ids": input_ids, "attention_mask": attn, "token_type_ids": tti,
    })[0]
    return out[0].astype("float32")  # (384,), already L2-normalized in-graph


def classify_scored(text: str) -> Optional[tuple[str, float]]:
    """Return (label, confidence) or None if the v2 artifacts are unavailable.

    None => caller falls back (never an implicit drop). ("NOISE", p) is an explicit drop.
    """
    if not text or not text.strip():
        return ("NOISE", 1.0)
    if not _load():
        return None

    import numpy as np

    v = _embed(text)
    xb = np.concatenate([v, np.ones(1, dtype="float32")])
    logits = xb @ _W
    z = logits - logits.max()
    p = np.exp(z)
    p = p / p.sum()
    top = int(np.argmax(p))
    label = (_labels or LABELS)[top]
    conf = float(p[top])
    # R2 floor: an unsure non-NOISE prediction is dropped as NOISE rather than minted
    # as a durable typed fact. ("NOISE", conf) is an explicit drop the caller honors.
    if label != "NOISE" and conf < _FLOOR:
        return ("NOISE", conf)
    return (label, conf)


def classify(text: str) -> Optional[str]:
    scored = classify_scored(text)
    return None if scored is None else scored[0]
