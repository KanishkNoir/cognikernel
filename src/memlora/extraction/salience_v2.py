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

# R2 (LOSSLESS reframe): the head does NOT drop low-confidence predictions. Confidence
# rides along as the event weight (set in pipeline._extract_via_head), so the RENDER path
# down-samples uncertain facts out of the budget-ranked block while the events store keeps
# them — lossless capture, down-sample at read, never delete at write. Anything dropped is
# always re-derivable from raw_evidence.

_lock = threading.Lock()
_loaded = False
_sess = None
_tok = None
_W = None
_labels: tuple[str, ...] | None = None
_context_input: bool = False


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
    global _loaded, _sess, _tok, _W, _labels, _context_input
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
            # P2: the head declares whether it was trained on role+context-composed
            # input. The pipeline composes iff this is set, so a bare (v2) head and a
            # context (P2) head can never be fed the wrong format (the CoT lesson).
            _context_input = bool(data["context_input"][0]) if "context_input" in data else False
            _sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            _tok = Tokenizer.from_file(str(tok_path))
            return True
        except Exception:
            _sess = None
            return False


def is_available() -> bool:
    """True if the v2 head + ONNX body + tokenizer all load."""
    return _load()


def expects_context() -> bool:
    """True if the loaded head was trained on role+prev-context-composed input
    (P2). The pipeline gates compose_head_input on this so format never drifts."""
    _load()
    return _context_input


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
    # LOSSLESS: return the head's actual call (no confidence DROP here). Low-confidence
    # facts are kept and DOWN-SAMPLED at render via weight (= conf, set by the pipeline),
    # never deleted at write — the events store stays lossless and re-tunable.
    return ((_labels or LABELS)[top], float(p[top]))


def classify(text: str) -> Optional[str]:
    scored = classify_scored(text)
    return None if scored is None else scored[0]
