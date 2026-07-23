"""Learned salience + type head over the existing bge-small embedding (v1 B).

Replaces the keyword `classify_event` scoring and the regex noise filters with a
single linear classifier on the embedding the recall path already loads. One
model, two heads: pooled vector → embeddings DB, class logits → events DB.

Design choices (applied-research notes):
  - LINEAR head over FROZEN, L2-normalized embeddings — a cosine-prototype
    classifier. Few labels suffice; the embedding does the heavy lifting.
  - RIDGE closed-form fit (W = (XᵀX + λI)⁻¹ XᵀY), not SGD. Deterministic — same
    seeds → byte-identical weights → stable content_hash and cache prefix.
  - INFERENCE is a numpy matmul (no torch, no sklearn at runtime). argmax over
    6 classes; NOISE is a first-class label that the pipeline drops.
  - GRACEFUL FALLBACK: if the embedding model isn't resident or the head file is
    missing, classify() returns None and the caller keeps the legacy path. The
    head never blocks extraction.

The head file is `heads/salience_v1.npz` with arrays:
    labels : (C,)  unicode      class names, index = column of W
    W      : (385, C) float32   bias-augmented weights (row 384 is the bias)
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

# Class vocabulary. Order is the canonical label index used by the trainer.
LABELS: tuple[str, ...] = (
    "NOISE",
    "DECISION",
    "CONSTRAINT_HARD",
    "CONSTRAINT_SOFT",
    "APPROACH_ABANDONED_DO_NOT_RETRY",
    "THREAD",
)

_HEAD_PATH = Path(__file__).resolve().parent / "heads" / "salience_v1.npz"

# Confidence floor: if the top class doesn't beat the runner-up by this softmax
# margin, fall back to NOISE. Keeps low-confidence facts out of memory. Tuned on
# the seed held-out split, never on the Relay S1 gold.
_MARGIN_MIN = 0.0

_lock = threading.Lock()
_loaded = False
_labels: tuple[str, ...] | None = None
_W = None  # numpy array (385, C)


def _load_head() -> bool:
    global _loaded, _labels, _W
    with _lock:
        if _loaded:
            return _W is not None
        _loaded = True
        try:
            import numpy as np

            if not _HEAD_PATH.exists():
                return False
            data = np.load(_HEAD_PATH, allow_pickle=False)
            _labels = tuple(str(x) for x in data["labels"])
            _W = data["W"].astype("float32")
            return True
        except Exception:
            _W = None
            return False


def is_available() -> bool:
    """True if the head file loads AND the embedding model can be made resident.

    Extraction runs in the Stop hook (off the interactive hot path), so this
    BLOCKS to load the model if needed (ensure_ready, not the non-blocking
    is_ready). Under the test/CI auto-warm guard ensure_ready still loads on an
    explicit request, but embed_text returns None there, so the head path
    self-disables and the caller falls back to legacy.
    """
    from cognikernel.embedding.model import ensure_ready

    return _load_head() and ensure_ready(timeout=180)


def expects_context() -> bool:
    """v1 (frozen head) is never trained on role+context-composed input."""
    return False


def classify_scored(text: str) -> Optional[tuple[str, float]]:
    """Return (label, confidence) or None if the head/model is unavailable.

    None means "fall back to the legacy classifier" — never an implicit drop.
    A returned ("NOISE", p) is an explicit drop signal the caller must honor.
    confidence is the softmax probability of the chosen class.
    """
    if not text or not text.strip():
        return ("NOISE", 1.0)
    if not _load_head():
        return None

    from cognikernel.embedding.model import embed_text

    v = embed_text(text)
    if v is None:
        return None

    import numpy as np

    xb = np.concatenate([np.asarray(v, dtype="float32"), np.ones(1, dtype="float32")])
    logits = xb @ _W  # (C,)
    z = logits - logits.max()
    p = np.exp(z)
    p = p / p.sum()
    order = np.argsort(p)[::-1]
    top, second = int(order[0]), int(order[1])
    if (p[top] - p[second]) < _MARGIN_MIN:
        return ("NOISE", float(p[top]))
    return ((_labels or LABELS)[top], float(p[top]))


def classify(text: str) -> Optional[str]:
    """Return a label in LABELS, or None if the head/model is unavailable."""
    scored = classify_scored(text)
    return None if scored is None else scored[0]
