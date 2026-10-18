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
      COGNIKERNEL_V2_BODY_DIR, else ~/.cognikernel/models/salience_v2, else repo models/ dir)

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
_BATCH_SIZE = 32

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
    env = os.environ.get("COGNIKERNEL_V2_BODY_DIR")
    if env:
        return Path(env)
    home = Path(os.environ.get("COGNIKERNEL_DIR") or (Path.home() / ".cognikernel")) / "models" / "salience_v2"
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
            # The shipped tokenizer serializes padding; disable it so encode_batch
            # produces the same unpadded inputs as the established scalar path.
            _tok.no_padding()
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


def _embed_batch(texts: list[str]):
    import numpy as np

    encodings = _tok.encode_batch(texts)
    groups: dict[int, list[tuple[int, list[int]]]] = {}
    for index, enc in enumerate(encodings):
        ids = enc.ids[:_MAX_TOKENS]
        groups.setdefault(len(ids), []).append((index, ids))

    # The exported graph's outputs can shift on mixed-length padded batches.
    embeddings = [None] * len(texts)
    for group in groups.values():
        for start in range(0, len(group), _BATCH_SIZE):
            batch = group[start:start + _BATCH_SIZE]
            input_ids = np.asarray([item[1] for item in batch], dtype="int64")
            attn = np.ones_like(input_ids)
            tti = np.zeros_like(input_ids)
            out = _sess.run(["embedding"], {
                "input_ids": input_ids, "attention_mask": attn, "token_type_ids": tti,
            })[0]
            for item, vector in zip(batch, out):
                embeddings[item[0]] = vector.astype("float32")
    return np.asarray(embeddings, dtype="float32")


def classify_scored_batch(texts: list[str]) -> list[Optional[tuple[str, float]]]:
    """Classify texts in bounded ONNX batches; unavailable artifacts yield None per item.

    None => caller falls back (never an implicit drop). ("NOISE", p) is an explicit drop.
    """
    results: list[Optional[tuple[str, float]]] = [
        ("NOISE", 1.0) if not text or not text.strip() else None for text in texts
    ]
    active = [i for i, text in enumerate(texts) if text and text.strip()]
    if not active:
        return results
    if not _load():
        return results

    import numpy as np

    vectors = _embed_batch([texts[i] for i in active])
    xb = np.concatenate(
        [vectors, np.ones((len(vectors), 1), dtype="float32")], axis=1
    )
    logits = xb @ _W
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    for index, probs in zip(active, probabilities):
        top = int(np.argmax(probs))
        # LOSSLESS: low-confidence facts are kept and down-sampled at render.
        results[index] = ((_labels or LABELS)[top], float(probs[top]))
    return results


def classify_scored(text: str) -> Optional[tuple[str, float]]:
    """Return (label, confidence) or None if the v2 artifacts are unavailable."""
    return classify_scored_batch([text])[0]


def classify(text: str) -> Optional[str]:
    scored = classify_scored(text)
    return None if scored is None else scored[0]
