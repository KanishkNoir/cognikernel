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
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

_MAX_TOKENS = 256
_BATCH_SIZE = 16
_DEFAULT_THRESHOLD = 0.9  # conservative fallback if threshold.json is absent

_lock = threading.Lock()
_loaded = False
_sess = None
_tok = None
_threshold = _DEFAULT_THRESHOLD
_deployed_min: Optional[float] = None
_inference_error: str | None = None
_inference_error_lock = threading.Lock()
_log = logging.getLogger("cognikernel.delta.supersede_xenc")
_diagnostics_dir: Path | None = None
_diagnostics_dir_lock = threading.Lock()


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
            # The exported graph is parity-safe for equal-length, unpadded pairs.
            _tok.no_padding()
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


def set_diagnostics_dir(path: Path) -> None:
    """Use the project's CogniKernel directory for persistent inference diagnostics."""
    global _diagnostics_dir
    with _diagnostics_dir_lock:
        _diagnostics_dir = path


def _diagnostic_path(diagnostics_dir: Path | None = None) -> Path:
    if diagnostics_dir is not None:
        root = diagnostics_dir
    else:
        with _diagnostics_dir_lock:
            root = _diagnostics_dir
        if root is None:
            root = Path(
                os.environ.get("COGNIKERNEL_DIR") or (Path.home() / ".cognikernel")
            )
    return root / "diagnostics" / "supersession_xenc.json"


def last_inference_error(diagnostics_dir: Path | None = None) -> str | None:
    """Latest persisted batch failure for the active model, if any."""
    try:
        record = json.loads(_diagnostic_path(diagnostics_dir).read_text(encoding="utf-8"))
        if Path(record["body_dir"]).resolve() == _body_dir().resolve():
            error = record.get("error")
            return error if isinstance(error, str) else None
    except FileNotFoundError:
        pass
    except Exception as exc:
        _log.warning("supersession_xenc.diagnostic_read_failed: %s", exc, exc_info=True)

    # Retain an in-process copy for diagnostics when the persistent directory is
    # unavailable, but never let that mask an on-disk record in another process.
    with _inference_error_lock:
        return _inference_error


def _write_inference_diagnostic(error: str | None) -> None:
    path = _diagnostic_path()
    temp_path: Path | None = None
    try:
        if error is None:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump({
                "body_dir": str(_body_dir().resolve()),
                "error": error,
            }, temp_file)
        os.replace(temp_path, path)
    except Exception as exc:
        _log.warning(
            "supersession_xenc.diagnostic_write_failed: %s",
            exc,
            exc_info=True,
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _record_inference_error(batch_size: int, exc: Exception) -> None:
    global _inference_error
    detail = f"batch_size={batch_size}: {exc}"
    with _inference_error_lock:
        _inference_error = detail
    _write_inference_diagnostic(detail)
    _log.warning(
        "supersession_xenc.inference_failed: %s",
        detail,
        exc_info=True,
    )


def _clear_inference_error() -> None:
    global _inference_error
    with _inference_error_lock:
        _inference_error = None
    _write_inference_diagnostic(None)


def prob_supersedes_batch(
    pairs: list[tuple[str, str]],
) -> list[Optional[float]]:
    """Score pairs together when token lengths match (the ONNX graph's safe batch shape).

    The exported graph has known padded-batch parity issues, so batches contain
    only unpadded sequences of equal length. A failed batch degrades to None for
    that batch, matching the single-pair fail-open behavior.
    """
    results: list[Optional[float]] = [None] * len(pairs)
    active = [i for i, (new, old) in enumerate(pairs) if new and old]
    if not active or not _load():
        return results

    import numpy as np

    groups: dict[int, list[tuple[int, list[int], list[int]]]] = {}
    try:
        encoded = _tok.encode_batch([pairs[i] for i in active])
        for i, enc in zip(active, encoded):
            ids = enc.ids[:_MAX_TOKENS]
            type_ids = enc.type_ids[:_MAX_TOKENS] if enc.type_ids else [0] * len(ids)
            groups.setdefault(len(ids), []).append((i, ids, type_ids))
    except Exception as exc:
        _record_inference_error(len(active), exc)
        return results

    failed = False
    scored = False
    for group in groups.values():
        for start in range(0, len(group), _BATCH_SIZE):
            batch = group[start:start + _BATCH_SIZE]
            try:
                input_ids = np.asarray([item[1] for item in batch], dtype="int64")
                token_type = np.asarray([item[2] for item in batch], dtype="int64")
                attn = np.ones_like(input_ids)
                logits = _sess.run(["logits"], {
                    "input_ids": input_ids,
                    "attention_mask": attn,
                    "token_type_ids": token_type,
                })[0].reshape(len(batch), -1)
                if logits.shape[1] == 1:
                    scores = 1.0 / (1.0 + np.exp(-logits[:, 0]))
                else:
                    shifted = logits - logits.max(axis=1, keepdims=True)
                    probs = np.exp(shifted)
                    probs /= probs.sum(axis=1, keepdims=True)
                    scores = probs[:, 1]
                for item, score in zip(batch, scores):
                    results[item[0]] = float(score)
                scored = True
            except Exception as exc:
                failed = True
                _record_inference_error(len(batch), exc)
                continue
    if scored and not failed:
        _clear_inference_error()
    return results


def prob_supersedes(new_desc: str, old_desc: str) -> Optional[float]:
    """P(new_desc supersedes old_desc) in [0,1], or None if the model is unavailable."""
    return prob_supersedes_batch([(new_desc, old_desc)])[0]


def supersedes(new_desc: str, old_desc: str) -> bool:
    """Bool predicate at the calibrated threshold (used by eval_supersession.py --predicate)."""
    p = prob_supersedes(new_desc, old_desc)
    return p is not None and p >= threshold()
