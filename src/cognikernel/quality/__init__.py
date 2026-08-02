"""Statement-quality detection and admission control.

This package is a LEAF: it imports only the stdlib and cognikernel.model.
Never import storage, injection, compression, integration, or delta here —
an import-linter contract enforces it (pyproject.toml).
"""
from cognikernel.quality.detectors import (
    BOX_DRAWING_RE,
    DetectorHit,
    detect_boilerplate,
    detect_junk_constraint,
    detect_subject_less,
    normalized_key,
)
from cognikernel.quality.gate import (
    GroundingContext,
    Verdict,
    admit,
    apply_verdict,
)

__all__ = [
    "BOX_DRAWING_RE",
    "DetectorHit",
    "GroundingContext",
    "Verdict",
    "admit",
    "apply_verdict",
    "detect_boilerplate",
    "detect_junk_constraint",
    "detect_subject_less",
    "normalized_key",
]
