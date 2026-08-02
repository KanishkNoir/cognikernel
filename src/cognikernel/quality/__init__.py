"""Statement-quality detection and admission control.

This package is a LEAF: it imports only the stdlib and cognikernel.model.
Never import storage, injection, compression, integration, or delta here —
an import-linter contract enforces it (pyproject.toml).
"""
from cognikernel.quality.detectors import DetectorHit, detect_subject_less

__all__ = ["DetectorHit", "detect_subject_less"]
