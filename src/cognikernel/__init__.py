"""CogniKernel — structured session-state memory layer for AI coding assistants."""

# Kept in sync with [project].version in pyproject.toml. Hardcoded rather than
# read via importlib.metadata so importing the package — which happens on every
# `python -m cognikernel hook-*` spawn — never pays a dist-info filesystem lookup.
__version__ = "0.1.0"

