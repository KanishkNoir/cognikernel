"""CK-6b — CLI output is encoding-safe on non-UTF-8 consoles (Windows cp1252)."""
from __future__ import annotations

import io
import sys
from unittest.mock import patch

from memlora.integration.cli import _ensure_utf8_output


def test_ensure_utf8_output_prevents_cp1252_crash() -> None:
    """A '→' (U+2192, in the skeleton) printed to a cp1252 stdout would raise
    UnicodeEncodeError; after _ensure_utf8_output it encodes as UTF-8 instead."""
    out_buf, err_buf = io.BytesIO(), io.BytesIO()
    out = io.TextIOWrapper(out_buf, encoding="cp1252", newline="")
    err = io.TextIOWrapper(err_buf, encoding="cp1252", newline="")
    with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
        _ensure_utf8_output()
        print("models.py → schemas.py")  # cp1252 cannot encode '→'
        sys.stdout.flush()
    assert "→".encode("utf-8") in out_buf.getvalue()


def test_ensure_utf8_output_is_safe_on_non_reconfigurable_stream() -> None:
    """A stream without reconfigure() (e.g. a plain StringIO) must not raise."""
    with patch.object(sys, "stdout", io.StringIO()), patch.object(sys, "stderr", io.StringIO()):
        _ensure_utf8_output()  # no AttributeError
