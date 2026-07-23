"""Regression tests for the embedding model loader.

These pin the two properties behind the "recall hung for 4 minutes" fix:
  - an interactive caller never blocks past its timeout on a slow/cold load, and
  - `warm()` / `embed_text()` never kick a background download under the auto-warm
    guard that conftest sets for the whole test session.

The real model is never loaded here — `_load` is monkeypatched to a fast sentinel.
"""
from __future__ import annotations

import os
import time

import pytest

from cognikernel.embedding import model as M


@pytest.fixture(autouse=True)
def _isolate_model_state():
    """Save/clear/restore the module-level loader singleton around each test."""
    with M._lock:
        saved = (M._thread, M._model_obj, M._load_done)
        M._thread = None
        M._model_obj = None
        M._load_done = False
    try:
        yield
    finally:
        with M._lock:
            t = M._thread
        if t is not None:
            t.join(5)
        with M._lock:
            M._thread, M._model_obj, M._load_done = saved


def test_ensure_ready_times_out_without_blocking(monkeypatch):
    """A slow load must not make ensure_ready(timeout) hang past the budget."""
    def _slow_load():
        time.sleep(0.5)
        return object()  # non-None sentinel "model"

    monkeypatch.setattr(M, "_load", _slow_load)

    t0 = time.monotonic()
    ready = M.ensure_ready(timeout=0.05)
    elapsed = time.monotonic() - t0

    assert ready is False          # not ready within 50ms
    assert elapsed < 0.4           # and we did NOT block for the full 0.5s load
    assert M.is_ready() is False   # the non-blocking check agrees

    # The background load still finishes; a patient caller eventually succeeds.
    assert M.ensure_ready(timeout=2.0) is True
    assert M.is_ready() is True


def test_warm_is_suppressed_by_auto_warm_guard(monkeypatch):
    """conftest sets COGNIKERNEL_DISABLE_AUTO_WARM=1 — warm() must not start a load."""
    assert os.environ.get("COGNIKERNEL_DISABLE_AUTO_WARM") == "1"

    called = {"load": False}

    def _tracking_load():
        called["load"] = True
        return object()

    monkeypatch.setattr(M, "_load", _tracking_load)

    M.warm()
    with M._lock:
        assert M._thread is None   # no loader thread spawned
    assert called["load"] is False
    assert M.is_ready() is False

    # ensure_ready is the EXPLICIT path and bypasses the guard — it does load.
    assert M.ensure_ready(timeout=2.0) is True
    assert called["load"] is True


def test_embed_text_does_not_download_under_guard(monkeypatch):
    """Under the guard, a cold embed_text returns None without starting a load."""
    called = {"load": False}

    def _tracking_load():
        called["load"] = True
        return object()

    monkeypatch.setattr(M, "_load", _tracking_load)

    assert M.embed_text("some text") is None
    with M._lock:
        assert M._thread is None
    assert called["load"] is False
