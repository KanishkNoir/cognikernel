"""install-heads: local-copy path, release-download fallback, checksum gate."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest

from cognikernel.integration import cli
from cognikernel.integration.cli import (
    _cmd_install_heads,
    _download_head_asset,
    _HEADS_RELEASE_ASSETS,
)


def _args(**over) -> argparse.Namespace:
    base = {"source": None, "force": False, "no_download": False}
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch):
    """No repo models/ dir, COGNIKERNEL_DIR under tmp — the adopter-install shape."""
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "cognikernel"))
    monkeypatch.setattr(cli, "_repo_models_dir", lambda: tmp_path / "no-such-models")
    return tmp_path


class TestDownloadFallback:
    def test_missing_local_source_downloads_all_assets(self, isolated, monkeypatch):
        fetched: list[str] = []

        def fake_download(url: str, target: Path, sha256: str) -> None:
            fetched.append(url.rsplit("/", 1)[1])
            target.write_bytes(b"artifact")

        monkeypatch.setattr(cli, "_download_head_asset", fake_download)
        _cmd_install_heads(_args())

        models = isolated / "cognikernel" / "models"
        assert (models / "salience_v2" / "body.onnx").exists()
        assert (models / "salience_v2" / "tokenizer.json").exists()
        assert (models / "supersession_xenc" / "body.onnx").exists()
        assert (models / "supersession_xenc" / "threshold.json").exists()
        expected = [a for head in _HEADS_RELEASE_ASSETS.values() for a, _, _ in head]
        assert sorted(fetched) == sorted(expected)

    def test_existing_files_skipped_without_force(self, isolated, monkeypatch):
        dest = isolated / "cognikernel" / "models" / "salience_v2"
        dest.mkdir(parents=True)
        (dest / "body.onnx").write_bytes(b"already-here")

        def fake_download(url: str, target: Path, sha256: str) -> None:
            target.write_bytes(b"fresh")

        monkeypatch.setattr(cli, "_download_head_asset", fake_download)
        _cmd_install_heads(_args())
        assert (dest / "body.onnx").read_bytes() == b"already-here"

    def test_all_downloads_failing_exits_nonzero(self, isolated, monkeypatch):
        def fail(url: str, target: Path, sha256: str) -> None:
            raise RuntimeError("offline")

        monkeypatch.setattr(cli, "_download_head_asset", fail)
        with pytest.raises(SystemExit) as exc:
            _cmd_install_heads(_args())
        assert exc.value.code == 1

    def test_no_download_flag_never_fetches(self, isolated, monkeypatch):
        def fail(url: str, target: Path, sha256: str) -> None:  # pragma: no cover
            raise AssertionError("must not be called")

        monkeypatch.setattr(cli, "_download_head_asset", fail)
        with pytest.raises(SystemExit) as exc:
            _cmd_install_heads(_args(no_download=True))
        assert exc.value.code == 1


class TestLocalSource:
    def test_explicit_source_copies_salience_head(self, isolated):
        src = isolated / "export"
        src.mkdir()
        (src / "body.onnx").write_bytes(b"onnx-bytes")
        (src / "tokenizer.json").write_text("{}", encoding="utf-8")

        _cmd_install_heads(_args(source=str(src)))
        dest = isolated / "cognikernel" / "models" / "salience_v2"
        assert (dest / "body.onnx").read_bytes() == b"onnx-bytes"


class TestDownloadHeadAsset:
    def test_verifies_checksum_and_installs(self, tmp_path: Path):
        payload = b"model-bytes"
        src = tmp_path / "asset.bin"
        src.write_bytes(payload)
        target = tmp_path / "out" / "body.onnx"
        target.parent.mkdir()

        _download_head_asset(src.as_uri(), target, hashlib.sha256(payload).hexdigest())
        assert target.read_bytes() == payload
        assert not target.with_suffix(target.suffix + ".part").exists()

    def test_checksum_mismatch_raises_and_leaves_no_file(self, tmp_path: Path):
        src = tmp_path / "asset.bin"
        src.write_bytes(b"tampered")
        target = tmp_path / "out" / "body.onnx"
        target.parent.mkdir()

        with pytest.raises(RuntimeError, match="checksum mismatch"):
            _download_head_asset(src.as_uri(), target, "0" * 64)
        assert not target.exists()
        assert not target.with_suffix(target.suffix + ".part").exists()
