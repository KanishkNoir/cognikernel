"""Meta-guard for the import-linter architecture contracts (audit P2 / #64).

The whole audit P2 root cause was that an import-linter contract named a module
that did not exist (`cognikernel.observability`). import-linter aborts the ENTIRE run
with "Module '...' does not exist." the instant any contract references a missing
module — so that one typo silently disabled enforcement of every contract, and
real layering drift accumulated underneath with zero signal.

There is no CI in this repo, so the guardrail lives where the guard runs: the
test suite. These tests assert (1) every module named by every contract is
importable — so a typo can never silently zero enforcement again — and (2) when
import-linter is installed, the contracts actually evaluate and all pass.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _contract_modules() -> set[str]:
    """Every cognikernel module named by any import-linter contract."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    importlinter = data.get("tool", {}).get("importlinter", {})
    modules: set[str] = set(importlinter.get("root_packages", []))
    for contract in importlinter.get("contracts", []):
        for key in ("layers", "source_modules", "forbidden_modules"):
            modules.update(contract.get(key, []))
    return modules


def test_every_contract_module_exists() -> None:
    """A contract that names a nonexistent module aborts the whole linter run and
    silently disables ALL enforcement — the exact bug behind audit P2. Fail loudly
    here instead."""
    missing = []
    for module in sorted(_contract_modules()):
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            missing.append(module)
    assert not missing, (
        "import-linter contracts reference nonexistent modules "
        f"{missing} — this aborts the entire lint run and disables ALL "
        "architecture enforcement. Fix the name or remove the contract."
    )


def test_contracts_are_declared() -> None:
    """Guard against an empty/typo'd config evaluating zero contracts."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    contracts = data.get("tool", {}).get("importlinter", {}).get("contracts", [])
    assert len(contracts) >= 3, "expected the layered + forbidden-upstream contracts"


def test_import_linter_passes_when_installed() -> None:
    """When import-linter is available, the contracts must actually evaluate and
    all pass — i.e. the run is not aborting and the layering holds."""
    if shutil.which("lint-imports") is None:
        pytest.skip("import-linter not installed in this environment")
    result = subprocess.run(
        ["lint-imports"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = result.stdout + result.stderr
    # The run must have evaluated contracts (not aborted on a missing module) ...
    assert "does not exist" not in out, f"import-linter aborted: {out}"
    assert "Contracts:" in out, f"import-linter did not evaluate contracts: {out}"
    # ... and all of them must pass.
    assert result.returncode == 0, f"import-linter found broken contracts:\n{out}"
    assert "broken" not in out.split("Contracts:")[-1] or "0 broken" in out
