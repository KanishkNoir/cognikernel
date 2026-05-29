from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_SCHEMA_VERSION: int = 12
EXPECTED_PROJECTION_VERSION: int = 1

VALID_HOOK_POLICIES = frozenset({"advisory", "strict"})

# Single authoritative ceiling for the rendered injection block, set from the
# Unit 7a baseline measurement (see research/beta/promotion_criteria.md). This
# is the one number the render path enforces: greedy selection, the section
# budgets, and the global backstop all derive from config.token_budget, which
# defaults to this.
DEFAULT_TOKEN_BUDGET: int = 1500


@dataclass
class SectionBudgets:
    """Per-section token caps for the injection block.

    Budgets are tuned to the U-shaped LLM recall curve:
      - Primacy zone sections (hard_constraints, active_thread) get small budgets:
        position alone already gives ~85% recall, so headroom is wasted there.
      - Recency zone sections (skeleton, summary) get the largest budget:
        these are the most query-critical and benefit most from high recall.
      - Middle (decay zone) sections get medium budgets and are first to drop
        events when over budget.

    Default sum: ~1470 tok (well under the 2000-tok global budget — leaves
    headroom for the header and section separators).
    """
    hard_constraints: int = 150
    active_thread: int = 80
    hot_files: int = 50
    graveyard: int = 120
    components: int = 80
    decisions: int = 150
    skeleton: int = 800
    summary: int = 40


@dataclass
class Config:
    memlora_dir: Path = field(default_factory=lambda: Path.home() / ".memlora")
    token_budget: int = DEFAULT_TOKEN_BUDGET
    # Sub-cap for the AST skeleton section. Shrunk 800 → 600 (Unit 6): with
    # PageRank-ranked entries the most-central files survive the smaller budget,
    # so this concentrates value rather than dropping content blindly. Non-skeleton
    # sections measure ~600 tok, so 600 + 600 leaves headroom under the 1500 ceiling.
    skeleton_budget: int = 600
    wal_warning_threshold_bytes: int = 100 * 1024 * 1024
    grep_cache_enabled: bool = False
    ckl_mode: bool = False
    ckl_v2: bool = False
    hook_policy: str = "advisory"  # "advisory" (legacy) | "strict" (deny-by-default)
    read_cache_ttl_hours: int = 24
    deny_retry_window_seconds: int = 60
    # When True, session_end stores a local embedding per event and supersession
    # uses the hybrid (semantic + temporal + authority) finder. Default off:
    # opt-in, gradual rollout, A/B-able. Degrades to lexical if the model is absent.
    embedding_enabled: bool = False
    section_budgets: SectionBudgets = field(default_factory=SectionBudgets)

    @property
    def projects_dir(self) -> Path:
        return self.memlora_dir / "projects"

    @property
    def logs_dir(self) -> Path:
        return self.memlora_dir / "logs"

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        *,
        project_path: str | Path | None = None,
    ) -> Config:
        """Load config from disk.

        Precedence (highest first):
          1. `<project_path>/.memlora/config.toml`  (when project_path is given)
          2. `~/.memlora/config.toml`  (the global file, or `config_path` if provided)
          3. Built-in defaults

        Each layer is read independently and merged via dataclasses.replace so
        per-project overrides only need to specify the keys that differ.

        The MEMLORA_DIR env var short-circuits everything for test/CI use.
        """
        # MEMLORA_DIR env var lets tests (and CI) redirect the data directory
        # without touching ~/.memlora. It overrides memlora_dir specifically,
        # but project-local overlays still apply on top of it.
        env_dir = os.environ.get("MEMLORA_DIR")

        if config_path is None:
            config_path = Path.home() / ".memlora" / "config.toml"

        # Layer 2 — global config.
        if env_dir:
            base = cls(memlora_dir=Path(env_dir))
        elif config_path.exists():
            base = cls._load_from_file(config_path)
        else:
            base = cls()

        # Layer 1 — project-local overlay (if any).
        if project_path is not None:
            project_cfg_path = Path(project_path) / ".memlora" / "config.toml"
            if project_cfg_path.exists():
                project_kwargs = cls._read_toml_kwargs(project_cfg_path)
                # Replace only the fields the project file specifies.
                # memlora_dir override from MEMLORA_DIR is preserved unless the
                # project config explicitly sets a different memlora_dir.
                from dataclasses import replace
                base = replace(base, **project_kwargs)

        return base

    @classmethod
    def _load_from_file(cls, config_path: Path) -> Config:
        kwargs = cls._read_toml_kwargs(config_path)
        return cls(**kwargs)

    @staticmethod
    def _read_toml_kwargs(config_path: Path) -> dict:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        kwargs: dict = {}
        if "memlora_dir" in data:
            kwargs["memlora_dir"] = Path(data["memlora_dir"])
        if "token_budget" in data:
            kwargs["token_budget"] = int(data["token_budget"])
        if "skeleton_budget" in data:
            kwargs["skeleton_budget"] = int(data["skeleton_budget"])
        if "wal_warning_threshold_mb" in data:
            kwargs["wal_warning_threshold_bytes"] = int(data["wal_warning_threshold_mb"]) * 1024 * 1024
        if "grep_cache_enabled" in data:
            kwargs["grep_cache_enabled"] = bool(data["grep_cache_enabled"])
        if "ckl_mode" in data:
            kwargs["ckl_mode"] = bool(data["ckl_mode"])
        if "ckl_v2" in data:
            kwargs["ckl_v2"] = bool(data["ckl_v2"])
        if "hook_policy" in data:
            policy = str(data["hook_policy"])
            if policy not in VALID_HOOK_POLICIES:
                raise ValueError(
                    f"invalid hook_policy {policy!r}; expected one of {sorted(VALID_HOOK_POLICIES)}"
                )
            kwargs["hook_policy"] = policy
        if "read_cache_ttl_hours" in data:
            kwargs["read_cache_ttl_hours"] = int(data["read_cache_ttl_hours"])
        if "deny_retry_window_seconds" in data:
            kwargs["deny_retry_window_seconds"] = int(data["deny_retry_window_seconds"])
        if "embedding_enabled" in data:
            kwargs["embedding_enabled"] = bool(data["embedding_enabled"])
        if "section_budgets" in data and isinstance(data["section_budgets"], dict):
            sb = data["section_budgets"]
            kwargs["section_budgets"] = SectionBudgets(
                **{k: int(v) for k, v in sb.items() if k in SectionBudgets.__dataclass_fields__}
            )

        return kwargs
