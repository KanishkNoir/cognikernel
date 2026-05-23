from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_SCHEMA_VERSION: int = 4
EXPECTED_PROJECTION_VERSION: int = 1


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
    token_budget: int = 2000
    skeleton_budget: int = 800
    wal_warning_threshold_bytes: int = 100 * 1024 * 1024
    grep_cache_enabled: bool = False
    ckl_mode: bool = False
    ckl_v2: bool = False
    section_budgets: SectionBudgets = field(default_factory=SectionBudgets)

    @property
    def projects_dir(self) -> Path:
        return self.memlora_dir / "projects"

    @property
    def logs_dir(self) -> Path:
        return self.memlora_dir / "logs"

    @classmethod
    def load(cls, config_path: Path | None = None) -> Config:
        # MEMLORA_DIR env var lets tests (and CI) redirect the data directory
        # without touching ~/.memlora.
        env_dir = os.environ.get("MEMLORA_DIR")
        if env_dir:
            return cls(memlora_dir=Path(env_dir))

        if config_path is None:
            config_path = Path.home() / ".memlora" / "config.toml"

        if not config_path.exists():
            return cls()

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
        if "section_budgets" in data and isinstance(data["section_budgets"], dict):
            sb = data["section_budgets"]
            kwargs["section_budgets"] = SectionBudgets(
                **{k: int(v) for k, v in sb.items() if k in SectionBudgets.__dataclass_fields__}
            )

        return cls(**kwargs)
