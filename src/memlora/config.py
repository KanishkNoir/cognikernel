from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_SCHEMA_VERSION: int = 4
EXPECTED_PROJECTION_VERSION: int = 1


@dataclass
class Config:
    memlora_dir: Path = field(default_factory=lambda: Path.home() / ".memlora")
    token_budget: int = 2000
    skeleton_budget: int = 800
    wal_warning_threshold_bytes: int = 100 * 1024 * 1024
    grep_cache_enabled: bool = False

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

        return cls(**kwargs)
