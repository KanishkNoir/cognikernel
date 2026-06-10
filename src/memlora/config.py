from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_SCHEMA_VERSION: int = 14
EXPECTED_PROJECTION_VERSION: int = 1

VALID_HOOK_POLICIES = frozenset({"advisory", "strict"})

# Extraction backend selector. `legacy` is the deterministic keyword/Aho-Corasick
# pipeline (Stage 2). The `v1*` modes use the frozen-backbone learned salience head
# (extraction/salience.py); the `v2*` modes use the SetFit fine-tuned head served
# torch-free via ONNX (extraction/salience_v2.py). Plain modes (`v1`/`v2`) filter +
# re-type the legacy candidate set; `-broad` modes classify every prose sentence.
# All head paths fail open: a missing head/model falls back to legacy, so selecting
# an encoder mode never breaks extraction.
VALID_EXTRACTORS = frozenset({"legacy", "v1", "v1-broad", "v2", "v2-broad"})

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
    # When True, a UserPromptSubmit hook injects a short memory snippet alongside
    # each user prompt — only when a high-confidence, non-redundant hit exists.
    # Default OFF (sprint-plan flag). Register `memlora hook-user-prompt` in
    # settings.json to enable. Ships behind the kill-criterion (inject on <~30% of
    # prompts, per-turn tokens under budget, quality not reduced — else stays as a
    # pull-only recall MCP tool). See integration/hooks.py:user_prompt_submit_main.
    query_time_injection: bool = False
    # Cosine relevance bar for per-prompt injection. Higher = more selective (fewer
    # injections). Start conservative; tune based on measured injection rate.
    query_injection_threshold: float = 0.75
    # Hard token ceiling for the per-turn snippet (excluding overhead).
    query_injection_max_tokens: int = 200
    # When True, SubagentStop fires the extraction pipeline on the subagent's
    # transcript and merges decisions into the parent project DB. Default ON once
    # SubagentStop is wired in settings.json (register hook-subagent-stop).
    capture_subagents: bool = True

    # Controls whether the *semantic axis fires for auto-supersession* (the
    # precision-risky path). Default OFF: real-data validation showed cosine alone
    # cannot separate a genuine correction from an unrelated same-project decision
    # (TP/FP cosine gap 0.004). Subject-keying (supersede.py) now provides the
    # structural discriminator, but the threshold is still unvalidated on a full
    # representative set, so the auto-supersession semantic axis stays opt-in.
    #
    # NOTE: embedding *storage* is now DECOUPLED from this flag — vectors are
    # always written when fastembed is installed (merge._store_event_embedding is
    # always called), so recall / find_related are semantic regardless of this
    # flag. This flag is now specifically about "use embeddings to find supersession
    # candidates" — not about whether to store them or use them for recall.
    embedding_enabled: bool = False

    # R5 — use the learned cross-encoder as an additive supersession candidate axis
    # (above the always-on temporal/authority/provenance gates). Default off and
    # fail-open: with the flag off or the ONNX model absent, supersession degrades to
    # the gated lexical (+optional cosine) baseline. Precision-safe by construction.
    cross_encoder_supersession: bool = False

    # Selects the Stage-2 extraction backend (see VALID_EXTRACTORS). Default
    # `legacy` (the deterministic keyword pipeline) so existing projects are
    # unchanged until they opt in. The `MEMLORA_EXTRACTOR` env var, when set,
    # overrides this for ops/tests. The encoder heads fail open to legacy when
    # their model artifacts are absent, so a non-legacy value is always safe.
    extractor: str = "legacy"
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
        if "cross_encoder_supersession" in data:
            kwargs["cross_encoder_supersession"] = bool(data["cross_encoder_supersession"])
        if "extractor" in data:
            extractor = str(data["extractor"]).lower()
            if extractor not in VALID_EXTRACTORS:
                raise ValueError(
                    f"invalid extractor {extractor!r}; expected one of {sorted(VALID_EXTRACTORS)}"
                )
            kwargs["extractor"] = extractor
        if "query_time_injection" in data:
            kwargs["query_time_injection"] = bool(data["query_time_injection"])
        if "query_injection_threshold" in data:
            kwargs["query_injection_threshold"] = float(data["query_injection_threshold"])
        if "query_injection_max_tokens" in data:
            kwargs["query_injection_max_tokens"] = int(data["query_injection_max_tokens"])
        if "capture_subagents" in data:
            kwargs["capture_subagents"] = bool(data["capture_subagents"])
        if "section_budgets" in data and isinstance(data["section_budgets"], dict):
            sb = data["section_budgets"]
            kwargs["section_budgets"] = SectionBudgets(
                **{k: int(v) for k, v in sb.items() if k in SectionBudgets.__dataclass_fields__}
            )

        return kwargs
