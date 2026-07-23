"""Stage 4 — Injection layer.

Public API:
  InjectionContext              — dataclass bundling sections + project metadata
  render_injection              — pure render, no budget enforcement
  render_with_budget_enforcement — render + backstop drop loop
  render_or_fallback            — main entry point; handles None ctx and corruption
  partition_events              — split flat event list into section buckets
  make_injection_context        — build InjectionContext from events + metadata
  generate_summary              — deterministic NL summary (no LLM call)
  count_tokens_accurate         — tiktoken count with len/4 fallback
  ProjectionCorruptedError      — raised when projection storage is unreadable
"""
from cognikernel.injection.fallback import ProjectionCorruptedError, render_or_fallback
from cognikernel.injection.ordering import make_injection_context, partition_events
from cognikernel.injection.template import (
    InjectionContext,
    count_tokens_accurate,
    generate_summary,
    render_injection,
    render_with_budget_enforcement,
)

__all__ = [
    "InjectionContext",
    "ProjectionCorruptedError",
    "count_tokens_accurate",
    "generate_summary",
    "make_injection_context",
    "partition_events",
    "render_injection",
    "render_or_fallback",
    "render_with_budget_enforcement",
]
