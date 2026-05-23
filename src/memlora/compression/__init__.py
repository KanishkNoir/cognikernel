"""Stage 3 — Compression and Ranking.

Public API:
  compute_weight        — full multiplicative weight formula
  greedy_fill           — knapsack fill by weight within token budget
  compress_field_level  — post-fill field truncation
  estimate_tokens       — len/4 token estimate for a single event
  recency_factor        — hyperbolic decay: 1/(1+α·t)
  repetition_factor     — logarithmic repetition boost
  activity_factor       — status-based activity multiplier
  centrality_factor     — PageRank-derived centrality multiplier [0.5, 1.5]
  compute_file_centrality — PageRank over an import graph
  BASE_WEIGHT           — per-type base weight priors
  TYPE_MULTIPLIER       — per-type retrieval probability multipliers
"""
from memlora.compression.centrality import centrality_factor, compute_file_centrality
from memlora.compression.greedy import compress_field_level, greedy_fill
from memlora.compression.recency import recency_factor
from memlora.compression.token_count import estimate_tokens
from memlora.compression.weights import (
    BASE_WEIGHT,
    TYPE_MULTIPLIER,
    activity_factor,
    compute_weight,
    repetition_factor,
)

__all__ = [
    "BASE_WEIGHT",
    "TYPE_MULTIPLIER",
    "activity_factor",
    "centrality_factor",
    "compress_field_level",
    "compute_file_centrality",
    "compute_weight",
    "estimate_tokens",
    "greedy_fill",
    "recency_factor",
    "repetition_factor",
]
