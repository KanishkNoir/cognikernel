from cognikernel.extraction.pipeline import SessionMetadata, extract_session, persist_events
from cognikernel.extraction.tokenize import Sentence, tokenize
from cognikernel.extraction.trie import TrieMatch, TrieScanner, get_scanner
from cognikernel.extraction.hashing import compute_content_hash, normalize_for_hash
from cognikernel.extraction.classifier import classify_constraint, classify_event
from cognikernel.extraction.git_augment import (
    FileChange,
    extract_git_events,
    infer_intent_from_path,
    cross_reference_signals,
    run_git_diff,
)

__all__ = [
    # pipeline
    "SessionMetadata",
    "extract_session",
    "persist_events",
    # tokenizer
    "Sentence",
    "tokenize",
    # trie
    "TrieMatch",
    "TrieScanner",
    "get_scanner",
    # hashing
    "compute_content_hash",
    "normalize_for_hash",
    # classifier
    "classify_constraint",
    "classify_event",
    # git
    "FileChange",
    "extract_git_events",
    "infer_intent_from_path",
    "cross_reference_signals",
    "run_git_diff",
]
