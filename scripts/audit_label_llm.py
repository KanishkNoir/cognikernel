"""Phase A statement-quality audit — three hosted LLMs independently pre-label
the blinded pool against the fixed codebook (Task 5A, sub-step 3a).

This is offline eval construction, not a runtime feature: nothing here ships
in the `cognikernel` package or runs during a CogniKernel session. It costs
real API tokens and needs `TOGETHER_API_KEY`.

Each model labels every row in the pool independently. A reply that fails to
parse against the codebook for a genuine reason (malformed JSON, an unknown
code, CLEAN mixed with other codes, OTHER without a note) gets one corrective
retry; a reply that was truncated (`finish_reason == "length"`) does NOT get
that retry, since a codebook-correction message cannot produce tokens the
model never generated — it is recorded as a failure directly. Either way, if
the row ends up unresolved it is omitted from that model's output file rather
than written with an invalid/empty label that would corrupt every downstream
row via `audit_report.load_labeled`'s all-or-nothing validation. A model with
a high parse-failure rate is not a usable labeler — report the rate, and
whether it is truncation or a real codebook violation, don't hide either.

Reads : research/statement_audit/pool_<stamp>.jsonl   (id, event_type, text only
        — never the pool_meta_<stamp>.jsonl sidecar; that carries blinding data)
Writes: research/statement_audit/labels_<model-slug>_<stamp>.jsonl (pool schema,
        consumable unchanged by `scripts/audit_report.py --labeler-b`)
        research/statement_audit/manifest_<stamp>.json (models, prompt SHA-256,
        pool filename, counts, per-model parse-failure and truncation rate)
        research/statement_audit/llm_cache/<sha256>.json (response cache,
        keyed by sha256(model + prompt + max_tokens + temperature), so a
        re-run, an added model, or a changed --max-tokens never re-spends
        on work already done)

Usage: uv run --with openai python scripts/audit_label_llm.py [--pool PATH]
           [--models deepseek-ai/DeepSeek-V4-Pro,moonshotai/Kimi-K2.6,openai/gpt-oss-120b,zai-org/GLM-5.2]
           [--max-tokens 8192] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("research/statement_audit")
CACHE_DIR = OUT_DIR / "llm_cache"

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
MODELS = ("deepseek-ai/DeepSeek-V4-Pro", "moonshotai/Kimi-K2.6", "openai/gpt-oss-120b",
          "zai-org/GLM-5.2")

# DeepSeek-V4-Pro, Kimi-K2.6 and GLM-5.2 are reasoning models that spend
# hidden thinking tokens before any visible output. An initial 1024 cap was
# calibrated on a trivial smoke prompt ("reply with this JSON") rather than
# the real prompt carrying the full 9-code codebook, which makes these models
# think much harder: a live 60-row run showed max successful completions of
# 1021/1016/974 tokens (essentially AT the 1024 ceiling) with medians of
# 600-800, and every parse failure on 3 of 4 models was verified as
# finish_reason="length" with EMPTY content on both attempts — token
# starvation, not a codebook-following defect. 8192 is ~8x the observed
# median, with headroom for the unobserved tail. Configurable via --max-tokens
# because this number is a measured-then-corrected estimate, not a constant.
_DEFAULT_MAX_TOKENS = 8192
_TEMPERATURE = 0
_MAX_ATTEMPTS = 6
_BACKOFF_BASE_S = 2.0
_BACKOFF_CAP_S = 30.0

# The codebook, reproduced verbatim from
# docs/superpowers/plans/2026-07-25-statement-quality-audit.md ("The labeling
# codebook" section). Its SHA-256 goes in the manifest so an edit here cannot
# silently change what the reported rate means.
_CODEBOOK = """A labeler marks **all** codes that apply. `CLEAN` is exclusive — if `CLEAN` is marked, nothing else may be.

| Code | Definition | Example from the real store |
|---|---|---|
| `DANGLING_REFERENCE` | Opens with or turns on a pronoun / deictic / definite reference whose referent is not in the statement itself. | *"With it, you see three sibling spans with relay.attempt=0,1,2…"* |
| `NOT_A_STATEMENT` | A fragment, table row, heading, or code debris rather than a sentence asserting a fact. | *"Timestamp column type (virtual_keys): TIMESTAMPTZ"* |
| `WRONG_TYPE` | Content does not match its event type — e.g. a definition or rationale stored as `CONSTRAINT_HARD`, an explanation stored as `DECISION`. | `CONSTRAINT_HARD`: *"Non-zero temperature means the caller explicitly wants non-deterministic output."* |
| `NOT_DURABLE` | Narration, status chatter, a question, or a prompt fragment. Should never have been stored at all. **These become abstain-gold in Phase B.** | `DECISION`: *"Now update the cache lookup in router.py to branch on streaming."* |
| `META_TALK` | About CogniKernel / the agent tooling itself rather than the host project. | *"The overridden constraints will be updated by the Stop hook when the session closes."* |
| `COMPOUND` | Carries 2+ independent facts that belong in separate events. | *"Also deferred: learned canonicalizer, cross-encoder reranking, decay tuning, and anything Hopfield-flavored."* |
| `MISSING_SUBJECT` | No grammatical subject, so the reader cannot tell what the statement is about. | *"Fails the build if any file other than app/auth/jwt.py calls .get_secret_value()."* |
| `OTHER` | Defective for a reason not listed. **Requires a note.** | — |
| `CLEAN` | None of the above. Usable as-is. | *"All upstream timeouts surface to the client as 504 Gateway Timeout, never as 500."* |

**Judge the statement standalone.** The question is whether *this text alone*, injected into a future session as authoritative memory, is correct and comprehensible. Do not open the source transcript to resolve a reference — needing to is exactly what `DANGLING_REFERENCE` measures."""

_CODES = ("DANGLING_REFERENCE", "NOT_A_STATEMENT", "WRONG_TYPE", "NOT_DURABLE",
          "META_TALK", "COMPOUND", "MISSING_SUBJECT", "OTHER", "CLEAN")

_SYSTEM_PROMPT = f"""You are labeling stored project-memory statements for a quality audit, using a fixed codebook.

{_CODEBOOK}

You will be given one statement's event_type and text. Reply with ONLY a JSON object of the exact form {{"labels": ["CODE", ...], "notes": "..."}} — no prose, no markdown code fences, nothing before or after the JSON.
- "labels" must be a non-empty list drawn only from: {", ".join(_CODES)}.
- If CLEAN applies, it must be the only entry in "labels".
- If OTHER is used, "notes" must contain a one-sentence explanation; otherwise "notes" may be "".
"""


def build_prompt(row: dict) -> list[dict]:
    """Chat messages carrying the codebook (system) + one statement (user)."""
    user = f"event_type: {row.get('event_type', '')}\nstatement: {row.get('text', '')}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _find_json_objects(text: str) -> list[str]:
    """Every balanced top-level {...} substring, in appearance order.

    Reasoning models often wrap the answer in prose or fences, or show a
    draft JSON object before the final one. A brace-depth scan finds every
    complete top-level object regardless of surrounding text; the caller
    tries them from last to first per the "prefer the last one" rule.
    """
    objs = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i + 1])
                    start = None
    return objs


def parse_labels(raw: str) -> tuple[list[str], str]:
    """(labels, notes) from a model's raw reply, enforcing the codebook.

    Raises ValueError on unparseable output or a codebook violation (unknown
    code, CLEAN mixed with other codes, OTHER without a note). Tolerates JSON
    wrapped in prose or fences by scanning for balanced {...} objects and
    preferring the last one, since a reasoning model may narrate a draft
    answer before the final one.
    """
    if not raw or not raw.strip():
        raise ValueError("empty response")

    candidates = _find_json_objects(raw)
    if not candidates:
        raise ValueError(f"no JSON object found in response: {raw[:200]!r}")

    skip_reason: Exception | None = None
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            skip_reason = exc
            continue
        if not isinstance(obj, dict) or "labels" not in obj:
            skip_reason = ValueError("JSON object has no 'labels' key")
            continue
        labels = obj.get("labels")
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            skip_reason = ValueError("'labels' must be a list of strings")
            continue

        labels = [x.strip().upper() for x in labels if x.strip()]
        notes = obj.get("notes") or ""
        notes = str(notes).strip()

        if not labels:
            raise ValueError("'labels' is empty")
        unknown = [x for x in labels if x not in _CODES]
        if unknown:
            raise ValueError(f"unknown code(s) not in codebook: {unknown}")
        if "CLEAN" in labels and len(labels) > 1:
            raise ValueError("CLEAN is exclusive but was mixed with other codes")
        if "OTHER" in labels and not notes:
            raise ValueError("OTHER requires a note")
        return labels, notes

    raise ValueError(f"no usable JSON object with a 'labels' key found "
                     f"(last issue: {skip_reason})")


# --- Transport --------------------------------------------------------------
#
# Together is OpenAI-compatible; use the `openai` SDK (never urllib/requests
# — Cloudflare fingerprint-blocks them with a bare 403 that looks like an
# auth failure but is not). `openai` is intentionally not imported at module
# scope: it is not a project dependency, and this module is loaded by
# tests/eval/test_statement_audit.py via importlib without it installed.


def _find_env_file() -> Path | None:
    """Search cwd and its parents for a .env file.

    This script may run from a git worktree, whose gitignored files (like
    .env) are not shared with the main checkout — plain Path(".env") can
    silently miss a key that exists one level up in the real repo root.
    """
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        p = candidate / ".env"
        if p.exists():
            return p
    return None


def _load_together_key() -> str:
    """TOGETHER_API_KEY from the environment, or a .env file found by
    searching upward. Never printed, never written to any output file."""
    key = os.environ.get("TOGETHER_API_KEY")
    if key:
        return key
    env_path = _find_env_file()
    if env_path is not None:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TOGETHER_API_KEY"):
                _, _, v = line.partition("=")
                v = v.strip().strip('"').strip("'")
                if v:
                    os.environ["TOGETHER_API_KEY"] = v
                    return v
    raise SystemExit("TOGETHER_API_KEY is not set and no .env with it was found "
                     "in the current directory or its parents.")


_client = None  # module-level singleton: avoid one httpx connection pool per call


def _together_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=_load_together_key(), base_url=TOGETHER_BASE_URL)
    return _client


def _is_retryable(exc: Exception) -> bool:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", 0) >= 500
    return False


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def retry_call(fn, *, max_attempts: int = _MAX_ATTEMPTS, base: float = _BACKOFF_BASE_S,
               cap: float = _BACKOFF_CAP_S):
    """Call fn(), retrying retryable failures with full-jitter backoff,
    honouring a server-sent retry-after header when present. Non-retryable
    errors (4xx auth/bad-request) propagate immediately."""
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised unless retryable
            if not _is_retryable(exc):
                raise
            last = exc
            if attempt == max_attempts - 1:
                break
            after = _retry_after_seconds(exc)
            if after is not None:
                time.sleep(after + random.uniform(0, 1.0))
            else:
                time.sleep(random.uniform(0, min(cap, base * (2 ** attempt))))
    raise last  # type: ignore[misc]


def _prompt_cache_key(model: str, messages: list[dict], max_tokens: int,
                      temperature: float = _TEMPERATURE) -> str:
    """sha256(model + prompt + generation params). max_tokens and temperature
    are part of the key (not just model + prompt) so that raising the cap to
    fix truncation cannot silently replay a stale truncated response cached
    under the old cap — a cache hit under a new max_tokens is only a hit if
    every param that could change the output matches too."""
    payload = json.dumps({"model": model, "messages": messages,
                          "max_tokens": max_tokens, "temperature": temperature},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _call_model(model: str, messages: list[dict], cache_dir: Path,
                max_tokens: int) -> tuple[str, str]:
    """(content, finish_reason) for (model, messages), via an on-disk cache
    keyed by sha256(model + prompt + max_tokens + temperature) so a re-run,
    an added model, or a raised cap never re-spends on work already done.
    finish_reason is surfaced (not just logged) so a truncated ("length")
    reply can be told apart from a genuine codebook violation when the
    parse-failure rate is reported."""
    key = _prompt_cache_key(model, messages, max_tokens)
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return cached["content"], cached.get("finish_reason", "")

    client = _together_client()

    def _once():
        return client.chat.completions.create(
            model=model, messages=messages, temperature=_TEMPERATURE,
            max_tokens=max_tokens,
        )

    resp = retry_call(_once)
    content = resp.choices[0].message.content or ""
    finish_reason = resp.choices[0].finish_reason or ""
    usage = getattr(resp, "usage", None)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        "model": model,
        "content": content,
        "finish_reason": finish_reason,
        "max_tokens": max_tokens,
        "temperature": _TEMPERATURE,
        "usage_in": getattr(usage, "prompt_tokens", 0) or 0,
        "usage_out": getattr(usage, "completion_tokens", 0) or 0,
    }, ensure_ascii=False), encoding="utf-8")
    return content, finish_reason


def _label_one(row: dict, model: str, cache_dir: Path,
               max_tokens: int) -> tuple[dict | None, bool, list[str]]:
    """Label one pool row with one model.

    Returns (output_row, parse_failed, finish_reasons). On a genuine parse or
    codebook violation, one corrective retry is made; if that also fails,
    output_row is None and the row is omitted from the output file entirely
    — writing it with an empty or invalid label would corrupt the whole file
    under audit_report.load_labeled's all-or-nothing validation.

    The corrective retry is skipped when the first attempt's finish_reason
    is "length": a codebook-correction message cannot produce tokens the
    model never generated, so retrying there only doubles the spend without
    any chance of success. That row is recorded as a failure directly.
    finish_reasons lets the caller separate "model can't follow the
    codebook" from "ran out of completion tokens" in the failure count.
    """
    messages = build_prompt(row)
    raw, fr1 = _call_model(model, messages, cache_dir, max_tokens)
    finish_reasons = [fr1]
    try:
        labels, notes = parse_labels(raw)
    except ValueError as first_err:
        if fr1 == "length":
            return None, True, finish_reasons
        correction = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                f"That reply violated the codebook ({first_err}). Reply again with "
                "ONLY a JSON object of the form {\"labels\": [\"CODE\", ...], "
                "\"notes\": \"...\"}, using codes only from the codebook above, "
                "CLEAN alone if it applies, and a non-empty note if OTHER is used."
            )},
        ]
        raw2, fr2 = _call_model(model, correction, cache_dir, max_tokens)
        finish_reasons.append(fr2)
        try:
            labels, notes = parse_labels(raw2)
        except ValueError:
            return None, True, finish_reasons

    out = {"id": row["id"], "event_type": row["event_type"], "text": row["text"],
           "labels": labels, "notes": notes}
    return out, False, finish_reasons


def _slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower()


def _latest_pool() -> Path | None:
    # "pool_*.jsonl" also matches "pool_meta_*.jsonl" — exclude it explicitly,
    # since the meta sidecar carries blinding data and must never be read here.
    pools = sorted(p for p in OUT_DIR.glob("pool_*.jsonl") if "pool_meta_" not in p.name)
    return pools[-1] if pools else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=None,
                    help="pool jsonl (default: most recent research/statement_audit/pool_*.jsonl)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=None, help="label only the first N rows (debug)")
    ap.add_argument("--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS,
                    help="completion token cap per call (default 8192; see module docstring)")
    args = ap.parse_args()

    pool_path = args.pool or _latest_pool()
    if pool_path is None or not pool_path.exists():
        sys.exit("no pool file found under research/statement_audit/; pass --pool explicitly")

    rows = [json.loads(line) for line in
            pool_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Blinding: only id/event_type/text are read from the pool. Never touch
    # the pool_meta_*.jsonl sidecar (heuristic verdicts) from this script.
    rows = [{"id": r["id"], "event_type": r["event_type"], "text": r["text"]} for r in rows]
    if args.limit:
        rows = rows[:args.limit]

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    _load_together_key()  # fail fast before spending anything

    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stamp": stamp,
        "pool_file": str(pool_path),
        "n_rows": len(rows),
        "system_prompt_sha256": hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "temperature": _TEMPERATURE,
        "max_tokens": args.max_tokens,
        "models": models,
        "per_model": {},
    }

    for model in models:
        slug = _slug(model)
        out_path = OUT_DIR / f"labels_{slug}_{stamp}.jsonl"
        out_rows: list[dict] = []
        failed_ids: list[str] = []
        truncated_ids: list[str] = []
        for row in rows:
            labeled, failed, finish_reasons = _label_one(row, model, CACHE_DIR, args.max_tokens)
            if any(fr == "length" for fr in finish_reasons):
                truncated_ids.append(row["id"])
            if failed:
                failed_ids.append(row["id"])
            else:
                out_rows.append(labeled)

        with out_path.open("w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        n = len(rows)
        rate = len(failed_ids) / n if n else 0.0
        failed_and_truncated = sorted(set(failed_ids) & set(truncated_ids))
        manifest["per_model"][model] = {
            "output_file": str(out_path),
            "attempted": n,
            "completed": len(out_rows),
            "parse_failures": len(failed_ids),
            "parse_failure_ids": failed_ids,
            "parse_failure_rate": rate,
            # finish_reason == "length" on any attempt for this row: a
            # truncated reply, not necessarily a codebook violation. Reported
            # separately so max_tokens=1024 headroom can be checked before
            # blaming the model for the parse-failure rate above.
            "truncated_response_count": len(truncated_ids),
            "truncated_response_ids": truncated_ids,
            "parse_failures_caused_by_truncation": len(failed_and_truncated),
        }
        print(f"{model}: {len(out_rows)}/{n} labeled, {len(failed_ids)} parse failures "
              f"({100 * rate:.1f}%), {len(truncated_ids)} truncated responses -> {out_path}")

    manifest_path = OUT_DIR / f"manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
