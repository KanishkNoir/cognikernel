"""Phase A statement-quality audit — three hosted LLMs independently pre-label
the blinded pool against the fixed codebook (Task 5A, sub-step 3a).

This is offline eval construction, not a runtime feature: nothing here ships
in the `cognikernel` package or runs during a CogniKernel session. It costs
real API tokens and needs `TOGETHER_API_KEY`.

Each of the three models labels every row in the pool independently. A model
whose reply cannot be parsed against the codebook gets one corrective retry;
if that also fails, the row is recorded as a parse failure (in the manifest)
and omitted from that model's output file, rather than written with an
invalid/empty label that would corrupt every downstream row via
`audit_report.load_labeled`'s all-or-nothing validation. A model with a high
parse-failure rate is not a usable labeler — report the rate, don't hide it.

Reads : research/statement_audit/pool_<stamp>.jsonl   (id, event_type, text only
        — never the pool_meta_<stamp>.jsonl sidecar; that carries blinding data)
Writes: research/statement_audit/labels_<model-slug>_<stamp>.jsonl (pool schema,
        consumable unchanged by `scripts/audit_report.py --labeler-b`)
        research/statement_audit/manifest_<stamp>.json (models, prompt SHA-256,
        pool filename, counts, per-model parse-failure rate)
        research/statement_audit/llm_cache/<sha256>.json (response cache,
        keyed by sha256(model + prompt), so a re-run never re-spends)

Usage: uv run --with openai python scripts/audit_label_llm.py [--pool PATH]
           [--models deepseek-ai/DeepSeek-V4-Pro,moonshotai/Kimi-K2.6,openai/gpt-oss-120b,zai-org/GLM-5.2]
           [--limit N]
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

# DeepSeek-V4-Pro and Kimi-K2.6 are reasoning models that spend hidden
# thinking tokens before any visible output. At max_tokens=50, DeepSeek
# returned finish_reason="length" with truncated JSON; the worse failure mode
# (empty content that parses as a confident wrong answer) is avoided by a
# generous cap. A trivial {"labels":["CLEAN"]} reply cost 39 output tokens on
# DeepSeek, 74 on Kimi, 85 on gpt-oss — 1024 is cheap headroom, not a signal
# we expect to use.
_MAX_COMPLETION_TOKENS = 1024
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


def _prompt_cache_key(model: str, messages: list[dict]) -> str:
    payload = model + json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _call_model(model: str, messages: list[dict], cache_dir: Path) -> tuple[str, str]:
    """(content, finish_reason) for (model, messages), via an on-disk cache
    keyed by sha256(model + prompt) so a re-run or an added model never
    re-spends on work already done. finish_reason is surfaced (not just
    logged) so a truncated ("length") reply can be told apart from a genuine
    codebook violation when the parse-failure rate is reported."""
    key = _prompt_cache_key(model, messages)
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return cached["content"], cached.get("finish_reason", "")

    client = _together_client()

    def _once():
        return client.chat.completions.create(
            model=model, messages=messages, temperature=0,
            max_tokens=_MAX_COMPLETION_TOKENS,
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
        "usage_in": getattr(usage, "prompt_tokens", 0) or 0,
        "usage_out": getattr(usage, "completion_tokens", 0) or 0,
    }, ensure_ascii=False), encoding="utf-8")
    return content, finish_reason


def _label_one(row: dict, model: str, cache_dir: Path) -> tuple[dict | None, bool, list[str]]:
    """Label one pool row with one model.

    Returns (output_row, parse_failed, finish_reasons). On a codebook
    violation, one corrective retry is made; if that also fails, output_row
    is None and the row is omitted from the output file entirely — writing
    it with an empty or invalid label would corrupt the whole file under
    audit_report.load_labeled's all-or-nothing validation. finish_reasons
    lets the caller separate "model can't follow the codebook" from "1024
    tokens wasn't enough" (finish_reason == "length") in the failure count.
    """
    messages = build_prompt(row)
    raw, fr1 = _call_model(model, messages, cache_dir)
    finish_reasons = [fr1]
    try:
        labels, notes = parse_labels(raw)
    except ValueError as first_err:
        correction = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                f"That reply violated the codebook ({first_err}). Reply again with "
                "ONLY a JSON object of the form {\"labels\": [\"CODE\", ...], "
                "\"notes\": \"...\"}, using codes only from the codebook above, "
                "CLEAN alone if it applies, and a non-empty note if OTHER is used."
            )},
        ]
        raw2, fr2 = _call_model(model, correction, cache_dir)
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
        "temperature": 0,
        "max_tokens": _MAX_COMPLETION_TOKENS,
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
            labeled, failed, finish_reasons = _label_one(row, model, CACHE_DIR)
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
