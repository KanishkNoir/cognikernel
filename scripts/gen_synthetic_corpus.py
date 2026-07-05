"""Generate grounded synthetic training data via the OpenAI API.

Fills the cells real mining can't cover cheaply: the register × label matrix
across MANY domains — especially the registers the baseline eval proved weak
(label_value 0.12, table_row 0.36, narration-shaped decisions) — and the
supersession pair kinds (value_evolution positives, restatement negatives,
same-topic hard negatives).

Design rules:
  - Exemplars in prompts are written FRESH here. Never paste eval items
    (research/model_eval) into prompts — that leaks the eval into training.
  - Every request is disk-cached (train_corpus/synth_cache/) so re-runs are free.
  - stdlib urllib only — no new project dependencies.
  - OPENAI_API_KEY from env; model via --model / OPENAI_MODEL (default gpt-4o-mini).

Outputs:
  research/train_corpus/synth_sentences.jsonl {text,label,register,domain,source}
  research/train_corpus/synth_pairs.jsonl     {new_text,old_text,label,kind,domain,source}

Usage:
  uv run python scripts/gen_synthetic_corpus.py --per-cell 8 --domains 6   # pilot
  uv run python scripts/gen_synthetic_corpus.py --per-cell 12 --domains 24 # full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
random.seed(20260704)

OUT_DIR = Path("research/train_corpus")
CACHE_DIR = OUT_DIR / "synth_cache"
API_URL = "https://api.openai.com/v1/chat/completions"

DOMAINS = [
    "fintech payments platform", "embedded firmware for industrial sensors",
    "game engine / gameplay systems", "iOS mobile app", "Android mobile app",
    "batch data pipeline / warehouse", "ML training infrastructure",
    "e-commerce storefront frontend", "healthcare records system (HIPAA)",
    "IoT device fleet management", "developer CLI tool", "compiler / language tooling",
    "logistics route optimization", "video streaming service", "social feed backend",
    "search & indexing infrastructure", "auth / identity provider", "usage-based billing",
    "observability / metrics stack", "robotics control software", "CAD plugin ecosystem",
    "email deliverability service", "edge CDN configuration", "desktop Electron app",
]

ONTOLOGY = """Label definitions (sentences from AI-coding-assistant session transcripts):
- DECISION: a concrete choice made for the project (tool, value, design), often with rationale.
- CONSTRAINT_HARD: an inviolable rule/invariant ("must", "never", "always", security/correctness).
- CONSTRAINT_SOFT: a convention or preference (naming, style, structure), violation is tolerable.
- APPROACH_ABANDONED_DO_NOT_RETRY: an approach explicitly rejected/ruled out, must not be re-proposed.
- THREAD: open/ongoing work — what's in progress or planned next session.
- NOISE: everything else — narration, questions, explanations, acknowledgments, meta-talk."""

# (label, register, register_instruction, fresh exemplars)
SENTENCE_CELLS = [
    ("DECISION", "plain",
     "Direct decision statements with a brief rationale.",
     ["We're going with server-driven pagination; the mobile clients keep only a cursor.",
      "Settled on protobuf for the device link — the bandwidth budget rules out JSON."]),
    ("DECISION", "label_value",
     "Label-value form: a parameter name, then a colon, then the decided value with a short parenthetical or dash rationale.",
     ["Heartbeat interval: 15 s (fast enough to detect a dead node before the LB retries).",
      "Max upload size: 25 MB — matches the gateway buffer, anything larger goes to the resumable path."]),
    ("DECISION", "table_row",
     "A markdown table row stating a decision or invariant, exactly as it appears inside a |-delimited table, including leading/trailing pipes.",
     ["| D3 | Session tokens rotate on every privilege change | prevents fixation |",
      "| retention | Raw events kept 30 days, aggregates forever | storage budget |"]),
    ("DECISION", "narration",
     "Assistant narration voice that CONTAINS a decision (an 'I'll …' opener carrying a chosen value or approach).",
     ["I'll cap the worker pool at 8 — beyond that the DB connection pool becomes the bottleneck.",
      "I'll default the debounce to 250ms; typing tests showed anything longer feels laggy."]),
    ("CONSTRAINT_HARD", "plain",
     "Inviolable rules with must/never/always and a correctness or security reason.",
     ["Device credentials never leave the secure element — provisioning happens on-chip.",
      "All timestamps are stored UTC; conversion happens only at render time."]),
    ("CONSTRAINT_HARD", "table_row",
     "A markdown invariant-table row (|-delimited) stating a hard rule.",
     ["| I2 | The ledger is append-only; corrections are reversing entries | audit requirement |"]),
    ("CONSTRAINT_SOFT", "plain",
     "Conventions and preferences: naming, file layout, style. Softer wording than hard rules.",
     ["Component files are PascalCase and colocate their stories and tests.",
      "Prefer composition over inheritance for the gameplay behaviors; subclassing is a last resort."]),
    ("APPROACH_ABANDONED_DO_NOT_RETRY", "plain",
     "Explicit rejections: an approach that was tried or considered and ruled out, with the reason.",
     ["Polling the vendor API is out — their rate limits made us miss webhook-critical windows twice.",
      "We abandoned client-side encryption for search fields; it made substring queries impossible."]),
    ("THREAD", "plain",
     "Open work items: what's in progress or explicitly next.",
     ["Next session: wire the retry queue to the dead-letter topic and add the replay command.",
      "The migration script is half done — schema moves are in, data backfill still pending."]),
    ("NOISE", "narration",
     "Pure assistant narration with NO decision content: announcing what it's about to look at or do.",
     ["Let me look at how the scheduler consumes these events before changing anything.",
      "Now running the integration suite to make sure the refactor holds."]),
    ("NOISE", "explanation",
     "Explanatory/rationale prose that describes how something works, without deciding anything.",
     ["When the cache misses, the request falls through to the origin and the response is written back on the way out.",
      "The optimizer inlines that call because the closure captures no free variables."]),
    ("NOISE", "question",
     "User questions about the system — interrogative, no decision stated.",
     ["Where does the session token get refreshed when the app comes back from background?",
      "Should the exporter batch by count or by time window?"]),
    ("NOISE", "instruction",
     "User prompt directives telling the assistant what to do next (imperative task requests).",
     ["Add pagination to the audit log endpoint and update the client hook.",
      "Refactor the uploader to stream chunks instead of buffering the whole file."]),
    ("NOISE", "memory_meta",
     "Meta-talk about a memory/notes system itself (recalling, recording, prior-session references) rather than a project fact.",
     ["The notes from last session already cover the throttling discussion.",
      "Recording this in the project log so the next session picks it up."]),
    ("NOISE", "ack",
     "Short acknowledgments and confirmations.",
     ["Confirmed — that matches what we set up earlier.", "Done. Both checks pass now."]),
    # ── casual register: how REAL users type in chat. CogniKernel's highest-
    # authority capture path (user_stated) is exactly this — lowercase, typos,
    # slang, terse, no trailing punctuation. Polished-only training would miss it.
    ("DECISION", "casual_chat",
     "Real chat-style USER decision statements: lowercase common, occasional typos, slang "
     "(yep/nah/btw/tbh/lets), terse, often missing punctuation. The decision content must still be concrete.",
     ["yep lets go with postgres, sqlite wont cut it once we have real traffic",
      "ok use stripe checkout for v1, custom payment ui later maybe"]),
    ("DECISION", "casual_update",
     "Chat-style USER messages that CHANGE an earlier decision mid-project (imperative update with a reason).",
     ["switch the default model to the bigger one btw, quality matters more than cost here",
      "actually make the batch size 500 not 100, the runs are way too slow"]),
    ("CONSTRAINT_HARD", "casual_chat",
     "Chat-style USER hard rules — informal wording but inviolable content.",
     ["never log the raw token pls, we had an incident with that last year",
      "whatever you do dont let migrations run automatically in prod"]),
    ("CONSTRAINT_SOFT", "casual_chat",
     "Chat-style USER conventions/preferences.",
     ["id rather we keep all the api types in one file tbh, easier to scan",
      "small thing but use single quotes everywhere, matches the rest"]),
    ("APPROACH_ABANDONED_DO_NOT_RETRY", "casual_chat",
     "Chat-style USER rejections — informal but explicit rule-outs with a reason.",
     ["nah scrap the redis idea, too much ops overhead for a side project",
      "we tried the websocket approach before and it was a nightmare, dont go there again"]),
    ("THREAD", "casual_chat",
     "Chat-style open-work notes from the user.",
     ["gotta do the password reset flow next time, ran out of steam today",
      "login works now, still need the refresh token thing"]),
    ("NOISE", "casual_chat",
     "Chat-style user messages with NO durable project fact: reactions, small questions, banter, vague acks.",
     ["hmm ok what does that flag actually do",
      "looks good ship it", "wait why is that test failing now"]),
    # ── ADVERSARIAL cells (Tier 3b) — the exact top confusions from the error
    # audit. These CONTAIN decision-shaped content but are NOISE by frame, and
    # isolated-sentence volume cannot teach the boundary; contrastive examples can.
    ("NOISE", "meta_wrapped",
     "A sentence that QUOTES or REFERENCES an existing decision inside memory/recall/"
     "narration framing, rather than stating a new project fact. The wrapped content "
     "looks like a decision but the sentence is meta-commentary about prior memory. "
     "These are the model's highest-confidence errors — make them realistic.",
     ["The recall surfaces the earlier decision to standardize on Postgres over SQLite.",
      "As the session notes already record, we ruled out Celery — no need to revisit it.",
      "Memory lists 'all monetary amounts are integer cents' as an established hard rule.",
      "The prior context confirms the argon2id migration was locked in last session."]),
    ("NOISE", "instruction_factish",
     "A USER directive/request that mentions decision-shaped nouns but is an imperative "
     "ASK for work, not a stated decision. Must read as a task request, not a fact.",
     ["List every hard constraint and operational invariant we should lock in now.",
      "Propose the schema and note which columns must be UUID primary keys.",
      "Walk through the retry policy and confirm where max_attempts actually lives."]),
    ("NOISE", "description_present_tense",
     "Present-tense prose DESCRIBING how the system currently behaves — explanatory, "
     "NOT a decree. Same topics as hard rules but stated as description of behavior.",
     ["Groups are tier-isolated, so a request to the frontier group falls through to the cheap group only when every frontier target is down.",
      "The limiter reads the configured caps at startup and caches them in-process, so no Redis lookup happens on the hot path."]),
    ("CONSTRAINT_HARD", "rule_present_tense",
     "The DECREED-rule twin of description_present_tense: same topic/vocabulary but "
     "an inviolable rule (must/never/always). Pairs adversarially with the description "
     "cell so the model learns decree-vs-description, not topic.",
     ["A request must never fall through from the frontier group to the cheap group — explicit tier selection fails loudly instead.",
      "The rate limiter must read its caps from Redis on every window, never from an in-process cache that drifts across instances."]),
]

PAIR_CELLS = [
    ("value_evolution", 1,
     "Pairs where OLD states a decided value/approach and NEW changes that SAME decision to a different value/approach, "
     "phrased so the two share very little vocabulary (paraphrased context, same underlying topic).",
     [{"old": "Retry backoff uses 25% jitter around the exponential delay.",
       "new": "Backoff sleeps a uniform random duration up to the exponential cap — full-jitter style."}]),
    ("explicit_supersession", 1,
     "Pairs where NEW clearly replaces OLD and says so with overlapping vocabulary (switch/migrate/now/instead).",
     [{"old": "Images are resized on upload to three fixed sizes.",
       "new": "Switch image resizing from upload-time fixed sizes to on-demand transforms behind the CDN."}]),
    ("restatement", 0,
     "Pairs where NEW merely RESTATES or quotes OLD (same fact, echo/recitation, maybe prefixed like 'as decided,'). NOT a supersession.",
     [{"old": "The ledger is append-only; corrections are reversing entries.",
       "new": "As established, the ledger stays append-only and corrections are posted as reversing entries."}]),
    ("same_topic_live", 0,
     "Pairs about the SAME subsystem/topic that are BOTH true simultaneously — related but neither replaces the other.",
     [{"old": "The scheduler claims jobs with SELECT FOR UPDATE SKIP LOCKED.",
       "new": "Scheduler heartbeats update the claim row every 10 seconds."}]),
]


CACHE_ONLY = False  # set by --cache-only: emit cached results, never call the API


def _api_key() -> str:
    """Project .env FIRST, then the environment.

    The shell profiles on this machine export a stale key, so the project-scoped
    .env (gitignored) is the authoritative place for the working one.
    """
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY"):
                _, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if val:
                    return val
    return os.environ.get("OPENAI_API_KEY", "")


def _call_openai(model: str, system: str, user: str, cache_key: str) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    if CACHE_ONLY:
        return None
    key = _api_key()
    if not key:
        sys.exit("OPENAI_API_KEY not set (checked <repo>/.env, then the environment)")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 1.0,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            content = json.loads(resp["choices"][0]["message"]["content"])
            cache_file.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
            return content
        except urllib.error.HTTPError as exc:
            # 4xx (except 429) is not retryable — a bad key/model must abort the
            # whole run loudly, not spin retries per cell (the 401 pilot hang).
            if exc.code != 429 and 400 <= exc.code < 500:
                sys.exit(f"OpenAI API {exc.code}: {exc.read().decode()[:300]}")
            wait = 2.0 * (attempt + 1)
            print(f"    api {exc.code}; retry in {wait}s")
            time.sleep(wait)
        except Exception as exc:
            wait = 2.0 * (attempt + 1)
            print(f"    api error ({exc}); retry in {wait}s")
            time.sleep(wait)
    return None


def gen_sentences(model: str, per_cell: int, domains: list[str]) -> list[dict]:
    out = []
    calls = 0
    for label, register, instruction, exemplars in SENTENCE_CELLS:
        for domain in domains:
            user = (
                f"Generate {per_cell} distinct sentences that would appear in an AI coding assistant "
                f"session transcript for a {domain} project.\n"
                f"Target label: {label}. Register: {register} — {instruction}\n"
                f"Style examples of the register (different domain, do NOT copy their content):\n"
                + "\n".join(f"  - {e}" for e in exemplars)
                + "\nVary voice (user vs assistant), specificity, and named technologies. "
                  'Return JSON: {"items": ["sentence", ...]}'
            )
            ck = hashlib.sha256(f"s|{model}|{label}|{register}|{domain}|{per_cell}".encode()).hexdigest()[:20]
            res = _call_openai(model, ONTOLOGY, user, ck)
            calls += 1
            for t in (res or {}).get("items", []):
                if isinstance(t, str) and 25 <= len(t) <= 400:
                    out.append({"text": t.strip(), "label": label, "register": register,
                                "domain": domain, "source": f"synthetic:{model}"})
    print(f"  sentence cells: {calls} calls")
    return out


def gen_pairs(model: str, per_cell: int, domains: list[str]) -> list[dict]:
    out = []
    calls = 0
    for kind, label, instruction, exemplars in PAIR_CELLS:
        for domain in domains:
            user = (
                f"Generate {per_cell} OLD/NEW statement pairs from a {domain} project's decision history.\n"
                f"Pair kind: {kind} (label={label}). {instruction}\n"
                f"Style example (different domain, do NOT copy):\n{json.dumps(exemplars)}\n"
                'Return JSON: {"pairs": [{"old": "...", "new": "..."}, ...]}'
            )
            ck = hashlib.sha256(f"p|{model}|{kind}|{domain}|{per_cell}".encode()).hexdigest()[:20]
            res = _call_openai(model, ONTOLOGY, user, ck)
            calls += 1
            for pr in (res or {}).get("pairs", []):
                if isinstance(pr, dict) and pr.get("old") and pr.get("new"):
                    out.append({"new_text": str(pr["new"]).strip(), "old_text": str(pr["old"]).strip(),
                                "label": label, "kind": kind, "domain": domain,
                                "source": f"synthetic:{model}"})
    print(f"  pair cells: {calls} calls")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--domains", type=int, default=6, help="how many domains to sample")
    ap.add_argument("--cache-only", action="store_true",
                    help="write outputs from cached calls only; never hit the API")
    args = ap.parse_args()
    global CACHE_ONLY
    CACHE_ONLY = args.cache_only

    domains = random.sample(DOMAINS, min(args.domains, len(DOMAINS)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} per_cell={args.per_cell} domains={len(domains)}")
    print(f"  (~{len(SENTENCE_CELLS) * len(domains) + len(PAIR_CELLS) * len(domains)} API calls; cached calls are free)")

    sents = gen_sentences(args.model, args.per_cell, domains)
    pairs = gen_pairs(args.model, args.per_cell, domains)

    with open(OUT_DIR / "synth_sentences.jsonl", "w", encoding="utf-8") as f:
        for it in sents:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "synth_pairs.jsonl", "w", encoding="utf-8") as f:
        for it in pairs:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"\nsentences: {len(sents)}")
    print("  by label:   ", dict(Counter(i['label'] for i in sents)))
    print("  by register:", dict(Counter(i['register'] for i in sents)))
    print(f"pairs: {len(pairs)}")
    print("  by kind:    ", dict(Counter(i['kind'] for i in pairs)))


if __name__ == "__main__":
    main()
