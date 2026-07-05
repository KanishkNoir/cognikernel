"""Build the frozen model-eval dataset from REAL local stores and transcripts.

Purpose: a stable yardstick for the learned components — the salience heads
(sentence -> 6-way label) and the supersession cross-encoder (pair -> supersedes)
— so the current models can be baselined and every retrain measured against the
SAME data. The retrain corpus will be external/universal (ADRs, public agent
trajectories); these projects are therefore pure held-out evaluation.

*** DO NOT TRAIN ON THIS DATA. It is the eval set. ***

Sentence labels come in trust tiers (label_source):
  seed      — hand-curated verbatim sentences with hand-assigned labels (highest trust)
  gold      — sentence matches a benchmark gold-fact matcher (question-guarded)
  register  — high-precision register rules (meta-narration/questions/ack -> NOISE)
Every item carries a `register` tag so regressions are attributable per slice.

Supersession pairs (kind):
  real_link        — actual superseded_by links across all local stores, filtered
                     for recitation noise (echo-shaped Jaccard >= 0.85, quote-meta)
  value_evolution  — curated chains whose link texts share little surface (hard recall)
  same_topic_live  — same store+type, topical overlap, both live -> hard negatives
  restatement      — echo-shaped pairs: NOT supersession by R3 doctrine -> negatives
  cross_project    — random cross-store pairs -> easy negatives

Sentences are preprocessed exactly as the deployed broad path sees them
(tokenize -> skip code blocks -> sanitize_description -> content-word floor).

Usage: uv run --extra embedding python scripts/build_model_eval.py
Writes: research/model_eval/{salience_eval.jsonl, supersession_eval.jsonl, README.md}
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
import re
import sqlite3
import sys
import zlib
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
random.seed(1106)

from memlora.delta.supersede import jaccard_similarity, normalize_for_overlap
from memlora.extraction.pipeline import _MEMORY_META_RE, _CONTENT_WORD_RE, _MIN_CONTENT_WORDS
from memlora.extraction.sanitize import sanitize_description
from memlora.extraction.tokenize import is_label_value_line, tokenize
from memlora.extraction.transcript import transcript_from_source
from memlora.utils.subject import derive_subject

OUT_DIR = Path("research/model_eval")

# The four benchmark CK arms (+ two extra runs for register diversity).
SENTENCE_STORES = {
    "relay":     "484b812967d795c6",
    "taskflow":  "961b42e80e47feef",
    "toolbelt":  "d5dce4e1032e6457",
    "conductor": "720e6b7c1e7d2266",
    "gamma":     "792ba53c22a9ceba",
    "mob_c":     "9d61801554d730b8",
}
PROJECTS_DIR = Path.home() / ".memlora" / "projects"

# ── gold-fact matchers (question-guarded positives) ──────────────────────────
# (project, expected_label, matcher) — matcher over the sanitized lowercase text.
GOLD = [
    # Taskflow (run-sheet decisions; see scripts/_taskflow_extraction_recall.py)
    ("taskflow", "CONSTRAINT_HARD", lambda d: "postgres" in d and ("sqlite" in d or "not" in d)),
    ("taskflow", "DECISION",        lambda d: "uuid" in d and ("primary key" in d or "primary keys" in d)),
    ("taskflow", "CONSTRAINT_HARD", lambda d: "alembic" in d and ("only" in d or "never" in d or "all schema" in d)),
    ("taskflow", "CONSTRAINT_HARD", lambda d: "taskflow_jwt_secret" in d),
    ("taskflow", "DECISION",        lambda d: "argon2" in d),
    ("taskflow", "APPROACH_ABANDONED_DO_NOT_RETRY", lambda d: "celery" in d and ("no " in d or "not " in d or "instead" in d or "rejected" in d)),
    # Relay (decision chains; see analysis_omega)
    ("relay", "APPROACH_ABANDONED_DO_NOT_RETRY", lambda d: "in-process counters" in d or ("in-process" in d and "counters" in d and ("out" in d or "fail" in d))),
    ("relay", "DECISION",        lambda d: "counters live in redis" in d or ("redis" in d and "counters" in d and "live" in d)),
    ("relay", "CONSTRAINT_HARD", lambda d: "money" in d and ("integer" in d or "cents" in d)),
    # Toolbelt (contract; see analysis_tb)
    ("toolbelt", "CONSTRAINT_HARD", lambda d: ("never re-implemented" in d or "never reimplemented" in d or ("imported" in d and "re-implement" in d))),
    ("toolbelt", "CONSTRAINT_HARD", lambda d: "retryable" in d and "fatal" in d),
    # Conductor (invariants; see conductor_three_arm.md)
    ("conductor", "CONSTRAINT_HARD", lambda d: "max_attempts" in d and ("guard" in d or "gate" in d or "attempt <" in d)),
    ("conductor", "DECISION",        lambda d: "bigserial" in d),
]

# ── hand-curated seeds (verbatim from stores/transcripts, hand-labeled) ──────
SEEDS = [
    # (label, register, text)
    ("APPROACH_ABANDONED_DO_NOT_RETRY", "plain",
     "In-process counters fail immediately with multiple instances — each instance gets its own budget, so the effective limit becomes configured_limit × N where N is the number of relay instances."),
    ("APPROACH_ABANDONED_DO_NOT_RETRY", "plain",
     "Sticky sessions via the LB is fragile (key rebalancing, instance death resets the counter, uneven load)."),
    ("DECISION", "plain",
     "The Postgres ledger is the source of truth for billing; Redis is the read cache for enforcement."),
    ("DECISION", "plain",
     "Switch the default alias from claude-sonnet to claude-opus — quality matters more than cost for the default tier."),
    ("CONSTRAINT_HARD", "plain",
     "The retry policy comes from toolbelt.retry.RetryPolicy — imported, never re-implemented."),
    ("CONSTRAINT_HARD", "plain",
     "Tools must signal retryable vs fatal failures explicitly — no bare exceptions."),
    ("CONSTRAINT_HARD", "plain",
     "All monetary amounts are integer cents; floats never touch money anywhere in the pipeline."),
    ("CONSTRAINT_SOFT", "plain",
     "Database columns use snake_case; the JSON API uses camelCase, converted at the serialization boundary."),
    ("THREAD", "plain",
     "Now I'll add max_attempts to worker.py (the test imports it), then write the test."),
    ("THREAD", "plain",
     "Next session: wire the JWT login endpoint and the refresh flow — the schema and hashing are done."),
    ("DECISION", "label_value",
     "Max attempts: 3 (schema default; the claim query gates on attempt < max_attempts)."),
    ("DECISION", "label_value",
     "Recovery window: 30 s — a stuck claim is reclaimed after the visibility timeout expires."),
    ("DECISION", "label_value",
     "Open threshold: 3 consecutive failures trips the breaker for that target."),
    ("DECISION", "ddl_adjacent",
     "Primary key type (tasks): UUID with gen_random_uuid() as the default."),
    # Narration-shaped DECISION — the register subtlety the head must get right
    # (an "I'll …" opener does not make a sentence noise when it carries a value).
    ("DECISION", "narration",
     "I'll use 120s as the default (a middle ground: handles long generations, short enough to prevent queue buildup)."),
    ("NOISE", "narration",
     "Let me find the alias configuration in the relay project."),
    ("NOISE", "narration",
     "Let me verify the tree looks right and double-check the error constructor for RateLimitedError."),
    ("NOISE", "question",
     "Where do the counters live? We run multiple gateway instances behind a load balancer."),
    ("NOISE", "question",
     "Does the retry policy need to change for tracing?"),
    ("NOISE", "memory_meta",
     "The session context block records this as a hard constraint from the prior session."),
    ("NOISE", "memory_meta",
     "The recall surfaces the prior decision about the deployment model."),
    ("NOISE", "ack",
     "Confirmed and recorded. Here's the full constraint set for extraction."),
    # Casual register — how real users actually type (the user_stated authority
    # path). Hand-written; the head must handle lowercase/typos/slang.
    ("DECISION", "casual",
     "yep lets go with postgres for this, sqlite wont survive the concurrent writes"),
    ("DECISION", "casual",
     "ok use zustand not redux, way less boilerplate for what we need"),
    ("DECISION", "casual",
     "actually bump the timeout to 30s, the pdf exports keep dying at 10"),
    ("DECISION", "casual",
     "switch the default branch protection to require 2 reviews btw"),
    ("CONSTRAINT_HARD", "casual",
     "never commit the .env file pls, rotate the key if it ever lands in git"),
    ("CONSTRAINT_HARD", "casual",
     "whatever happens dont run the seed script against prod, it truncates tables"),
    ("CONSTRAINT_SOFT", "casual",
     "id prefer all the sql in one queries.py per module tbh, easier to review"),
    ("APPROACH_ABANDONED_DO_NOT_RETRY", "casual",
     "nah drop the graphql idea, rest is fine and the team knows it — dont bring it back"),
    ("APPROACH_ABANDONED_DO_NOT_RETRY", "casual",
     "we already tried caching at the orm layer and it caused stale reads everywhere, not again"),
    ("THREAD", "casual",
     "signup works now, still gotta do email verification next time"),
    ("NOISE", "casual",
     "hmm ok what does that flag actually do"),
    ("NOISE", "casual",
     "looks good ship it"),
    ("NOISE", "casual",
     "wait why is the ci red again"),
    ("NOISE", "casual",
     "lol ok that was the missing import all along"),
    # CONSTRAINT_SOFT is under-represented in the mined data; convention-style seeds.
    ("CONSTRAINT_SOFT", "plain",
     "Prefer relative imports within the package; absolute imports are for cross-package references only."),
    ("CONSTRAINT_SOFT", "plain",
     "Error messages are sentence-case and end without a period."),
    ("CONSTRAINT_SOFT", "plain",
     "Keep route handlers under 30 lines; push logic into service-layer functions."),
    ("CONSTRAINT_SOFT", "plain",
     "Frontend components live in src/components, one component per file, PascalCase filenames."),
]

# ── value-evolution supersession seeds (texts share little surface) ──────────
VALUE_EVOLUTION_PAIRS = [
    ("Retry jitter is full jitter: sleep a uniform random duration between zero and the exponential cap.",
     "Retry backoff uses 25% jitter around the exponential delay."),
    ("Password hashing uses argon2id via passlib; new signups and rehash-on-login migrate existing hashes.",
     "Passwords are hashed with bcrypt (12 rounds) via passlib."),
    ("The default alias now points at claude-opus for the default tier.",
     "Default model alias: claude-sonnet — best cost/quality balance for the default tier."),
    ("Cache entries are invalidated by version stamp; TTL is only the fallback ceiling.",
     "Cache policy: fixed 5-minute TTL on every entry."),
]

_QUESTIONISH = re.compile(r"\?\s*$")
_NARRATION_RE = re.compile(
    r"^(let me |i'll |i will |now i|first,? i|looking at |running |checking |"
    r"here's |here is |perfect|great|good catch|done\.|ok(ay)?[,.! ])", re.I)
# A narration-shaped opener does NOT make a sentence noise when it carries a
# decision ("I'll use 120s as the default …") — found in the precision sample.
_DECISION_CUE = re.compile(
    r"\b(must|never|always|default|use \w+ (as|not|instead)|decided|convention|"
    r"instead of|rather than|lock(ed)? in|\d+\s?(s|ms|seconds|minutes|%|rounds?))\b", re.I)


def _content_ok(desc: str) -> bool:
    return len(_CONTENT_WORD_RE.findall(desc.lower())) >= _MIN_CONTENT_WORDS


def iter_store_sentences(pid: str):
    """Yield (sanitized_text, role) exactly as the deployed broad path sees them."""
    db = PROJECTS_DIR / f"{pid}.db"
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    seen: set[str] = set()
    for r in con.execute("SELECT source_type, content_blob, content_encoding FROM raw_evidence"):
        blob = r["content_blob"]
        if r["content_encoding"] == "zlib":
            blob = zlib.decompress(blob)
        text = transcript_from_source(r["source_type"], blob.decode("utf-8", errors="replace"))
        for s in tokenize(text):
            if s.is_code_block:
                continue
            desc = sanitize_description(s.text.strip())
            if not desc or not _content_ok(desc):
                continue
            key = desc.lower()
            if key in seen:
                continue
            seen.add(key)
            yield desc, s.role
    con.close()


def build_salience() -> list[dict]:
    items: list[dict] = []
    ids: set[str] = set()

    def add(text, label, register, source, label_source):
        iid = hashlib.sha256(text.lower().encode()).hexdigest()[:12]
        if iid in ids:
            return
        ids.add(iid)
        items.append({"id": iid, "text": text, "label": label, "register": register,
                      "source": source, "label_source": label_source})

    for label, register, text in SEEDS:
        add(text, label, register, "curated", "seed")

    # Manual tier BEFORE mining: add() dedups by text, and a hand-assigned label
    # must win over a gold/register auto-label for the same sentence. Text is
    # re-run through the CURRENT sanitizer so the eval reflects what the pipeline
    # produces today — manual_labels.jsonl was captured before the table-
    # scaffolding fix, so its table_row items still carry raw debris otherwise.
    manual = OUT_DIR / "manual_labels.jsonl"
    if manual.exists():
        for line in manual.read_text(encoding="utf-8").splitlines():
            it = json.loads(line)
            clean = sanitize_description(it["text"]) or it["text"]
            add(clean, it["label"], it["register"], it.get("source", "manual"), "manual")

    per_bucket_cap = 25
    bucket_counts: dict[tuple, int] = {}
    for proj, pid in SENTENCE_STORES.items():
        for desc, role in iter_store_sentences(pid):
            low = desc.lower()
            # gold positives (question-guarded)
            if not _QUESTIONISH.search(desc):
                for gproj, glabel, match in GOLD:
                    if gproj == proj and match(low):
                        reg = "label_value" if is_label_value_line(desc) else "plain"
                        b = (proj, glabel, "gold")
                        if bucket_counts.get(b, 0) < per_bucket_cap:
                            bucket_counts[b] = bucket_counts.get(b, 0) + 1
                            add(desc, glabel, reg, proj, "gold")
                        break
                else:
                    pass
            # high-precision NOISE registers
            if _MEMORY_META_RE.search(desc):
                b = (proj, "NOISE", "memory_meta")
                if bucket_counts.get(b, 0) < per_bucket_cap:
                    bucket_counts[b] = bucket_counts.get(b, 0) + 1
                    add(desc, "NOISE", "memory_meta", proj, "register")
            elif role == "user" and _QUESTIONISH.search(desc) and len(desc) < 160:
                b = (proj, "NOISE", "question")
                if bucket_counts.get(b, 0) < 15:
                    bucket_counts[b] = bucket_counts.get(b, 0) + 1
                    add(desc, "NOISE", "question", proj, "register")
            elif _NARRATION_RE.match(desc) and len(desc) < 140 and not _DECISION_CUE.search(desc):
                b = (proj, "NOISE", "narration")
                if bucket_counts.get(b, 0) < 15:
                    bucket_counts[b] = bucket_counts.get(b, 0) + 1
                    add(desc, "NOISE", "narration", proj, "register")
    return items


def _live_events(con, min_words: int = 5):
    out = []
    for r in con.execute(
        "SELECT id, event_type, payload, created_at, superseded_by, archived FROM events"
    ):
        p = json.loads(r["payload"])
        d = (p.get("description") or "").strip()
        if len(normalize_for_overlap(d)) >= min_words:
            out.append({"id": r["id"], "type": r["event_type"], "desc": d,
                        "sup_by": r["superseded_by"], "archived": r["archived"]})
    return out


_QUOTE_META = re.compile(r"^(the hard constraints? say|the session context|memory (shows|says)|per the (block|memory))", re.I)


def build_supersession() -> list[dict]:
    pairs: list[dict] = []
    seen: set[str] = set()

    def add(new, old, label, kind, source):
        key = hashlib.sha256(f"{new.lower()}|{old.lower()}".encode()).hexdigest()[:12]
        if key in seen:
            return
        seen.add(key)
        pairs.append({"id": key, "new_text": new, "old_text": old, "label": label,
                      "kind": kind, "source": source})

    for new, old in VALUE_EVOLUTION_PAIRS:
        add(new, old, 1, "value_evolution", "curated")

    all_dbs = sorted(glob.glob(str(PROJECTS_DIR / "*.db")))
    per_store_live: dict[str, list] = {}
    for dbp in all_dbs:
        pid = Path(dbp).stem
        try:
            con = sqlite3.connect(f"file:{Path(dbp).as_posix()}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            evs = _live_events(con)
        except Exception:
            continue
        by_id = {e["id"]: e for e in evs}
        # real links (filtered), restatement negatives
        n_links = 0
        for e in evs:
            if e["sup_by"] and e["sup_by"] in by_id and n_links < 20:
                new, old = by_id[e["sup_by"]]["desc"], e["desc"]
                if _QUOTE_META.match(new) or _QUOTE_META.match(old):
                    continue
                j = jaccard_similarity(new, old)
                if j >= 0.85:
                    add(new, old, 0, "restatement", pid)   # echo, not evolution (R3)
                else:
                    add(new, old, 1, "real_link", pid)
                    n_links += 1
        # hard negatives: same type, topical overlap, both live, no link either way
        live = [e for e in evs if e["archived"] == 0 and e["sup_by"] is None]
        per_store_live[pid] = live
        random.shuffle(live)
        n_hard = 0
        for i in range(len(live)):
            if n_hard >= 8:
                break
            for k in range(i + 1, min(i + 12, len(live))):
                a, b = live[i], live[k]
                if a["type"] != b["type"]:
                    continue
                j = jaccard_similarity(a["desc"], b["desc"])
                same_subj = derive_subject(a["desc"]) and derive_subject(a["desc"]) == derive_subject(b["desc"])
                if 0.15 <= j < 0.6 or (same_subj and j < 0.6):
                    add(a["desc"], b["desc"], 0, "same_topic_live", pid)
                    n_hard += 1
                    break
        con.close()

    # easy negatives across projects
    stores = [s for s in per_store_live.values() if len(s) >= 3]
    rng = random.Random(7)
    for _ in range(60):
        if len(stores) < 2:
            break
        s1, s2 = rng.sample(stores, 2)
        add(rng.choice(s1)["desc"], rng.choice(s2)["desc"], 0, "cross_project", "mixed")
    return pairs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sal = build_salience()
    sup = build_supersession()

    with open(OUT_DIR / "salience_eval.jsonl", "w", encoding="utf-8") as f:
        for it in sal:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "supersession_eval.jsonl", "w", encoding="utf-8") as f:
        for it in sup:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"salience: {len(sal)} items")
    print("  by label:       ", dict(Counter(i['label'] for i in sal)))
    print("  by register:    ", dict(Counter(i['register'] for i in sal)))
    print("  by label_source:", dict(Counter(i['label_source'] for i in sal)))
    print(f"supersession: {len(sup)} pairs")
    print("  by kind/label:  ", dict(Counter((i['kind'], i['label']) for i in sup)))


if __name__ == "__main__":
    main()
