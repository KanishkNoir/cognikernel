"""#41 — structured schema-decision CAPTURE from DDL code blocks.

Recovers schema decisions stated in DDL (which `sanitize` strips, so they never
reach extraction — the measured Taskflow D2 loss) and mints them as DECISION
events that ride into later sessions' context block. The win is **capture**: a
decision that was previously absent from memory now appears in the next
session's block (advisory, like CK-1/PreToolUse surfacing), so the agent is
reminded of it at the relevant prompt.

Scope discipline (research-bounded — see research/decision_key_canonicalization_*):
  - DECLARATIVE DDL only (`CREATE TABLE`), never arbitrary code → no K4
    illustrative-code precision trap.
  - **Table-qualified** subjects: each decision is keyed to its table+role
    (`users primary key type`), so a project with per-table PK choices
    (e.g. BIGSERIAL PK + external_id UUID) does NOT collapse to one line. An
    earlier draft used a project-global role key; auditing the real DBs showed
    it conflated genuinely-distinct decisions (Conductor pk-type: 5→1), the
    "wrong key worse than no key" hazard — so keying is left to the normal
    decision_key ladder over these table-qualified subjects.
  - NO latest-wins authority claim: the captured DDL is in the assistant turn
    (assistant_decided); a later prose re-decision is also assistant_decided, so
    recency would win. Reconciliation across phrasings needs reliable topic
    identity (deferred — the canonicalization bake-off showed it is hard).
"""
from __future__ import annotations

import re

_CREATE_TABLE = re.compile(r"create\s+table(?:\s+if\s+not\s+exists)?\s+\"?(\w+)\"?", re.I)
_COL = re.compile(r"^\s*\"?(\w+)\"?\s+([A-Za-z][\w]*)\b(.*)$")
_PK_INLINE = re.compile(r"primary\s+key", re.I)
_MONEY_COL = re.compile(r"amount|price|balance|total|cost|cents|currency", re.I)
_TS_COL = {"created_at", "updated_at", "due_date", "deleted_at"}
_ILLUSTRATIVE = re.compile(
    r"\b(could be|for example|e\.?g\.?|let'?s start simple|hypothetical|"
    r"placeholder|something like|roughly|pseudo)\b", re.I
)
_VALUE_CANON = {
    "int": "INTEGER", "integer": "INTEGER", "biginteger": "BIGINT",
    "bigint": "BIGINT", "serial": "SERIAL", "bigserial": "BIGSERIAL",
    "uuid": "UUID", "numeric": "NUMERIC", "decimal": "DECIMAL",
    "float": "FLOAT", "real": "REAL", "timestamptz": "TIMESTAMPTZ",
    "timestamp": "TIMESTAMP", "date": "DATE", "datetime": "TIMESTAMP",
}
_PK_TYPES = {"uuid", "serial", "bigserial", "int", "integer", "bigint", "biginteger"}
_MONEY_TYPES = {"numeric", "decimal", "int", "integer", "bigint", "float", "real"}
_TS_TYPES = {"timestamptz", "timestamp", "date", "datetime"}


def _is_ddl(text: str) -> bool:
    return bool(_CREATE_TABLE.search(text) or _PK_INLINE.search(text))


def _mint(meta, subject: str, desc: str, source_role: str):
    try:
        from memlora.extraction.authority import default_authority_for_role
        from memlora.extraction.hashing import compute_content_hash
        from memlora.storage.events import Event
    except Exception:
        return None
    chash = compute_content_hash("DECISION", desc)
    return Event(
        project_id=meta.project_id,
        session_id=meta.session_id,
        event_type="DECISION",
        payload={
            "description": desc, "rationale": "", "confidence": 0.75,
            "source_role": source_role, "matched_phrase": "DDL",
            "affected_files": [], "authority": default_authority_for_role(source_role),
            "provenance": "schema_ddl", "subject": subject,
        },
        content_hash=chash, weight=0.75,
    )


def extract_schema_decisions(sentences, session_meta) -> list:
    """Mint table-qualified schema DECISION events from DDL code blocks.

    Fail-open → []. One decision per (table, role) so a multi-column table
    yields at most one PK / money / timestamp decision.
    """
    out: list = []
    seen: set[str] = set()
    try:
        for s in sentences:
            if not getattr(s, "is_code_block", False):
                continue
            text = getattr(s, "text", "") or ""
            if not _is_ddl(text) or _ILLUSTRATIVE.search(text):
                continue
            role = getattr(s, "role", "assistant")
            table = "table"
            done: set[str] = set()       # (table, role-name) already minted
            for line in text.splitlines():
                ct = _CREATE_TABLE.search(line)
                if ct:
                    table = ct.group(1).lower()
                    done = set()
                    # the CREATE line may also carry an inline column; fall through
                m = _COL.match(line)
                if not m:
                    continue
                col, typ, rest = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
                if col in {"create", "constraint", "primary", "foreign", "unique", "check"}:
                    continue
                canon = _VALUE_CANON.get(typ)
                if canon is None:
                    continue
                kind = subj = desc = None
                if "pk" not in done and _PK_INLINE.search(rest) and typ in _PK_TYPES:
                    kind, subj = "pk", f"{table} primary key type"
                    desc = f"Primary key type ({table}): {canon}"
                elif "money" not in done and _MONEY_COL.search(col) and typ in _MONEY_TYPES:
                    kind, subj = "money", f"{table} {col} money type"
                    desc = f"Money column type ({table}.{col}): {canon}"
                elif "ts" not in done and col in _TS_COL and typ in _TS_TYPES:
                    kind, subj = "ts", f"{table} timestamp type"
                    desc = f"Timestamp column type ({table}): {canon}"
                else:
                    continue
                done.add(kind)
                ev = _mint(session_meta, subj, desc, role)
                if ev is not None and ev.content_hash not in seen:
                    seen.add(ev.content_hash)
                    out.append(ev)
    except Exception:
        return out
    return out
