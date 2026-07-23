"""#41 — structured schema-decision CAPTURE from DDL (table-qualified)."""
from __future__ import annotations

from types import SimpleNamespace

from cognikernel.extraction.decision_key import derive_decision_key
from cognikernel.extraction.schema_decisions import extract_schema_decisions


def _sent(text, role="assistant", code=True):
    return SimpleNamespace(text=text, is_code_block=code, role=role)


META = SimpleNamespace(project_id="p" * 16, session_id="s1")


class TestCapture:
    DDL = ("CREATE TABLE users (\n"
           "    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n"
           "    email TEXT NOT NULL,\n"
           "    created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n"
           ");")

    def test_captures_pk_and_timestamp(self):
        evs = extract_schema_decisions([_sent(self.DDL)], META)
        descs = {e.payload["description"] for e in evs}
        assert "Primary key type (users): UUID" in descs
        assert "Timestamp column type (users): TIMESTAMPTZ" in descs
        assert all(e.event_type == "DECISION" for e in evs)
        assert all(e.payload["provenance"] == "schema_ddl" for e in evs)

    def test_captures_money_column(self):
        ddl = "CREATE TABLE invoices (\n  amount INTEGER NOT NULL,\n  id BIGSERIAL PRIMARY KEY\n);"
        descs = {e.payload["description"] for e in extract_schema_decisions([_sent(ddl)], META)}
        assert "Money column type (invoices.amount): INTEGER" in descs
        assert "Primary key type (invoices): BIGSERIAL" in descs

    def test_illustrative_block_not_minted(self):
        ddl = "-- could be UUID but let's start simple\nCREATE TABLE t (\n  id INTEGER PRIMARY KEY\n);"
        assert extract_schema_decisions([_sent(ddl)], META) == []

    def test_non_ddl_code_block_ignored(self):
        code = "def add(a, b):\n    return a + b  # primary key uuid in a comment"
        assert extract_schema_decisions([_sent(code)], META) == []

    def test_prose_sentences_ignored(self):
        assert extract_schema_decisions([_sent(self.DDL, code=False)], META) == []

    def test_authority_from_turn_role(self):
        user = extract_schema_decisions([_sent(self.DDL, role="user")], META)
        asst = extract_schema_decisions([_sent(self.DDL, role="assistant")], META)
        assert user and asst
        assert user[0].payload["authority"] != asst[0].payload["authority"]


class TestTableQualifiedKeysDoNotCollapse:
    """The gap-2 fix: per-table PK choices must NOT collapse to one decision_key
    (Conductor: BIGSERIAL PK + external_id UUID were distinct decisions)."""

    DUAL = ("CREATE TABLE events (\n  id BIGSERIAL PRIMARY KEY\n);\n"
            "CREATE TABLE sessions (\n  id UUID PRIMARY KEY DEFAULT gen_random_uuid()\n);")

    def test_two_tables_two_pk_decisions(self):
        evs = extract_schema_decisions([_sent(self.DUAL)], META)
        pk = [e for e in evs if "Primary key" in e.payload["description"]]
        assert len(pk) == 2
        descs = {e.payload["description"] for e in pk}
        assert "Primary key type (events): BIGSERIAL" in descs
        assert "Primary key type (sessions): UUID" in descs

    def test_distinct_tables_distinct_keys(self):
        evs = extract_schema_decisions([_sent(self.DUAL)], META)
        keys = {derive_decision_key(e.payload, e.event_type)
                for e in evs if "Primary key" in e.payload["description"]}
        assert len(keys) == 2          # NOT collapsed to one schema:pk-type
        assert all(k for k in keys)    # both non-empty (table-qualified subject)
