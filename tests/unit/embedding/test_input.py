"""Tests for the embedding input composer (E1) — pure, no model."""
from __future__ import annotations

from memlora.embedding.input import embedding_input


class TestEmbeddingInput:
    def test_subject_leads_description(self) -> None:
        out = embedding_input({"subject": "password hashing", "description": "use bcrypt"}, "DECISION")
        assert out == "password hashing: use bcrypt"

    def test_triple_subject_fallback(self) -> None:
        out = embedding_input({"triple": {"subject": "auth"}, "description": "use jwt"}, "DECISION")
        assert out == "auth: use jwt"

    def test_component_leads_with_path(self) -> None:
        out = embedding_input(
            {"path": "api/auth.py", "status": "modified", "description": "refactor"},
            "COMPONENT_STATUS",
        )
        assert out.startswith("api/auth.py")
        assert "refactor" in out

    def test_description_only_when_no_subject(self) -> None:
        assert embedding_input({"description": "use bcrypt"}, "DECISION") == "use bcrypt"

    def test_empty_payload(self) -> None:
        assert embedding_input({}, "DECISION") == ""
