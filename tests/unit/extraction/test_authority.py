"""Tests for cognikernel.extraction.authority — A-4 helper module."""
from __future__ import annotations

import pytest

from cognikernel.extraction.authority import (
    ASSISTANT_ANSWER_TO_QUESTION,
    ASSISTANT_DECIDED,
    CONFIRMING_AUTHORITIES,
    INFERRED_FROM_CODE,
    LLM,
    USER_STATED,
    default_authority_for_role,
    normalize_subject,
)


class TestNormalizeSubject:
    def test_empty(self) -> None:
        assert normalize_subject("") == ""
        assert normalize_subject("   ") == ""

    def test_basic_lowercase(self) -> None:
        assert normalize_subject("PostgreSQL") == "postgresql"

    def test_strips_punctuation(self) -> None:
        assert normalize_subject("JWT_SECRET_KEY.") == "jwt_secret_key"

    def test_collapses_whitespace(self) -> None:
        assert normalize_subject("Material  UI") == "material ui"

    def test_drops_leading_articles(self) -> None:
        assert normalize_subject("the JWT secret") == "jwt secret"
        assert normalize_subject("a config file") == "config file"
        assert normalize_subject("an env var") == "env var"
        assert normalize_subject("our database") == "database"

    def test_drops_multiple_leading_articles(self) -> None:
        # "the the JWT" would be a typo but should still collapse.
        assert normalize_subject("the the JWT") == "jwt"

    def test_articles_only_at_start(self) -> None:
        """Articles inside the subject are preserved (still part of the name)."""
        assert normalize_subject("App for the User") == "app for the user"

    def test_paraphrase_collapse(self) -> None:
        """The core property: paraphrased mentions become equal."""
        a = normalize_subject("JWT_SECRET_KEY")
        b = normalize_subject("The JWT_SECRET_KEY!")
        c = normalize_subject("the JWT_SECRET_KEY")
        assert a == b == c


class TestDefaultAuthority:
    def test_user_role(self) -> None:
        assert default_authority_for_role("user") == USER_STATED

    def test_assistant_role(self) -> None:
        assert default_authority_for_role("assistant") == ASSISTANT_DECIDED

    def test_unknown_role_defaults_to_assistant(self) -> None:
        # Defensive fallback — never blow up on a role string we don't know.
        assert default_authority_for_role("") == ASSISTANT_DECIDED
        assert default_authority_for_role("system") == ASSISTANT_DECIDED


class TestConstants:
    def test_all_constants_distinct(self) -> None:
        assert len({
            USER_STATED, ASSISTANT_DECIDED, ASSISTANT_ANSWER_TO_QUESTION,
            INFERRED_FROM_CODE, LLM,
        }) == 5

    def test_confirming_authorities_excludes_co_capture(self) -> None:
        """Co-captures must NEVER suppress themselves — would defeat the
        Pending Confirmation flow."""
        assert ASSISTANT_ANSWER_TO_QUESTION not in CONFIRMING_AUTHORITIES

    def test_confirming_authorities_includes_user_stated(self) -> None:
        assert USER_STATED in CONFIRMING_AUTHORITIES
