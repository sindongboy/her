"""Unit tests for apps/agent/anonymize.py.

Tests:
- redact / restore round-trip
- longer-name-first ordering avoids prefix collision
- empty people list
- partial overlaps (e.g., "어머니" vs "할머니" share "머니")
"""

from __future__ import annotations

import pytest

from apps.agent.anonymize import Anonymizer
from apps.memory.store import Person


def _person(pid: int, name: str) -> Person:
    return Person(
        id=pid,
        name=name,
        relation=None,
        birthday=None,
        preferences={},
        created_at="2026-01-01",
        updated_at="2026-01-01",
    )


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def basic_people() -> list[Person]:
    return [
        _person(1, "아내"),
        _person(2, "어머니"),
        _person(3, "아버지"),
    ]


@pytest.fixture()
def overlap_people() -> list[Person]:
    """Names that share a suffix — 어머니 / 할머니 both end in "머니"."""
    return [
        _person(1, "어머니"),
        _person(2, "할머니"),
    ]


# ── tests ─────────────────────────────────────────────────────────────────


class TestRedactRestore:
    def test_round_trip_single_name(self, basic_people: list[Person]) -> None:
        anon = Anonymizer(basic_people)
        original = "아내가 케이크를 좋아해요."
        redacted, alias_map = anon.redact(original)
        assert "아내" not in redacted
        assert alias_map  # at least one alias created
        restored = anon.restore(redacted, alias_map)
        assert restored == original

    def test_round_trip_multiple_names(self, basic_people: list[Person]) -> None:
        anon = Anonymizer(basic_people)
        original = "어머니와 아버지가 여행을 가셨어요."
        redacted, alias_map = anon.redact(original)
        assert "어머니" not in redacted
        assert "아버지" not in redacted
        restored = anon.restore(redacted, alias_map)
        assert restored == original

    def test_name_not_in_text_ignored(self, basic_people: list[Person]) -> None:
        anon = Anonymizer(basic_people)
        original = "오늘 날씨가 좋네요."
        redacted, alias_map = anon.redact(original)
        assert redacted == original
        assert alias_map == {}

    def test_restore_empty_alias_map(self, basic_people: list[Person]) -> None:
        anon = Anonymizer(basic_people)
        text = "P001 went shopping."
        result = anon.restore(text, {})
        assert result == text

    def test_round_trip_all_people(self, basic_people: list[Person]) -> None:
        anon = Anonymizer(basic_people)
        original = "아내와 어머니와 아버지가 모두 있어요."
        redacted, alias_map = anon.redact(original)
        assert len(alias_map) == 3
        restored = anon.restore(redacted, alias_map)
        assert restored == original


class TestLongerNameFirst:
    def test_longer_name_replaced_first(self, overlap_people: list[Person]) -> None:
        """어머니 and 할머니 both contain "머니". Redacting shorter first would corrupt longer."""
        anon = Anonymizer(overlap_people)
        original = "어머니와 할머니가 함께 오셨어요."
        redacted, alias_map = anon.redact(original)
        assert "어머니" not in redacted
        assert "할머니" not in redacted
        # Ensure restored text matches exactly
        restored = anon.restore(redacted, alias_map)
        assert restored == original

    def test_single_longer_name_distinct(self) -> None:
        people = [
            _person(1, "어머니"),
            _person(2, "어머"),  # shorter prefix
        ]
        anon = Anonymizer(people)
        original = "어머니가 오셨어요."
        redacted, alias_map = anon.redact(original)
        restored = anon.restore(redacted, alias_map)
        assert restored == original


class TestEmptyPeople:
    def test_empty_list_noop(self) -> None:
        anon = Anonymizer([])
        text = "아내가 케이크를 좋아해요."
        redacted, alias_map = anon.redact(text)
        assert redacted == text
        assert alias_map == {}

    def test_restore_with_empty_map_noop(self) -> None:
        anon = Anonymizer([])
        text = "Hello world"
        assert anon.restore(text, {}) == text


class TestAliasStability:
    def test_same_name_gets_same_alias(self, basic_people: list[Person]) -> None:
        """Repeated mentions of the same name → same alias within one redact call."""
        anon = Anonymizer(basic_people)
        original = "아내가 아내를 위해 준비했어요."
        redacted, alias_map = anon.redact(original)
        # Should only have one alias for 아내, and it appears twice.
        alias = next(k for k, v in alias_map.items() if v == "아내")
        assert redacted.count(alias) == 2

    def test_aliases_are_p_format(self, basic_people: list[Person]) -> None:
        anon = Anonymizer(basic_people)
        original = "아내와 어머니."
        _, alias_map = anon.redact(original)
        for alias in alias_map:
            assert alias.startswith("P")
            assert alias[1:].isdigit()
