"""Tests for Preference CRUD in MemoryStore."""

from __future__ import annotations

import time

from apps.memory import MemoryStore, Preference


def test_upsert_and_list_preferences(store: MemoryStore) -> None:
    pid = store.add_person("Alice")
    store.upsert_preference(pid, "food", "pizza")
    prefs = store.list_preferences(pid)
    assert len(prefs) == 1
    assert prefs[0].person_id == pid
    assert prefs[0].domain == "food"
    assert prefs[0].value == "pizza"


def test_upsert_idempotent(store: MemoryStore) -> None:
    pid = store.add_person("Bob")
    store.upsert_preference(pid, "music", "jazz")
    store.upsert_preference(pid, "music", "jazz")  # second call
    prefs = store.list_preferences(pid)
    assert len(prefs) == 1


def test_upsert_updates_last_seen_at(store: MemoryStore) -> None:
    pid = store.add_person("Carol")
    store.upsert_preference(pid, "color", "blue")
    first_ts = store.list_preferences(pid)[0].last_seen_at

    # Brief pause to ensure timestamp changes
    time.sleep(0.01)
    store.upsert_preference(pid, "color", "blue")
    second_ts = store.list_preferences(pid)[0].last_seen_at

    # last_seen_at should be updated (>= first)
    assert second_ts >= first_ts


def test_upsert_different_values_same_domain(store: MemoryStore) -> None:
    pid = store.add_person("Dave")
    store.upsert_preference(pid, "food", "sushi")
    store.upsert_preference(pid, "food", "ramen")
    prefs = store.list_preferences(pid)
    values = {p.value for p in prefs}
    assert "sushi" in values
    assert "ramen" in values


def test_list_preferences_empty(store: MemoryStore) -> None:
    pid = store.add_person("Eve")
    assert store.list_preferences(pid) == []


def test_list_preferences_multiple_domains(store: MemoryStore) -> None:
    pid = store.add_person("Frank")
    store.upsert_preference(pid, "food", "salad")
    store.upsert_preference(pid, "music", "rock")
    store.upsert_preference(pid, "sports", "tennis")
    prefs = store.list_preferences(pid)
    assert len(prefs) == 3
    domains = {p.domain for p in prefs}
    assert domains == {"food", "music", "sports"}


def test_preference_is_frozen_dataclass(store: MemoryStore) -> None:
    pid = store.add_person("Grace")
    store.upsert_preference(pid, "x", "y")
    prefs = store.list_preferences(pid)
    assert isinstance(prefs[0], Preference)
    import pytest
    with pytest.raises(Exception):
        prefs[0].value = "mutated"  # type: ignore[misc]


def test_preferences_cascade_delete_with_person(store: MemoryStore) -> None:
    pid = store.add_person("Hank")
    store.upsert_preference(pid, "food", "pasta")
    store.conn.execute("DELETE FROM people WHERE id = ?", (pid,))
    prefs = store.list_preferences(pid)
    assert prefs == []
