"""Tests for Episode CRUD in MemoryStore."""

from __future__ import annotations

import sqlite3

import pytest

from apps.memory import Episode, MemoryStore


def test_add_and_get_episode(store: MemoryStore) -> None:
    eid = store.add_episode("hello world", primary_channel="text")
    ep = store.get_episode(eid)
    assert ep is not None
    assert ep.id == eid
    assert ep.summary == "hello world"
    assert ep.primary_channel == "text"
    assert ep.when_at is not None


def test_add_episode_with_explicit_when_at(store: MemoryStore) -> None:
    eid = store.add_episode(when_at="2026-01-01T00:00:00", primary_channel="voice")
    ep = store.get_episode(eid)
    assert ep is not None
    assert ep.when_at == "2026-01-01T00:00:00"
    assert ep.primary_channel == "voice"


def test_add_episode_no_summary(store: MemoryStore) -> None:
    eid = store.add_episode(primary_channel="mixed")
    ep = store.get_episode(eid)
    assert ep is not None
    assert ep.summary is None


def test_get_episode_not_found(store: MemoryStore) -> None:
    assert store.get_episode(9999) is None


def test_list_recent_episodes_order(store: MemoryStore) -> None:
    store.add_episode("old", when_at="2026-01-01T00:00:00", primary_channel="text")
    store.add_episode("newer", when_at="2026-04-01T00:00:00", primary_channel="voice")
    store.add_episode("newest", when_at="2026-04-30T00:00:00", primary_channel="mixed")

    episodes = store.list_recent_episodes(limit=10)
    assert len(episodes) == 3
    # Should be ordered descending by when_at
    assert episodes[0].summary == "newest"
    assert episodes[1].summary == "newer"
    assert episodes[2].summary == "old"


def test_list_recent_episodes_limit(store: MemoryStore) -> None:
    for i in range(5):
        store.add_episode(f"ep{i}", primary_channel="text")
    episodes = store.list_recent_episodes(limit=3)
    assert len(episodes) == 3


def test_set_episode_summary(store: MemoryStore) -> None:
    eid = store.add_episode(primary_channel="text")
    store.set_episode_summary(eid, "updated summary")
    ep = store.get_episode(eid)
    assert ep is not None
    assert ep.summary == "updated summary"


def test_episode_primary_channel_check_constraint(store: MemoryStore) -> None:
    # Python guard raises ValueError; DB constraint would raise IntegrityError.
    # Either is acceptable — the key invariant is that invalid channels are rejected.
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        store.add_episode(primary_channel="fax")


def test_episode_is_frozen_dataclass(store: MemoryStore) -> None:
    eid = store.add_episode(summary="x", primary_channel="text")
    ep = store.get_episode(eid)
    assert isinstance(ep, Episode)
    with pytest.raises(Exception):
        ep.summary = "mutated"  # type: ignore[misc]
