"""Tests for Episode embedding upsert and semantic recall."""

from __future__ import annotations

import struct

import pytest

from apps.memory import EMBED_DIM, EMBED_MODEL_ID, EpisodeMatch, MemoryStore


def _vec(value: float, dim: int = EMBED_DIM) -> list[float]:
    return [value] * dim


def test_upsert_embedding_round_trip(store: MemoryStore) -> None:
    eid = store.add_episode("embed me", primary_channel="text")
    vec = _vec(0.1)
    store.upsert_episode_embedding(
        eid, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
    )

    results = store.search_episodes_by_embedding(vec, limit=1)
    assert len(results) == 1
    assert results[0].episode.id == eid


def test_upsert_embedding_idempotent(store: MemoryStore) -> None:
    eid = store.add_episode("twice", primary_channel="text")
    vec = _vec(0.3)
    store.upsert_episode_embedding(
        eid, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
    )
    # Second upsert should not raise
    store.upsert_episode_embedding(
        eid, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
    )
    results = store.search_episodes_by_embedding(vec, limit=5)
    assert len(results) == 1


def test_upsert_embedding_wrong_dim_raises(store: MemoryStore) -> None:
    eid = store.add_episode("bad dim", primary_channel="text")
    with pytest.raises(ValueError, match="vector length"):
        store.upsert_episode_embedding(
            eid,
            [0.1] * 100,  # wrong dim
            model_id=EMBED_MODEL_ID,
            dim=EMBED_DIM,
            task_type="RETRIEVAL_DOCUMENT",
        )


def test_search_returns_episode_match_type(store: MemoryStore) -> None:
    eid = store.add_episode("typed", primary_channel="voice")
    vec = _vec(0.5)
    store.upsert_episode_embedding(
        eid, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
    )
    results = store.search_episodes_by_embedding(vec, limit=1)
    assert len(results) == 1
    m = results[0]
    assert isinstance(m, EpisodeMatch)
    assert m.distance >= 0.0
    assert m.score > 0.0


def test_recency_boosts_recent_episode(store: MemoryStore) -> None:
    """Recent episode should rank higher than old one even at same distance."""
    old_id = store.add_episode("old ep", when_at="2020-01-01T00:00:00", primary_channel="text")
    new_id = store.add_episode("new ep", primary_channel="text")  # uses CURRENT_TIMESTAMP

    # Give both the same vector (distance = 0 for both)
    vec = _vec(0.1)
    store.upsert_episode_embedding(
        old_id, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
    )
    store.upsert_episode_embedding(
        new_id, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
    )

    results = store.search_episodes_by_embedding(
        vec, limit=2, recency_days=7, recency_weight=1.5
    )
    assert len(results) == 2
    # Recent episode should have higher score (boosted by recency_weight)
    recent_match = next(m for m in results if m.episode.id == new_id)
    old_match = next(m for m in results if m.episode.id == old_id)
    assert recent_match.score > old_match.score
    # Top result should be the recent one
    assert results[0].episode.id == new_id


def test_search_empty_db_returns_empty(store: MemoryStore) -> None:
    results = store.search_episodes_by_embedding(_vec(0.1), limit=5)
    assert results == []


def test_search_respects_limit(store: MemoryStore) -> None:
    vec = _vec(0.2)
    for i in range(10):
        eid = store.add_episode(f"ep{i}", primary_channel="text")
        store.upsert_episode_embedding(
            eid,
            [float(i) / 100 + 0.1] * EMBED_DIM,
            model_id=EMBED_MODEL_ID,
            dim=EMBED_DIM,
            task_type="RETRIEVAL_DOCUMENT",
        )
    results = store.search_episodes_by_embedding(vec, limit=3)
    assert len(results) <= 3
