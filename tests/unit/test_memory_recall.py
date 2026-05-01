"""Tests for Session embedding upsert and semantic recall."""

from __future__ import annotations

import pytest

from apps.memory import EMBED_DIM, EMBED_MODEL_ID, MemoryStore, SessionMatch


def _vec(value: float, dim: int = EMBED_DIM) -> list[float]:
    return [value] * dim


def test_upsert_embedding_round_trip(store: MemoryStore) -> None:
    sid = store.add_session(summary="embed me")
    vec = _vec(0.1)
    store.upsert_session_embedding(
        sid, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
    )

    results = store.search_sessions_by_embedding(vec, limit=1)
    assert len(results) == 1
    assert results[0].session.id == sid


def test_upsert_embedding_idempotent(store: MemoryStore) -> None:
    sid = store.add_session(summary="twice")
    vec = _vec(0.3)
    for _ in range(2):
        store.upsert_session_embedding(
            sid, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
        )
    assert len(store.search_sessions_by_embedding(vec, limit=5)) == 1


def test_upsert_embedding_wrong_dim_raises(store: MemoryStore) -> None:
    sid = store.add_session()
    with pytest.raises(ValueError, match="vector length"):
        store.upsert_session_embedding(
            sid,
            [0.1] * 100,
            model_id=EMBED_MODEL_ID,
            dim=EMBED_DIM,
            task_type="RETRIEVAL_DOCUMENT",
        )


def test_search_returns_session_match_type(store: MemoryStore) -> None:
    sid = store.add_session(summary="typed")
    vec = _vec(0.5)
    store.upsert_session_embedding(
        sid, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
    )
    results = store.search_sessions_by_embedding(vec, limit=1)
    assert len(results) == 1
    m = results[0]
    assert isinstance(m, SessionMatch)
    assert m.distance >= 0.0
    assert m.score > 0.0


def test_recency_boosts_recent_session(store: MemoryStore) -> None:
    old_id = store.add_session(summary="old", started_at="2020-01-01T00:00:00")
    new_id = store.add_session(summary="new")  # CURRENT_TIMESTAMP

    vec = _vec(0.1)
    store.upsert_session_embedding(
        old_id, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
    )
    store.upsert_session_embedding(
        new_id, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
    )

    results = store.search_sessions_by_embedding(
        vec, limit=2, recency_days=7, recency_weight=1.5
    )
    assert len(results) == 2
    recent = next(m for m in results if m.session.id == new_id)
    old = next(m for m in results if m.session.id == old_id)
    assert recent.score > old.score
    assert results[0].session.id == new_id


def test_search_empty_db_returns_empty(store: MemoryStore) -> None:
    assert store.search_sessions_by_embedding(_vec(0.1), limit=5) == []


def test_search_respects_limit(store: MemoryStore) -> None:
    vec = _vec(0.2)
    for i in range(10):
        sid = store.add_session(summary=f"s{i}")
        store.upsert_session_embedding(
            sid,
            [float(i) / 100 + 0.1] * EMBED_DIM,
            model_id=EMBED_MODEL_ID,
            dim=EMBED_DIM,
            task_type="RETRIEVAL_DOCUMENT",
        )
    assert len(store.search_sessions_by_embedding(vec, limit=3)) <= 3


# ── per-message embeddings ───────────────────────────────────────────────


def test_upsert_message_embedding_round_trip(store: MemoryStore) -> None:
    sid = store.add_session()
    mid = store.add_message(sid, "user", "안녕")
    vec = _vec(0.4)
    store.upsert_message_embedding(
        mid, vec, model_id=EMBED_MODEL_ID, dim=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"
    )
    # Just ensure no exceptions and meta row exists.
    row = store.conn.execute(
        "SELECT model_id FROM message_embedding_meta WHERE message_id = ?", (mid,)
    ).fetchone()
    assert row is not None
    assert row["model_id"] == EMBED_MODEL_ID
