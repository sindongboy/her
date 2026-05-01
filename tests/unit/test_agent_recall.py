"""Unit tests for apps/agent/recall.py — §5.4 three-strategy recall.

All tests use an in-memory SQLite store seeded with fixture data.
No live Gemini calls — embedding is either mocked or not triggered
(recency fallback path).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from apps.agent import RecallContext
from apps.agent.recall import (
    _recent_attachments_for_episode,
    _recency_fallback,
    _structured_recall,
    recall,
)
from apps.memory.store import MemoryStore


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test_recall.db")


@pytest.fixture()
def seeded_store(store: MemoryStore) -> MemoryStore:
    """Seed with people, facts, events, and episodes."""
    # Add people
    mom_id = store.add_person("어머니", relation="mother")
    dad_id = store.add_person("아버지", relation="father")

    # Add facts
    store.conn.execute(
        "INSERT INTO facts (subject_person_id, predicate, object, confidence) VALUES (?,?,?,?)",
        (mom_id, "좋아하는 음식", "단호박 케이크", 0.9),
    )
    store.conn.execute(
        "INSERT INTO facts (subject_person_id, predicate, object, confidence) VALUES (?,?,?,?)",
        (dad_id, "취미", "등산", 0.85),
    )
    # Archived fact — should not appear
    store.conn.execute(
        "INSERT INTO facts (subject_person_id, predicate, object, confidence, archived_at) "
        "VALUES (?,?,?,?,datetime('now'))",
        (mom_id, "이전 취미", "독서", 0.7),
    )

    # Add upcoming events
    store.conn.execute(
        "INSERT INTO events (person_id, type, title, when_at, status) VALUES (?,?,?,?,?)",
        (mom_id, "birthday", "어머니 생신", "2099-06-15 00:00:00", "pending"),
    )
    # Past event — should not appear in structured recall
    store.conn.execute(
        "INSERT INTO events (person_id, type, title, when_at, status) VALUES (?,?,?,?,?)",
        (mom_id, "trip", "제주도 여행", "2020-01-01 00:00:00", "pending"),
    )
    # Cancelled event — should not appear
    store.conn.execute(
        "INSERT INTO events (person_id, type, title, when_at, status) VALUES (?,?,?,?,?)",
        (dad_id, "golf", "골프 약속", "2099-12-01 00:00:00", "cancelled"),
    )

    # Add episodes (for recency fallback)
    store.conn.execute(
        "INSERT INTO episodes (when_at, summary, primary_channel) VALUES "
        "(datetime('now', '-2 days'), '어머니와 케이크 이야기', 'text')"
    )
    store.conn.execute(
        "INSERT INTO episodes (when_at, summary, primary_channel) VALUES "
        "(datetime('now', '-5 days'), '저녁 식사 이야기', 'text')"
    )

    return store


# ── structured recall tests ────────────────────────────────────────────────


class TestStructuredRecall:
    def test_returns_facts_for_matched_person(self, seeded_store: MemoryStore) -> None:
        facts, events = _structured_recall(seeded_store, "어머니 생신 선물 뭐가 좋을까요?")
        assert len(facts) >= 1
        fact_predicates = [f[2] for f in facts]
        assert "좋아하는 음식" in fact_predicates

    def test_archived_fact_excluded(self, seeded_store: MemoryStore) -> None:
        facts, _ = _structured_recall(seeded_store, "어머니 이야기 해줘")
        predicates = [f[2] for f in facts]
        assert "이전 취미" not in predicates

    def test_returns_upcoming_events(self, seeded_store: MemoryStore) -> None:
        _, events = _structured_recall(seeded_store, "어머니 일정 알려줘")
        titles = [e[1] for e in events]
        assert "어머니 생신" in titles

    def test_past_event_excluded(self, seeded_store: MemoryStore) -> None:
        _, events = _structured_recall(seeded_store, "어머니 과거 일정")
        titles = [e[1] for e in events]
        assert "제주도 여행" not in titles

    def test_cancelled_event_excluded(self, seeded_store: MemoryStore) -> None:
        _, events = _structured_recall(seeded_store, "아버지 일정")
        titles = [e[1] for e in events]
        assert "골프 약속" not in titles

    def test_no_match_returns_empty(self, seeded_store: MemoryStore) -> None:
        facts, events = _structured_recall(seeded_store, "오늘 날씨 어때?")
        assert facts == []
        assert events == []

    def test_fact_tuple_structure(self, seeded_store: MemoryStore) -> None:
        facts, _ = _structured_recall(seeded_store, "어머니 취향 알려줘")
        for fact in facts:
            fact_id, person_name, predicate, obj = fact
            assert isinstance(fact_id, int)
            assert isinstance(person_name, str)
            assert isinstance(predicate, str)
            assert isinstance(obj, str)

    def test_multiple_people_in_message(self, seeded_store: MemoryStore) -> None:
        facts, _ = _structured_recall(seeded_store, "어머니와 아버지 이야기")
        person_names = {f[1] for f in facts}
        assert "어머니" in person_names
        assert "아버지" in person_names


# ── recency fallback tests ─────────────────────────────────────────────────


class TestRecencyFallback:
    def test_returns_most_recent_episodes(self, seeded_store: MemoryStore) -> None:
        results = _recency_fallback(seeded_store)
        assert len(results) >= 1
        assert len(results) <= 5  # max 5

    def test_result_structure(self, seeded_store: MemoryStore) -> None:
        results = _recency_fallback(seeded_store)
        for ep_id, summary, score in results:
            assert isinstance(ep_id, int)
            assert isinstance(summary, str)
            assert isinstance(score, float)

    def test_empty_db_returns_empty(self, store: MemoryStore) -> None:
        results = _recency_fallback(store)
        assert results == []

    def test_score_is_1_for_fallback(self, seeded_store: MemoryStore) -> None:
        results = _recency_fallback(seeded_store)
        for _, _, score in results:
            assert score == 1.0


# ── full recall() integration tests (mocked embed) ────────────────────────


def _make_fake_gemini(embed_result: list[float] | None = None) -> Any:
    """Return a mock GeminiClient that returns a canned embedding."""
    mock = MagicMock()
    if embed_result is None:
        # Return a zero vector — distance calculations will still work.
        embed_result = [0.0] * 768
    mock.embed.return_value = embed_result
    return mock


class TestRecallIntegration:
    @pytest.mark.asyncio
    async def test_recall_returns_recall_context(self, seeded_store: MemoryStore) -> None:
        fake_gemini = _make_fake_gemini()
        ctx = await recall(seeded_store, "어머니 생신 뭐가 좋을까요?", fake_gemini)
        assert isinstance(ctx, RecallContext)

    @pytest.mark.asyncio
    async def test_structured_facts_populated(self, seeded_store: MemoryStore) -> None:
        fake_gemini = _make_fake_gemini()
        ctx = await recall(seeded_store, "어머니 취향 알려줘", fake_gemini)
        assert len(ctx.facts) >= 1

    @pytest.mark.asyncio
    async def test_upcoming_events_populated(self, seeded_store: MemoryStore) -> None:
        fake_gemini = _make_fake_gemini()
        ctx = await recall(seeded_store, "어머니 일정 알려줘", fake_gemini)
        event_titles = [e[1] for e in ctx.upcoming_events]
        assert "어머니 생신" in event_titles

    @pytest.mark.asyncio
    async def test_recency_fallback_when_no_semantic(self, seeded_store: MemoryStore) -> None:
        """With empty vec_episodes, semantic fails → recency fallback triggers."""
        # vec_episodes is empty in seeded_store (no embeddings inserted)
        # The semantic strategy should return [] and fallback runs.
        fake_gemini = _make_fake_gemini()
        ctx = await recall(seeded_store, "오늘 일정 뭐야?", fake_gemini)
        # Fallback should give us episode results.
        assert len(ctx.episodes) >= 1

    @pytest.mark.asyncio
    async def test_semantic_failure_falls_back(self, seeded_store: MemoryStore) -> None:
        """If embed() raises, recall should still succeed via fallback."""
        mock = MagicMock()
        mock.embed.side_effect = RuntimeError("API unavailable")
        ctx = await recall(seeded_store, "오늘 일정", mock)
        assert isinstance(ctx, RecallContext)
        # Fallback episodes should be present
        assert len(ctx.episodes) >= 1


# ── Strategy 4: recent attachments ───────────────────────────────────────


class TestRecentAttachmentsForEpisode:
    def test_returns_empty_for_episode_without_attachments(
        self, store: MemoryStore
    ) -> None:
        ep_id = store.add_episode(primary_channel="text")
        result = _recent_attachments_for_episode(store, ep_id)
        assert result == []

    def test_returns_attachments_for_episode(self, store: MemoryStore) -> None:
        ep_id = store.add_episode(primary_channel="text")
        store.add_attachment(
            ep_id,
            sha256="a" * 64,
            path="/tmp/file.txt",
            mime="text/plain",
            ext=".txt",
            byte_size=100,
            description="메모",
        )
        result = _recent_attachments_for_episode(store, ep_id)
        assert len(result) == 1
        att_id, path, description = result[0]
        assert isinstance(att_id, int)
        assert "/tmp/file.txt" in path
        assert description == "메모"

    def test_respects_limit(self, store: MemoryStore) -> None:
        ep_id = store.add_episode(primary_channel="text")
        for i in range(5):
            store.add_attachment(
                ep_id,
                sha256=f"{'a' * 63}{i}",
                path=f"/tmp/file{i}.txt",
                mime="text/plain",
                ext=".txt",
                byte_size=10,
            )
        result = _recent_attachments_for_episode(store, ep_id, limit=3)
        assert len(result) == 3

    def test_does_not_return_attachments_for_other_episode(
        self, store: MemoryStore
    ) -> None:
        ep1 = store.add_episode(primary_channel="text")
        ep2 = store.add_episode(primary_channel="text")
        store.add_attachment(
            ep1,
            sha256="b" * 64,
            path="/tmp/ep1.txt",
            mime="text/plain",
            ext=".txt",
            byte_size=10,
        )
        result = _recent_attachments_for_episode(store, ep2)
        assert result == []

    def test_tuple_structure(self, store: MemoryStore) -> None:
        ep_id = store.add_episode(primary_channel="text")
        store.add_attachment(
            ep_id,
            sha256="c" * 64,
            path="/tmp/doc.pdf",
            mime="application/pdf",
            ext=".pdf",
            byte_size=1024,
        )
        result = _recent_attachments_for_episode(store, ep_id)
        assert len(result) == 1
        att_id, path, desc = result[0]
        assert isinstance(att_id, int)
        assert isinstance(path, str)
        assert isinstance(desc, str)


class TestRecallWithEpisodeId:
    @pytest.mark.asyncio
    async def test_recall_without_episode_id_no_attachments(
        self, seeded_store: MemoryStore
    ) -> None:
        """When episode_id=None, attachments field should be empty."""
        fake_gemini = _make_fake_gemini()
        ctx = await recall(seeded_store, "테스트", fake_gemini)
        assert ctx.attachments == []
        assert ctx.attachment_ids == []

    @pytest.mark.asyncio
    async def test_recall_with_episode_id_includes_attachments(
        self, store: MemoryStore
    ) -> None:
        """When episode_id is given and attachments exist, they are recalled."""
        ep_id = store.add_episode(primary_channel="text")
        store.add_attachment(
            ep_id,
            sha256="d" * 64,
            path="/tmp/test.txt",
            mime="text/plain",
            ext=".txt",
            byte_size=50,
            description="테스트 파일",
        )
        fake_gemini = _make_fake_gemini()
        ctx = await recall(store, "파일 내용 봐줘", fake_gemini, episode_id=ep_id)
        assert len(ctx.attachments) >= 1
        assert len(ctx.attachment_ids) == len(ctx.attachments)
        # Verify the attachment ID is in the list.
        assert ctx.attachment_ids[0] > 0

    @pytest.mark.asyncio
    async def test_recall_attachment_ids_match_attachments(
        self, store: MemoryStore
    ) -> None:
        ep_id = store.add_episode(primary_channel="text")
        store.add_attachment(
            ep_id,
            sha256="e" * 64,
            path="/tmp/notes.md",
            mime="text/markdown",
            ext=".md",
            byte_size=200,
        )
        fake_gemini = _make_fake_gemini()
        ctx = await recall(store, "노트", fake_gemini, episode_id=ep_id)
        assert len(ctx.attachment_ids) == len(ctx.attachments)
        for att_id, (tup_id, _, _) in zip(ctx.attachment_ids, ctx.attachments):
            assert att_id == tup_id
