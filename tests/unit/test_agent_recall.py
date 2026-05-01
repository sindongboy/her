"""Unit tests for apps/agent/recall.py — sessions/messages/notes recall."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from apps.agent import RecallContext
from apps.agent.recall import (
    _notes_recall,
    _recency_fallback,
    _recent_attachments_for_session,
    _structured_recall,
    recall,
)
from apps.memory.store import MemoryStore


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test_recall.db")


@pytest.fixture()
def seeded_store(store: MemoryStore) -> MemoryStore:
    """Seed with people, facts, events, sessions, messages, notes."""
    mom_id = store.add_person("어머니", relation="mother")
    dad_id = store.add_person("아버지", relation="father")

    store.conn.execute(
        "INSERT INTO facts (subject_person_id, predicate, object, confidence) VALUES (?,?,?,?)",
        (mom_id, "좋아하는 음식", "단호박 케이크", 0.9),
    )
    store.conn.execute(
        "INSERT INTO facts (subject_person_id, predicate, object, confidence) VALUES (?,?,?,?)",
        (dad_id, "취미", "등산", 0.85),
    )
    store.conn.execute(
        "INSERT INTO facts (subject_person_id, predicate, object, confidence, archived_at) "
        "VALUES (?,?,?,?,datetime('now'))",
        (mom_id, "이전 취미", "독서", 0.7),
    )

    store.conn.execute(
        "INSERT INTO events (person_id, type, title, when_at, status) VALUES (?,?,?,?,?)",
        (mom_id, "birthday", "어머니 생신", "2099-06-15 00:00:00", "pending"),
    )
    store.conn.execute(
        "INSERT INTO events (person_id, type, title, when_at, status) VALUES (?,?,?,?,?)",
        (mom_id, "trip", "제주도 여행", "2020-01-01 00:00:00", "pending"),
    )
    store.conn.execute(
        "INSERT INTO events (person_id, type, title, when_at, status) VALUES (?,?,?,?,?)",
        (dad_id, "golf", "골프 약속", "2099-12-01 00:00:00", "cancelled"),
    )

    store.conn.execute(
        "INSERT INTO sessions (started_at, last_active_at, summary) VALUES "
        "(datetime('now', '-2 days'), datetime('now', '-2 days'), '어머니와 케이크 이야기')"
    )
    store.conn.execute(
        "INSERT INTO sessions (started_at, last_active_at, summary) VALUES "
        "(datetime('now', '-5 days'), datetime('now', '-5 days'), '저녁 식사 이야기')"
    )

    store.add_note("매주 금요일 저녁은 외식하기로 함", tags=["routine"])
    store.add_note("아내 워크숍 일정 확인 필요", tags=["todo"])

    return store


# ── structured recall ────────────────────────────────────────────────────


class TestStructuredRecall:
    def test_returns_facts_for_matched_person(self, seeded_store: MemoryStore) -> None:
        facts, _ = _structured_recall(seeded_store, "어머니 생신 선물 뭐가 좋을까요?")
        assert len(facts) >= 1
        predicates = [f[2] for f in facts]
        assert "좋아하는 음식" in predicates

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


# ── recency fallback ─────────────────────────────────────────────────────


class TestRecencyFallback:
    def test_returns_most_recent_sessions(self, seeded_store: MemoryStore) -> None:
        results = _recency_fallback(seeded_store)
        assert 1 <= len(results) <= 5

    def test_result_structure(self, seeded_store: MemoryStore) -> None:
        results = _recency_fallback(seeded_store)
        for sid, summary, score in results:
            assert isinstance(sid, int)
            assert isinstance(summary, str)
            assert isinstance(score, float)

    def test_empty_db_returns_empty(self, store: MemoryStore) -> None:
        assert _recency_fallback(store) == []

    def test_score_is_1_for_fallback(self, seeded_store: MemoryStore) -> None:
        for _, _, score in _recency_fallback(seeded_store):
            assert score == 1.0

    def test_archived_session_excluded(self, store: MemoryStore) -> None:
        sid_active = store.add_session(summary="활성")
        sid_archived = store.add_session(summary="아카이브됨")
        store.archive_session(sid_archived)
        ids = [r[0] for r in _recency_fallback(store)]
        assert sid_active in ids
        assert sid_archived not in ids


# ── notes recall ─────────────────────────────────────────────────────────


class TestNotesRecall:
    def test_returns_matching_note(self, seeded_store: MemoryStore) -> None:
        results = _notes_recall(seeded_store, "외식")
        assert len(results) >= 1
        contents = [r[1] for r in results]
        assert any("외식" in c for c in contents)

    def test_no_match_returns_empty(self, seeded_store: MemoryStore) -> None:
        assert _notes_recall(seeded_store, "전혀관련없는단어xyz") == []

    def test_empty_query_returns_empty(self, seeded_store: MemoryStore) -> None:
        assert _notes_recall(seeded_store, "") == []
        assert _notes_recall(seeded_store, " ") == []

    def test_archived_note_excluded(self, store: MemoryStore) -> None:
        nid = store.add_note("아카이브된 메모 외식")
        store.archive_note(nid)
        results = _notes_recall(store, "외식")
        assert results == []


# ── recall() integration with mocked embed ───────────────────────────────


def _fake_gemini(embed_result: list[float] | None = None) -> Any:
    mock = MagicMock()
    mock.embed.return_value = embed_result if embed_result is not None else [0.0] * 768
    return mock


class TestRecallIntegration:
    @pytest.mark.asyncio
    async def test_returns_recall_context(self, seeded_store: MemoryStore) -> None:
        ctx = await recall(seeded_store, "어머니 생신 뭐가 좋을까요?", _fake_gemini())
        assert isinstance(ctx, RecallContext)

    @pytest.mark.asyncio
    async def test_facts_populated(self, seeded_store: MemoryStore) -> None:
        ctx = await recall(seeded_store, "어머니 취향 알려줘", _fake_gemini())
        assert len(ctx.facts) >= 1

    @pytest.mark.asyncio
    async def test_upcoming_events_populated(self, seeded_store: MemoryStore) -> None:
        ctx = await recall(seeded_store, "어머니 일정 알려줘", _fake_gemini())
        titles = [e[1] for e in ctx.upcoming_events]
        assert "어머니 생신" in titles

    @pytest.mark.asyncio
    async def test_recency_fallback_when_no_semantic(
        self, seeded_store: MemoryStore
    ) -> None:
        # vec_messages is empty in seeded_store → semantic returns []
        ctx = await recall(seeded_store, "오늘 일정 뭐야?", _fake_gemini())
        assert len(ctx.sessions) >= 1

    @pytest.mark.asyncio
    async def test_semantic_failure_falls_back(self, seeded_store: MemoryStore) -> None:
        mock = MagicMock()
        mock.embed.side_effect = RuntimeError("API unavailable")
        ctx = await recall(seeded_store, "오늘 일정", mock)
        assert isinstance(ctx, RecallContext)
        assert len(ctx.sessions) >= 1

    @pytest.mark.asyncio
    async def test_notes_populated_when_keyword_matches(
        self, seeded_store: MemoryStore
    ) -> None:
        ctx = await recall(seeded_store, "외식", _fake_gemini())
        assert len(ctx.notes) >= 1


# ── Recent attachments for session ───────────────────────────────────────


class TestRecentAttachmentsForSession:
    def test_returns_empty_when_no_attachments(self, store: MemoryStore) -> None:
        sid = store.add_session()
        assert _recent_attachments_for_session(store, sid) == []

    def test_returns_attachments_for_session(self, store: MemoryStore) -> None:
        sid = store.add_session()
        store.add_attachment(
            sid,
            sha256="a" * 64,
            path="/tmp/file.txt",
            mime="text/plain",
            ext=".txt",
            byte_size=100,
            description="메모",
        )
        result = _recent_attachments_for_session(store, sid)
        assert len(result) == 1
        att_id, path, description = result[0]
        assert isinstance(att_id, int)
        assert "/tmp/file.txt" in path
        assert description == "메모"

    def test_respects_limit(self, store: MemoryStore) -> None:
        sid = store.add_session()
        for i in range(5):
            store.add_attachment(
                sid,
                sha256=f"{'a' * 63}{i}",
                path=f"/tmp/file{i}.txt",
                mime="text/plain",
                ext=".txt",
                byte_size=10,
            )
        assert len(_recent_attachments_for_session(store, sid, limit=3)) == 3

    def test_does_not_return_attachments_for_other_session(
        self, store: MemoryStore
    ) -> None:
        s1 = store.add_session()
        s2 = store.add_session()
        store.add_attachment(
            s1,
            sha256="b" * 64,
            path="/tmp/s1.txt",
            mime="text/plain",
            ext=".txt",
            byte_size=10,
        )
        assert _recent_attachments_for_session(store, s2) == []


class TestRecallWithSessionId:
    @pytest.mark.asyncio
    async def test_recall_without_session_id_no_attachments(
        self, seeded_store: MemoryStore
    ) -> None:
        ctx = await recall(seeded_store, "테스트", _fake_gemini())
        assert ctx.attachments == []
        assert ctx.attachment_ids == []

    @pytest.mark.asyncio
    async def test_recall_with_session_id_includes_attachments(
        self, store: MemoryStore
    ) -> None:
        sid = store.add_session()
        store.add_attachment(
            sid,
            sha256="d" * 64,
            path="/tmp/test.txt",
            mime="text/plain",
            ext=".txt",
            byte_size=50,
            description="테스트 파일",
        )
        ctx = await recall(store, "파일 내용 봐줘", _fake_gemini(), session_id=sid)
        assert len(ctx.attachments) >= 1
        assert len(ctx.attachment_ids) == len(ctx.attachments)
        assert ctx.attachment_ids[0] > 0
