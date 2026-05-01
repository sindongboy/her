"""Unit tests for apps/agent/core.py — AgentCore orchestrator.

All tests use:
- in-memory SQLite store (per-test tmp file)
- fake GeminiClient injected via `client=` — no live API calls
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apps.agent import AgentCore, AgentResponse, AttachmentRef
from apps.memory.store import MemoryStore


class FakeGeminiClient:
    def __init__(self, response: str = "네, 알겠습니다.") -> None:
        self.response = response
        self.generate_calls: list[dict[str, Any]] = []
        self.stream_chunks: list[str] = ["안녕", "하세요", "!"]
        self.embed_calls: list[str] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        parts: list[Any] | None = None,
    ) -> str:
        self.generate_calls.append({"messages": messages, "system": system, "parts": parts})
        return self.response

    async def generate_stream(  # type: ignore[override]
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        parts: list[Any] | None = None,
    ):
        for chunk in self.stream_chunks:
            yield chunk

    def embed(self, text: str, *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        self.embed_calls.append(text)
        return [0.0] * 768


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test_core.db")


@pytest.fixture()
def fake_client() -> FakeGeminiClient:
    return FakeGeminiClient()


@pytest.fixture()
def agent(store: MemoryStore, fake_client: FakeGeminiClient) -> AgentCore:
    return AgentCore(store, client=fake_client, enable_anonymization=False)


@pytest.fixture()
def agent_with_anon(store: MemoryStore, fake_client: FakeGeminiClient) -> AgentCore:
    return AgentCore(store, client=fake_client, enable_anonymization=True)


# ── basic respond() ──────────────────────────────────────────────────────


class TestRespond:
    @pytest.mark.asyncio
    async def test_returns_agent_response(self, agent: AgentCore) -> None:
        resp = await agent.respond("안녕하세요")
        assert isinstance(resp, AgentResponse)

    @pytest.mark.asyncio
    async def test_response_text_matches_fake_client(
        self, agent: AgentCore, fake_client: FakeGeminiClient
    ) -> None:
        fake_client.response = "테스트 응답입니다."
        resp = await agent.respond("질문")
        assert resp.text == "테스트 응답입니다."

    @pytest.mark.asyncio
    async def test_session_id_is_created(self, agent: AgentCore) -> None:
        resp = await agent.respond("안녕")
        assert isinstance(resp.session_id, int)
        assert resp.session_id > 0

    @pytest.mark.asyncio
    async def test_session_id_reused_if_provided(
        self, agent: AgentCore, store: MemoryStore
    ) -> None:
        existing_id = store.add_session()
        resp = await agent.respond("이어서 대화", session_id=existing_id)
        assert resp.session_id == existing_id

    @pytest.mark.asyncio
    async def test_used_fact_ids_is_list(self, agent: AgentCore) -> None:
        resp = await agent.respond("안녕")
        assert isinstance(resp.used_fact_ids, list)

    @pytest.mark.asyncio
    async def test_used_session_ids_is_list(self, agent: AgentCore) -> None:
        resp = await agent.respond("안녕")
        assert isinstance(resp.used_session_ids, list)

    @pytest.mark.asyncio
    async def test_gemini_generate_called_once(
        self, agent: AgentCore, fake_client: FakeGeminiClient
    ) -> None:
        await agent.respond("질문")
        assert len(fake_client.generate_calls) == 1

    @pytest.mark.asyncio
    async def test_text_addon_in_system_prompt(
        self, agent: AgentCore, fake_client: FakeGeminiClient
    ) -> None:
        await agent.respond("안녕")
        system = fake_client.generate_calls[0]["system"]
        assert "텍스트" in system or "마크다운" in system

    @pytest.mark.asyncio
    async def test_session_summary_updated(
        self, agent: AgentCore, store: MemoryStore
    ) -> None:
        resp = await agent.respond("오늘 날씨 어때?")
        s = store.get_session(resp.session_id)
        assert s is not None
        assert s.summary is not None
        assert len(s.summary) > 0

    @pytest.mark.asyncio
    async def test_messages_persisted(
        self, agent: AgentCore, store: MemoryStore
    ) -> None:
        resp = await agent.respond("안녕")
        msgs = store.list_messages(resp.session_id)
        roles = [m.role for m in msgs]
        assert roles == ["user", "assistant"]


# ── recall integration ────────────────────────────────────────────────────


class TestRecallIntegration:
    @pytest.mark.asyncio
    async def test_facts_included_in_recall(
        self, agent: AgentCore, store: MemoryStore, fake_client: FakeGeminiClient
    ) -> None:
        mom_id = store.add_person("어머니", relation="mother")
        store.conn.execute(
            "INSERT INTO facts (subject_person_id, predicate, object, confidence) "
            "VALUES (?, ?, ?, ?)",
            (mom_id, "좋아하는 음식", "단호박 케이크", 0.9),
        )
        await agent.respond("어머니 좋아하는 음식 뭐야?")
        content = fake_client.generate_calls[0]["messages"][0]["content"]
        assert "단호박 케이크" in content

    @pytest.mark.asyncio
    async def test_upcoming_events_in_recall(
        self, agent: AgentCore, store: MemoryStore, fake_client: FakeGeminiClient
    ) -> None:
        mom_id = store.add_person("어머니", relation="mother")
        store.conn.execute(
            "INSERT INTO events (person_id, type, title, when_at, status) VALUES (?,?,?,?,?)",
            (mom_id, "birthday", "어머니 생신", "2099-06-15", "pending"),
        )
        await agent.respond("어머니 일정 뭐야?")
        content = fake_client.generate_calls[0]["messages"][0]["content"]
        assert "어머니 생신" in content

    @pytest.mark.asyncio
    async def test_notes_included_in_recall(
        self, agent: AgentCore, store: MemoryStore, fake_client: FakeGeminiClient
    ) -> None:
        store.add_note("매주 금요일 외식")
        await agent.respond("외식 계획")
        content = fake_client.generate_calls[0]["messages"][0]["content"]
        assert "외식" in content


# ── anonymization ────────────────────────────────────────────────────────


class TestAnonymization:
    @pytest.mark.asyncio
    async def test_anon_redacts_names_before_llm(
        self,
        agent_with_anon: AgentCore,
        store: MemoryStore,
        fake_client: FakeGeminiClient,
    ) -> None:
        store.add_person("아내", relation="spouse")
        await agent_with_anon.respond("아내가 오늘 뭘 좋아하지?")
        content = fake_client.generate_calls[0]["messages"][0]["content"]
        assert "아내" not in content

    @pytest.mark.asyncio
    async def test_anon_disabled_passes_names(
        self, agent: AgentCore, store: MemoryStore, fake_client: FakeGeminiClient
    ) -> None:
        store.add_person("아내", relation="spouse")
        await agent.respond("아내가 오늘 뭘 좋아하지?")
        content = fake_client.generate_calls[0]["messages"][0]["content"]
        assert "아내" in content


# ── stream_respond() ─────────────────────────────────────────────────────


class TestStreamRespond:
    @pytest.mark.asyncio
    async def test_yields_chunks(
        self, agent: AgentCore, fake_client: FakeGeminiClient
    ) -> None:
        fake_client.stream_chunks = ["안녕", "하세요", "!"]
        chunks: list[str] = []
        async for chunk in agent.stream_respond("안녕"):
            chunks.append(chunk)
        assert chunks == ["안녕", "하세요", "!"]

    @pytest.mark.asyncio
    async def test_session_created_after_stream(
        self, agent: AgentCore, store: MemoryStore
    ) -> None:
        async for _ in agent.stream_respond("스트리밍 테스트"):
            pass
        row = store.conn.execute("SELECT id FROM sessions LIMIT 1").fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_summary_persisted_after_stream(
        self, agent: AgentCore, store: MemoryStore, fake_client: FakeGeminiClient
    ) -> None:
        fake_client.stream_chunks = ["안녕", "하세요"]
        async for _ in agent.stream_respond("스트리밍 질문"):
            pass
        row = store.conn.execute(
            "SELECT id, summary FROM sessions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row["summary"] is not None

    @pytest.mark.asyncio
    async def test_stream_session_id_reused(
        self, agent: AgentCore, store: MemoryStore
    ) -> None:
        existing_id = store.add_session()
        async for _ in agent.stream_respond("이어서", session_id=existing_id):
            pass
        count = store.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert count == 1


# ── edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_message(self, agent: AgentCore) -> None:
        resp = await agent.respond("")
        assert isinstance(resp, AgentResponse)
        assert resp.session_id > 0

    @pytest.mark.asyncio
    async def test_long_summary_truncated(
        self, agent: AgentCore, store: MemoryStore, fake_client: FakeGeminiClient
    ) -> None:
        fake_client.response = "x" * 500
        resp = await agent.respond("y" * 200)
        s = store.get_session(resp.session_id)
        assert s is not None
        assert s.summary is not None
        assert len(s.summary) <= 200


# ── multimodal ───────────────────────────────────────────────────────────


class TestMultimodalAttachments:
    @pytest.fixture()
    def capturing_client(self) -> FakeGeminiClient:
        return FakeGeminiClient()

    @pytest.fixture()
    def agent_capture(
        self, store: MemoryStore, capturing_client: FakeGeminiClient
    ) -> AgentCore:
        return AgentCore(store, client=capturing_client, enable_anonymization=False)

    @pytest.mark.asyncio
    async def test_respond_without_attachments_passes_empty_parts(
        self,
        agent_capture: AgentCore,
        capturing_client: FakeGeminiClient,
    ) -> None:
        await agent_capture.respond("안녕")
        call = capturing_client.generate_calls[0]
        assert call["parts"] is None or call["parts"] == []

    @pytest.mark.asyncio
    async def test_respond_with_txt_attachment_passes_parts(
        self,
        agent_capture: AgentCore,
        capturing_client: FakeGeminiClient,
        tmp_path: Path,
    ) -> None:
        txt = tmp_path / "note.txt"
        txt.write_text("테스트 노트 내용", encoding="utf-8")
        ref = AttachmentRef(path=txt, mime="text/plain")
        await agent_capture.respond("이 파일 봐줘", attachments=[ref])
        call = capturing_client.generate_calls[0]
        assert call["parts"] is not None
        assert len(call["parts"]) == 1

    @pytest.mark.asyncio
    async def test_agent_response_has_used_attachment_ids_field(
        self, agent_capture: AgentCore
    ) -> None:
        resp = await agent_capture.respond("안녕")
        assert hasattr(resp, "used_attachment_ids")
        assert isinstance(resp.used_attachment_ids, list)

    @pytest.mark.asyncio
    async def test_used_attachment_ids_populated_from_recall(
        self, agent_capture: AgentCore, store: MemoryStore
    ) -> None:
        sid = store.add_session()
        store.add_attachment(
            sid,
            sha256="a" * 64,
            path="/tmp/fake.txt",
            mime="text/plain",
            ext=".txt",
            byte_size=10,
        )
        resp = await agent_capture.respond("파일 있어?", session_id=sid)
        assert resp.session_id == sid
        assert isinstance(resp.used_attachment_ids, list)
        assert len(resp.used_attachment_ids) >= 1
