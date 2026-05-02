"""Tests for AgentCore.maybe_remember and the trigger heuristic."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from apps.agent.core import AgentCore, _looks_like_remember_request
from apps.memory.store import MemoryStore


class FakeFlash:
    """Mimics the bits of GeminiClient that maybe_remember actually calls."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[Any] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        parts: list[Any] | None = None,
    ) -> str:
        self.calls.append({"messages": messages, "system": system})
        return json.dumps(self.payload)


class FakeProClient:
    def generate(self, *_a, **_k) -> str:
        return ""

    async def generate_stream(self, *_a, **_k):
        if False:
            yield ""

    def embed(self, *_a, **_k) -> list[float]:
        return [0.0] * 768


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "remember.db")


@pytest.fixture()
def agent(store: MemoryStore) -> AgentCore:
    return AgentCore(store, client=FakeProClient(), enable_anonymization=False)


# ── trigger heuristic ──────────────────────────────────────────────────


class TestTriggerHeuristic:
    @pytest.mark.parametrize("msg", [
        "어머니가 단호박 좋아하셔, 기억해줘",
        "이건 메모해둬: 매주 금요일 외식",
        "잊지마 — 다음 주 치과 예약",
        "remember this: my daughter's birthday is march 5",
        "Make a note that I prefer green tea",
        "딸은 숭의초등학교에 다니고 있어. 이거 기억해.",  # bare 기억해
        "그냥 메모해. 매주 화요일 운동.",                  # bare 메모해
    ])
    def test_match(self, msg: str) -> None:
        assert _looks_like_remember_request(msg)

    @pytest.mark.parametrize("msg", [
        "오늘 날씨 어때?",
        "어머니 생신 선물 추천해줘",   # "추천해줘" should NOT trigger
        "what's the weather like",
    ])
    def test_no_match(self, msg: str) -> None:
        assert not _looks_like_remember_request(msg)


class TestPersonResolution:
    def test_exact_name(self, agent: AgentCore, store: MemoryStore) -> None:
        pid = store.add_person("신유하", relation="딸")
        assert agent._resolve_or_create_person("신유하") == pid

    def test_relation_match(self, agent: AgentCore, store: MemoryStore) -> None:
        """Saying 딸 in chat should resolve to the existing person whose
        relation = 딸, NOT create a duplicate person named '딸'."""
        pid = store.add_person("신유하", relation="딸")
        before = len(store.list_people())
        assert agent._resolve_or_create_person("딸") == pid
        assert len(store.list_people()) == before  # no duplicate

    def test_create_when_unknown(self, agent: AgentCore, store: MemoryStore) -> None:
        store.add_person("어머니", relation="mother")
        out = agent._resolve_or_create_person("처음 만난 사람")
        assert out is not None
        names = {p.name for p in store.list_people()}
        assert "처음 만난 사람" in names


# ── maybe_remember ─────────────────────────────────────────────────────


def test_maybe_remember_no_trigger_returns_none(agent: AgentCore) -> None:
    out = asyncio.run(agent.maybe_remember("오늘 날씨 어때?", session_id=None))
    assert out is None


def test_maybe_remember_persists_fact(agent: AgentCore, store: MemoryStore) -> None:
    flash = FakeFlash({
        "facts": [{
            "subject_person_name": "어머니",
            "predicate": "좋아한다",
            "object": "단호박 케이크",
            "confidence": 0.95,
        }],
        "notes": [],
    })
    agent._flash_client = flash  # inject

    sid = store.add_session()
    out = asyncio.run(agent.maybe_remember("어머니가 단호박 케이크 좋아해, 기억해줘", session_id=sid))
    assert out is not None
    assert len(out["facts"]) == 1
    assert out["facts"][0]["object"] == "단호박 케이크"

    # Person was auto-created
    people = {p.name for p in store.list_people()}
    assert "어머니" in people

    # Fact persisted with the right source_session_id
    person_id = next(p.id for p in store.list_people() if p.name == "어머니")
    facts = store.list_active_facts(person_id)
    assert len(facts) == 1
    assert facts[0].source_session_id == sid
    assert facts[0].object == "단호박 케이크"


def test_maybe_remember_persists_note(agent: AgentCore, store: MemoryStore) -> None:
    flash = FakeFlash({
        "facts": [],
        "notes": [{"content": "매주 금요일 외식", "tags": ["routine"]}],
    })
    agent._flash_client = flash

    sid = store.add_session()
    out = asyncio.run(agent.maybe_remember("이거 메모해둬: 매주 금요일 외식", session_id=sid))
    assert out is not None
    assert len(out["notes"]) == 1

    notes = store.list_notes()
    assert any(n.content == "매주 금요일 외식" and n.tags == ["routine"] for n in notes)


def test_maybe_remember_demotes_subjectless_fact_to_note(
    agent: AgentCore, store: MemoryStore
) -> None:
    """Without a subject_person_name, the extractor's fact becomes a note."""
    flash = FakeFlash({
        "facts": [{
            "subject_person_name": "",
            "predicate": "약속",
            "object": "다음 주 치과",
            "confidence": 0.9,
        }],
        "notes": [],
    })
    agent._flash_client = flash

    sid = store.add_session()
    out = asyncio.run(agent.maybe_remember("다음 주 치과 약속 잊지마", session_id=sid))
    assert out is not None
    assert len(out["facts"]) == 0
    assert len(out["notes"]) == 1
    assert "약속" in out["notes"][0]["content"]


def test_maybe_remember_handles_malformed_json(
    agent: AgentCore, store: MemoryStore
) -> None:
    class BadFlash:
        def generate(self, *_a, **_k) -> str:
            return "this is not json {{"
    agent._flash_client = BadFlash()  # type: ignore[assignment]
    out = asyncio.run(agent.maybe_remember("뭐든 기억해줘", session_id=None))
    assert out is None


def test_maybe_remember_strips_code_fence(agent: AgentCore, store: MemoryStore) -> None:
    """Some Flash responses come back wrapped in ```json fences."""
    class FencedFlash:
        def generate(self, *_a, **_k) -> str:
            return '```json\n{"facts": [], "notes": [{"content": "x", "tags": []}]}\n```'
    agent._flash_client = FencedFlash()  # type: ignore[assignment]
    out = asyncio.run(agent.maybe_remember("이거 기억해줘", session_id=None))
    assert out is not None
    assert len(out["notes"]) == 1
