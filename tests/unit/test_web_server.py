"""Smoke tests for apps/web/server.py REST routes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.memory.store import MemoryStore
from apps.web.eventbus import EventBus
from apps.web.server import create_app


class FakeAgent:
    def __init__(self) -> None:
        self._gemini = type("G", (), {"embed": lambda *_a, **_k: [0.0] * 768})()


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(tmp_path / "web.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def app(store: MemoryStore) -> Any:
    return create_app(agent=FakeAgent(), store=store, bus=EventBus())


@pytest.fixture()
def client(app: Any) -> TestClient:
    return TestClient(app)


def test_index_serves_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "her" in r.text.lower()


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "subscribers" in body


def test_create_then_list_sessions(client: TestClient) -> None:
    assert client.get("/api/sessions").json() == []

    created = client.post("/api/sessions").json()
    assert isinstance(created["id"], int)
    assert created["archived_at"] is None if "archived_at" in created else True

    listed = client.get("/api/sessions").json()
    assert len(listed) == 1
    assert listed[0]["id"] == created["id"]


def test_messages_404_for_unknown_session(client: TestClient) -> None:
    r = client.get("/api/sessions/9999/messages")
    assert r.status_code == 404


def test_get_messages_for_session(
    client: TestClient, store: MemoryStore
) -> None:
    sid = store.add_session()
    store.add_message(sid, "user", "안녕")
    store.add_message(sid, "assistant", "네!")

    r = client.get(f"/api/sessions/{sid}/messages")
    assert r.status_code == 200
    msgs = r.json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "안녕"


def test_archive_session_soft_deletes(
    client: TestClient, store: MemoryStore
) -> None:
    sid = store.add_session(title="bye")
    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

    listed = client.get("/api/sessions").json()
    assert all(s["id"] != sid for s in listed)
    # Row still exists, just archived.
    assert store.get_session(sid) is not None


def test_archive_unknown_session_returns_404(client: TestClient) -> None:
    assert client.delete("/api/sessions/9999").status_code == 404


def test_memory_notes_endpoint(client: TestClient, store: MemoryStore) -> None:
    store.add_note("매주 외식", tags=["routine"])
    notes = client.get("/api/memory/notes").json()
    assert len(notes) == 1
    assert notes[0]["content"] == "매주 외식"
    assert notes[0]["tags"] == ["routine"]


def test_memory_people_endpoint(client: TestClient, store: MemoryStore) -> None:
    store.add_person("어머니", relation="mother")
    people = client.get("/api/memory/people").json()
    assert len(people) == 1
    assert people[0]["name"] == "어머니"
    assert people[0]["relation"] == "mother"


def test_memory_facts_endpoint(client: TestClient, store: MemoryStore) -> None:
    pid = store.add_person("아빠", relation="father")
    store.add_fact(pid, "직업", "의사", confidence=0.9)
    facts = client.get("/api/memory/facts").json()
    assert len(facts) == 1
    assert facts[0]["predicate"] == "직업"
    assert facts[0]["object"] == "의사"
    assert facts[0]["person_name"] == "아빠"


def test_memory_probe_combines_sources(
    client: TestClient, store: MemoryStore
) -> None:
    pid = store.add_person("어머니", relation="mother")
    store.add_fact(pid, "좋아한다", "단호박 케이크", confidence=0.9)
    store.add_note("냉장고에 우유 사기")

    r = client.get("/api/memory/probe")
    assert r.status_code == 200
    body = r.json()
    assert any(n["content"] == "냉장고에 우유 사기" for n in body["notes"])
    assert any(p["name"] == "어머니" for p in body["people"])
    assert any(f["object"] == "단호박 케이크" for f in body["facts"])
    assert isinstance(body["upcoming_events"], list)
