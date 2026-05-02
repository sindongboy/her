"""CRUD lifecycle tests for /api/memory/{notes,people,facts,events}."""

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
    s = MemoryStore(tmp_path / "crud.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def client(store: MemoryStore) -> TestClient:
    return TestClient(create_app(agent=FakeAgent(), store=store, bus=EventBus()))


# ── notes ────────────────────────────────────────────────────────────────


class TestNotesCRUD:
    def test_full_lifecycle(self, client: TestClient) -> None:
        # create
        r = client.post("/api/memory/notes", json={"content": "외식", "tags": ["routine"]})
        assert r.status_code == 200
        nid = r.json()["id"]

        # list (active)
        listed = client.get("/api/memory/notes").json()
        assert any(n["id"] == nid and n["content"] == "외식" for n in listed)

        # update
        r = client.patch(f"/api/memory/notes/{nid}", json={"content": "외식 금요일"})
        assert r.status_code == 200
        listed = client.get("/api/memory/notes").json()
        assert next(n for n in listed if n["id"] == nid)["content"] == "외식 금요일"

        # archive
        assert client.delete(f"/api/memory/notes/{nid}").status_code == 200
        active = client.get("/api/memory/notes").json()
        assert all(n["id"] != nid for n in active)
        all_ = client.get("/api/memory/notes?include_archived=true").json()
        assert any(n["id"] == nid for n in all_)

        # restore
        assert client.post(f"/api/memory/notes/{nid}/restore").status_code == 200
        active = client.get("/api/memory/notes").json()
        assert any(n["id"] == nid for n in active)

    def test_empty_content_400(self, client: TestClient) -> None:
        assert client.post("/api/memory/notes", json={"content": "  "}).status_code == 400


# ── people ───────────────────────────────────────────────────────────────


class TestPeopleCRUD:
    def test_full_lifecycle(self, client: TestClient) -> None:
        r = client.post("/api/memory/people", json={"name": "어머니", "relation": "mother"})
        pid = r.json()["id"]

        # update
        r = client.patch(f"/api/memory/people/{pid}", json={"birthday": "1960-06-15"})
        assert r.status_code == 200

        listed = client.get("/api/memory/people").json()
        assert next(p for p in listed if p["id"] == pid)["birthday"] == "1960-06-15"

        # archive + restore
        assert client.delete(f"/api/memory/people/{pid}").status_code == 200
        assert all(p["id"] != pid for p in client.get("/api/memory/people").json())
        assert client.post(f"/api/memory/people/{pid}/restore").status_code == 200
        assert any(p["id"] == pid for p in client.get("/api/memory/people").json())

    def test_unknown_id_404(self, client: TestClient) -> None:
        assert client.patch("/api/memory/people/9999", json={"name": "x"}).status_code == 404
        assert client.delete("/api/memory/people/9999").status_code == 404


# ── facts ────────────────────────────────────────────────────────────────


class TestFactsCRUD:
    def test_full_lifecycle(self, client: TestClient, store: MemoryStore) -> None:
        pid = store.add_person("아버지")
        r = client.post("/api/memory/facts", json={
            "subject_person_id": pid,
            "predicate": "취미",
            "object": "등산",
            "confidence": 0.9,
        })
        fid = r.json()["id"]

        # update
        r = client.patch(f"/api/memory/facts/{fid}", json={"object": "낚시"})
        assert r.status_code == 200
        listed = client.get("/api/memory/facts").json()
        assert next(f for f in listed if f["id"] == fid)["object"] == "낚시"

        # archive + restore
        assert client.delete(f"/api/memory/facts/{fid}").status_code == 200
        assert all(f["id"] != fid for f in client.get("/api/memory/facts").json())
        all_ = client.get("/api/memory/facts?include_archived=true").json()
        assert any(f["id"] == fid for f in all_)
        assert client.post(f"/api/memory/facts/{fid}/restore").status_code == 200
        assert any(f["id"] == fid for f in client.get("/api/memory/facts").json())

    def test_invalid_confidence_400(self, client: TestClient, store: MemoryStore) -> None:
        pid = store.add_person("test")
        r = client.post("/api/memory/facts", json={
            "subject_person_id": pid, "predicate": "x", "object": "y", "confidence": 1.5,
        })
        assert r.status_code == 400


# ── events ───────────────────────────────────────────────────────────────


class TestEventsCRUD:
    def test_full_lifecycle(self, client: TestClient) -> None:
        r = client.post("/api/memory/events", json={
            "type": "birthday",
            "title": "어머니 생신",
            "when_at": "2099-06-15T00:00:00",
        })
        eid = r.json()["id"]

        listed = client.get("/api/memory/events").json()
        assert any(e["id"] == eid for e in listed)

        # update
        r = client.patch(f"/api/memory/events/{eid}", json={"title": "어머니 생신🎂"})
        assert r.status_code == 200
        listed = client.get("/api/memory/events").json()
        assert next(e for e in listed if e["id"] == eid)["title"] == "어머니 생신🎂"

        # archive + restore
        assert client.delete(f"/api/memory/events/{eid}").status_code == 200
        assert all(e["id"] != eid for e in client.get("/api/memory/events").json())
        assert client.post(f"/api/memory/events/{eid}/restore").status_code == 200
        assert any(e["id"] == eid for e in client.get("/api/memory/events").json())

    def test_required_fields_400(self, client: TestClient) -> None:
        r = client.post("/api/memory/events", json={"type": "x"})
        assert r.status_code == 400
