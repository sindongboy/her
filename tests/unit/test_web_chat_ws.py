"""WebSocket /ws/chat protocol tests with a fake agent."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.memory.store import MemoryStore
from apps.web.eventbus import EventBus
from apps.web.server import create_app


class FakeGemini:
    def embed(self, text: str, *, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        return [0.0] * 768


class FakeAgent:
    """Mimics enough of AgentCore for the WS handler."""

    def __init__(self, store: MemoryStore, *, chunks: list[str]) -> None:
        self._store = store
        self._chunks = chunks
        self._gemini = FakeGemini()
        self.calls: list[dict[str, Any]] = []

    async def stream_respond(  # type: ignore[override]
        self, message: str, *, session_id: int | None = None, **_: Any
    ):
        # If session_id is None the server should have allocated one already.
        assert session_id is not None
        self.calls.append({"message": message, "session_id": session_id})
        # Persist user + assistant messages so /api/sessions/{id}/messages reflects
        # what really happened.
        self._store.add_message(session_id, "user", message)
        full = ""
        for chunk in self._chunks:
            full += chunk
            yield chunk
        self._store.add_message(session_id, "assistant", full)


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(tmp_path / "ws.db")
    try:
        yield s
    finally:
        s.close()


def _build_client(store: MemoryStore, chunks: list[str]) -> TestClient:
    agent = FakeAgent(store, chunks=chunks)
    app = create_app(agent=agent, store=store, bus=EventBus())
    return TestClient(app)


def _frames(ws: Any, until_type: str) -> list[dict]:
    """Pull frames from the WS until a frame with ``type==until_type`` arrives."""
    out: list[dict] = []
    while True:
        text = ws.receive_text()
        msg = json.loads(text)
        out.append(msg)
        if msg.get("type") == until_type:
            return out


def test_hello_on_connect(store: MemoryStore) -> None:
    client = _build_client(store, chunks=[])
    with client.websocket_connect("/ws/chat") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello"
        assert isinstance(hello["schema_version"], int)


def test_full_turn_streams_tokens_and_done(store: MemoryStore) -> None:
    client = _build_client(store, chunks=["안녕", "하세요", "!"])
    with client.websocket_connect("/ws/chat") as ws:
        json.loads(ws.receive_text())  # hello
        ws.send_text(json.dumps({"type": "message", "content": "테스트"}))

        frames = _frames(ws, until_type="done")

    types = [f["type"] for f in frames]
    assert "recall" in types
    tokens = [f["text"] for f in frames if f["type"] == "token"]
    assert tokens == ["안녕", "하세요", "!"]
    assert frames[-1]["type"] == "done"


def test_invalid_json_yields_error(store: MemoryStore) -> None:
    client = _build_client(store, chunks=["x"])
    with client.websocket_connect("/ws/chat") as ws:
        json.loads(ws.receive_text())
        ws.send_text("not-json")
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "JSON" in msg["message"]


def test_missing_content_yields_error(store: MemoryStore) -> None:
    client = _build_client(store, chunks=["x"])
    with client.websocket_connect("/ws/chat") as ws:
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "message", "content": "   "}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"


def test_unknown_session_id_yields_error(store: MemoryStore) -> None:
    client = _build_client(store, chunks=["x"])
    with client.websocket_connect("/ws/chat") as ws:
        json.loads(ws.receive_text())
        ws.send_text(
            json.dumps({"type": "message", "content": "hi", "session_id": 9999})
        )
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"


def test_creates_session_when_none_provided(store: MemoryStore) -> None:
    client = _build_client(store, chunks=["a", "b"])
    with client.websocket_connect("/ws/chat") as ws:
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "message", "content": "hi"}))
        frames = _frames(ws, until_type="done")
    sid = frames[-1]["session_id"]
    assert sid == 1
    sessions = store.list_recent_sessions()
    assert any(s.id == sid for s in sessions)


def test_reuses_provided_session_id(store: MemoryStore) -> None:
    sid = store.add_session()
    client = _build_client(store, chunks=["x"])
    with client.websocket_connect("/ws/chat") as ws:
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "message", "content": "hi", "session_id": sid}))
        frames = _frames(ws, until_type="done")
    assert frames[-1]["session_id"] == sid
    # Only one session in DB.
    assert len(store.list_recent_sessions()) == 1
