"""Unit tests for apps.presence.server — FastAPI + WebSocket presence server.

Covers:
  1. GET / returns 200 text/html.
  2. GET /static/missing.css returns 404.
  3. GET /healthz returns {"status": "ok", ...}.
  4. WebSocket: receive hello on connect; receive published event; close cleanly.
  5. Multiple WS connections all receive the same published event (fan-out).
  6. Host validation: _validate_host raises SystemExit for 0.0.0.0.
  7. Static dir override works (custom path for tests).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Stub EventBus — avoids depending on apps.presence.eventbus being importable.
# In practice eventbus-eng has delivered the file, but we decouple the test.
# ---------------------------------------------------------------------------

try:
    from apps.presence.eventbus import Event, EventBus, reset_default_bus
    _HAS_REAL_BUS = True
except ImportError:  # pragma: no cover
    _HAS_REAL_BUS = False


class StubBus:
    """Minimal synchronous stub that mimics EventBus for server tests."""

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[Event | None]] = []  # type: ignore[type-arg]

    def subscriber_count(self) -> int:
        return len(self._queues)

    def publish(self, event: Event) -> None:  # type: ignore[name-defined]
        for q in self._queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def subscribe(self):  # type: ignore[return]
        q: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=256)  # type: ignore[type-arg]
        self._queues.append(q)
        try:
            while True:
                item = await q.get()
                if item is None:
                    return
                yield item
        finally:
            try:
                self._queues.remove(q)
            except ValueError:
                pass

    async def close(self) -> None:
        for q in self._queues:
            await q.put(None)


def _make_event(ev_type: str = "state", payload: dict | None = None) -> "Event":  # type: ignore[name-defined]
    if _HAS_REAL_BUS:
        return Event(type=ev_type, payload=payload or {}, ts=time.monotonic())  # type: ignore[call-arg]
    # Fallback: a simple namespace object that looks like Event.
    from types import SimpleNamespace
    return SimpleNamespace(type=ev_type, payload=payload or {}, ts=time.monotonic())  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from apps.presence.server import create_app  # noqa: E402


def _make_client(
    bus: StubBus | None = None,
    static_dir: Path | None = None,
    *,
    tmp_path: Path | None = None,
) -> TestClient:
    """Return a TestClient with an isolated bus and static dir."""
    if bus is None:
        bus = StubBus()
    if static_dir is None and tmp_path is not None:
        static_dir = tmp_path
    app = create_app(bus=bus, static_dir=static_dir)  # type: ignore[arg-type]
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIndexRoute:
    """GET / — should return 200 text/html."""

    def test_returns_200(self, tmp_path: Path) -> None:
        # Create a minimal index.html in a temp dir.
        (tmp_path / "index.html").write_text("<html><body>orb</body></html>")
        client = _make_client(static_dir=tmp_path)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_content_type_html(self, tmp_path: Path) -> None:
        (tmp_path / "index.html").write_text("<html><body>orb</body></html>")
        client = _make_client(static_dir=tmp_path)
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_fallback_when_no_index(self, tmp_path: Path) -> None:
        """Returns fallback HTML when static dir has no index.html."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        client = _make_client(static_dir=empty_dir)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestStaticRoute:
    """GET /static/* — 404 for missing files."""

    def test_missing_file_returns_404(self, tmp_path: Path) -> None:
        (tmp_path / "index.html").write_text("<html></html>")
        client = _make_client(static_dir=tmp_path)
        resp = client.get("/static/missing.css")
        assert resp.status_code == 404

    def test_existing_file_returns_200(self, tmp_path: Path) -> None:
        (tmp_path / "index.html").write_text("<html></html>")
        (tmp_path / "style.css").write_text("body { margin: 0; }")
        client = _make_client(static_dir=tmp_path)
        resp = client.get("/static/style.css")
        assert resp.status_code == 200


class TestHealthz:
    """GET /healthz — returns status ok with subscriber count."""

    def test_returns_ok(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path=tmp_path)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "subscribers" in body
        assert isinstance(body["subscribers"], int)

    def test_subscriber_count_zero_initially(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path=tmp_path)
        resp = client.get("/healthz")
        assert resp.json()["subscribers"] == 0


class TestWebSocket:
    """WS /ws — hello, event forwarding, clean disconnect."""

    def test_hello_on_connect(self, tmp_path: Path) -> None:
        bus = StubBus()
        app = create_app(bus=bus, static_dir=tmp_path)  # type: ignore[arg-type]
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                msg = ws.receive_text()
                data = json.loads(msg)
                assert data["type"] == "hello"
                assert data["payload"]["schema_version"] == 1

    def test_receive_published_event(self, tmp_path: Path) -> None:
        bus = StubBus()
        event = _make_event("state", {"value": "listening", "channel": "voice"})
        app = create_app(bus=bus, static_dir=tmp_path)  # type: ignore[arg-type]
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # Consume hello.
                ws.receive_text()
                # Publish an event.
                bus.publish(event)
                msg = ws.receive_text()
                data = json.loads(msg)
                assert data["type"] == "state"
                assert data["payload"]["value"] == "listening"

    def test_close_cleanly(self, tmp_path: Path) -> None:
        bus = StubBus()
        app = create_app(bus=bus, static_dir=tmp_path)  # type: ignore[arg-type]
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.receive_text()  # hello
            # No exception means clean disconnect.


class TestFanOut:
    """Multiple WS connections all receive the same published event."""

    def test_two_clients_both_receive(self, tmp_path: Path) -> None:
        bus = StubBus()
        event = _make_event("state", {"value": "thinking", "channel": "text"})
        app = create_app(bus=bus, static_dir=tmp_path)  # type: ignore[arg-type]

        received_a: list[dict] = []  # type: ignore[type-arg]
        received_b: list[dict] = []  # type: ignore[type-arg]

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws_a:
                with client.websocket_connect("/ws") as ws_b:
                    ws_a.receive_text()  # hello
                    ws_b.receive_text()  # hello

                    bus.publish(event)

                    received_a.append(json.loads(ws_a.receive_text()))
                    received_b.append(json.loads(ws_b.receive_text()))

        assert received_a[0]["type"] == "state"
        assert received_b[0]["type"] == "state"
        assert received_a[0]["payload"]["value"] == "thinking"
        assert received_b[0]["payload"]["value"] == "thinking"


class TestHostValidation:
    """_validate_host raises SystemExit for non-loopback addresses."""

    def test_reject_0_0_0_0(self) -> None:
        from apps.presence.__main__ import _validate_host

        with pytest.raises(SystemExit):
            _validate_host("0.0.0.0")

    def test_reject_public_ip(self) -> None:
        from apps.presence.__main__ import _validate_host

        with pytest.raises(SystemExit):
            _validate_host("192.168.1.100")

    def test_allow_127_0_0_1(self) -> None:
        from apps.presence.__main__ import _validate_host

        _validate_host("127.0.0.1")  # Must not raise.

    def test_allow_localhost(self) -> None:
        from apps.presence.__main__ import _validate_host

        _validate_host("localhost")  # Must not raise.

    def test_allow_ipv6_loopback(self) -> None:
        from apps.presence.__main__ import _validate_host

        _validate_host("::1")  # Must not raise.


class TestStaticDirOverride:
    """create_app(static_dir=...) serves from the given directory."""

    def test_custom_dir_serves_file(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "custom_static"
        custom_dir.mkdir()
        (custom_dir / "index.html").write_text("<html><body>custom</body></html>")
        (custom_dir / "app.js").write_text("console.log('orb')")

        app = create_app(bus=StubBus(), static_dir=custom_dir)  # type: ignore[arg-type]
        with TestClient(app) as client:
            resp = client.get("/static/app.js")
            assert resp.status_code == 200
            assert "orb" in resp.text


# ---------------------------------------------------------------------------
# POST /publish — new tests (added by presence-fix-eng)
# ---------------------------------------------------------------------------


class TestPublishEndpoint:
    """POST /publish — remote bus bridge into the local event bus."""

    def test_valid_body_returns_200_and_event_reaches_bus(self, tmp_path: Path) -> None:
        """A well-formed POST reaches the local bus subscribers."""
        bus = StubBus()
        received: list[object] = []

        # We can't await the async generator in a sync test directly, so we
        # verify via bus.publish being intercepted.
        original_publish = bus.publish
        published_events: list[object] = []

        def capturing_publish(event: object) -> None:
            published_events.append(event)
            original_publish(event)  # type: ignore[arg-type]

        bus.publish = capturing_publish  # type: ignore[method-assign]

        client = _make_client(bus=bus, static_dir=tmp_path)
        resp = client.post(
            "/publish",
            json={"type": "state", "payload": {"value": "thinking", "channel": "voice"}, "ts": 1.23},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert len(published_events) == 1
        ev = published_events[0]
        assert getattr(ev, "type") == "state"
        assert getattr(ev, "payload") == {"value": "thinking", "channel": "voice"}

    def test_missing_type_returns_400(self, tmp_path: Path) -> None:
        """Body without 'type' field → 400."""
        client = _make_client(tmp_path=tmp_path)
        resp = client.post(
            "/publish",
            json={"payload": {"value": "idle"}, "ts": 0.0},
        )
        assert resp.status_code == 400
        assert "type" in resp.json()["error"]

    def test_missing_payload_returns_400(self, tmp_path: Path) -> None:
        """Body without 'payload' field → 400."""
        client = _make_client(tmp_path=tmp_path)
        resp = client.post(
            "/publish",
            json={"type": "state", "ts": 0.0},
        )
        assert resp.status_code == 400
        assert "payload" in resp.json()["error"]

    def test_missing_ts_returns_400(self, tmp_path: Path) -> None:
        """Body without 'ts' field → 400."""
        client = _make_client(tmp_path=tmp_path)
        resp = client.post(
            "/publish",
            json={"type": "state", "payload": {"value": "idle"}},
        )
        assert resp.status_code == 400
        assert "ts" in resp.json()["error"]

    def test_non_string_type_returns_400(self, tmp_path: Path) -> None:
        """'type' must be a non-empty string; int → 400."""
        client = _make_client(tmp_path=tmp_path)
        resp = client.post(
            "/publish",
            json={"type": 42, "payload": {}, "ts": 0.0},
        )
        assert resp.status_code == 400

    def test_payload_must_be_dict_returns_400(self, tmp_path: Path) -> None:
        """'payload' must be a JSON object; list → 400."""
        client = _make_client(tmp_path=tmp_path)
        resp = client.post(
            "/publish",
            json={"type": "state", "payload": ["not", "a", "dict"], "ts": 0.0},
        )
        assert resp.status_code == 400

    def test_non_loopback_host_returns_403(self, tmp_path: Path) -> None:
        """A request from a non-loopback host must be rejected with 403.

        Starlette TestClient uses ``request.client.host == "testclient"`` which
        is accepted by _is_loopback.  We verify the rejection path by
        unit-testing _is_loopback directly with a public IP.
        """
        from apps.presence.server import _is_loopback

        assert not _is_loopback("192.168.1.100")
        assert not _is_loopback("10.0.0.1")
        assert not _is_loopback("0.0.0.0")
        # Loopback addresses must be accepted.
        assert _is_loopback("127.0.0.1")
        assert _is_loopback("::1")
        assert _is_loopback("localhost")
        assert _is_loopback("testclient")

    def test_valid_body_ts_as_integer_is_accepted(self, tmp_path: Path) -> None:
        """ts as integer (not float) should also be accepted."""
        client = _make_client(tmp_path=tmp_path)
        resp = client.post(
            "/publish",
            json={"type": "state", "payload": {"value": "idle", "channel": "system"}, "ts": 0},
        )
        assert resp.status_code == 200
