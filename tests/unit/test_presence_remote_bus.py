"""Unit tests for apps.presence.remote_bus.RemoteEventBus.

Mocks httpx.AsyncClient so no real HTTP calls are made.

Covers:
  1. publish() inside asyncio loop → schedules POST with correct body.
  2. publish() swallows network errors; never raises.
  3. publish_state() constructs the right event and POSTs.
  4. close() calls aclose() on the underlying AsyncClient.
  5. publish() with no running loop → silent drop (no raise).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.presence.eventbus import Event
from apps.presence.remote_bus import RemoteEventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(ev_type: str = "state", payload: dict | None = None) -> Event:
    return Event(type=ev_type, payload=payload or {}, ts=time.monotonic())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. publish() inside asyncio loop → schedules POST with correct body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_schedules_post_with_correct_body() -> None:
    """publish() from inside a running loop creates a task that POSTs the event."""
    posted_bodies: list[dict] = []  # type: ignore[type-arg]

    async def fake_post(url: str, *, json: dict) -> MagicMock:  # type: ignore[type-arg]
        posted_bodies.append(json)
        resp = MagicMock()
        resp.status_code = 200
        return resp

    bus = RemoteEventBus("http://127.0.0.1:8765")
    # Replace the client's post method with a coroutine mock.
    bus._client.post = fake_post  # type: ignore[method-assign]

    event = _make_event("state", {"value": "listening", "channel": "voice"})
    bus.publish(event)

    # Drain pending tasks so the POST coroutine runs.
    await asyncio.sleep(0)

    assert len(posted_bodies) == 1
    body = posted_bodies[0]
    assert body["type"] == "state"
    assert body["payload"] == {"value": "listening", "channel": "voice"}
    assert isinstance(body["ts"], float)
    assert body["ts"] == pytest.approx(event.ts)

    await bus.close()


# ---------------------------------------------------------------------------
# 2. publish() swallows network errors; never raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_swallows_network_errors() -> None:
    """publish() must not raise even when the server is unreachable."""

    async def failing_post(url: str, *, json: dict) -> None:  # type: ignore[type-arg]
        raise httpx_connect_error()

    def httpx_connect_error() -> Exception:
        import httpx
        return httpx.ConnectError("connection refused")

    bus = RemoteEventBus("http://127.0.0.1:8765")
    bus._client.post = failing_post  # type: ignore[method-assign]

    event = _make_event("error", {"message": "oops", "where": "test"})

    # publish() itself must not raise.
    bus.publish(event)

    # Drain — _post must swallow the error.
    await asyncio.sleep(0)

    await bus.close()
    # Reaching here means no exception was propagated.


# ---------------------------------------------------------------------------
# 3. publish_state() constructs the right event and POSTs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_state_posts_correct_event() -> None:
    """publish_state() creates a 'state' event and fires the HTTP POST."""
    posted_bodies: list[dict] = []  # type: ignore[type-arg]

    async def fake_post(url: str, *, json: dict) -> MagicMock:  # type: ignore[type-arg]
        posted_bodies.append(json)
        resp = MagicMock()
        resp.status_code = 200
        return resp

    bus = RemoteEventBus("http://127.0.0.1:8765")
    bus._client.post = fake_post  # type: ignore[method-assign]

    bus.publish_state("thinking", channel="text")
    await asyncio.sleep(0)

    assert len(posted_bodies) == 1
    body = posted_bodies[0]
    assert body["type"] == "state"
    assert body["payload"]["value"] == "thinking"
    assert body["payload"]["channel"] == "text"
    assert isinstance(body["ts"], float)

    await bus.close()


# ---------------------------------------------------------------------------
# 4. close() calls aclose() on the underlying AsyncClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_calls_aclose() -> None:
    """close() must call aclose() on the httpx.AsyncClient."""
    bus = RemoteEventBus("http://127.0.0.1:8765")
    aclose_mock = AsyncMock()
    bus._client.aclose = aclose_mock  # type: ignore[method-assign]

    await bus.close()

    aclose_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# 5. publish() with no running loop → silent drop (no raise)
# ---------------------------------------------------------------------------


def test_publish_no_running_loop_does_not_raise() -> None:
    """publish() called from a plain thread (no loop) must silently drop the event."""
    bus = RemoteEventBus("http://127.0.0.1:8765")
    event = _make_event("state", {"value": "idle", "channel": "system"})
    # This runs outside any event loop — asyncio.get_running_loop() raises RuntimeError.
    bus.publish(event)  # Must not raise.
    # No cleanup needed; no async client was used.
