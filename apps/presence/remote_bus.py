"""RemoteEventBus — fire-and-forget HTTP bridge to a remote presence server.

Drop-in replacement for the local EventBus's publish surface so that channel
processes (voice, text) can drive the orb from a separate process by posting
events to ``POST /publish`` on the running presence server.

subscribe() is intentionally NOT supported: only the server-side bus has
subscribers.  Callers that only call publish() / publish_state() work without
any changes.

CLAUDE.md references:
  §2.3  Local-First — server is 127.0.0.1 only; this client always targets
        a loopback URL.
  §6.3  Presence Channel — observer / pub-sub pattern.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog

from apps.presence.eventbus import Event, EventType

log = structlog.get_logger(__name__)


class RemoteEventBus:
    """Publishes events to a remote presence server via HTTP POST.

    Parameters
    ----------
    url:
        Base URL of the presence server, e.g. ``http://127.0.0.1:8765``.
        ``/publish`` is appended automatically.
    timeout_s:
        Per-request HTTP timeout.  Default 1 s — the orb is best-effort; a
        slow / absent server must never stall the assistant.
    """

    def __init__(self, url: str, *, timeout_s: float = 1.0) -> None:
        self._publish_url = url.rstrip("/") + "/publish"
        self._timeout = timeout_s
        self._client = httpx.AsyncClient(timeout=timeout_s)

    # ── publish ───────────────────────────────────────────────────────────────

    def publish(self, event: Event) -> None:
        """Fire-and-forget POST to ``{url}/publish``.

        Never raises — a network failure is logged at DEBUG level so the
        assistant keeps working even when the orb is offline.

        Must be called from an asyncio event loop (either directly or from a
        coroutine running on the loop).  When no running loop is detected the
        call is silently dropped and a debug log is emitted.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop — we're in a bare thread or test context.
            log.debug(
                "remote_bus.publish_no_loop_dropped",
                event_type=event.type,
            )
            return

        loop.create_task(self._post(event))

    def publish_state(self, value: str, *, channel: str = "system") -> None:
        """Convenience wrapper — constructs a ``state`` event and publishes it.

        Equivalent to::

            bus.publish(Event(type="state",
                              payload={"value": value, "channel": channel},
                              ts=time.monotonic()))
        """
        self.publish(
            Event(
                type="state",
                payload={"value": value, "channel": channel},
                ts=time.monotonic(),
            )
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying HTTP client, releasing connections."""
        await self._client.aclose()

    # ── internals ─────────────────────────────────────────────────────────────

    async def _post(self, event: Event) -> None:
        """Perform the actual HTTP POST.  Never raises — all errors are swallowed."""
        body: dict[str, Any] = {
            "type": event.type,
            "payload": event.payload,
            "ts": event.ts,
        }
        try:
            resp = await self._client.post(self._publish_url, json=body)
            if resp.status_code not in (200, 201):
                log.debug(
                    "remote_bus.post_unexpected_status",
                    status=resp.status_code,
                    event_type=event.type,
                )
        except Exception as exc:
            log.debug(
                "remote_bus.post_failed",
                event_type=event.type,
                error=str(exc),
            )
