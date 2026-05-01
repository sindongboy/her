"""Async fan-out pub-sub event bus for the Presence Channel.

All channels (voice, text, agent) publish state events here; the presence
WebSocket server and any other subscribers consume them.

CLAUDE.md references:
  §4    Architecture — Agent Core / channels publish; presence server subscribes.
  §6.3  Presence Channel — Samantha-style abstract orb.

Concurrency model:
  - Designed for a single asyncio event loop. NOT thread-safe.
  - publish() is synchronous and never blocks.
  - subscribe() is an async generator — yields Events until the bus is closed.
  - Multiple concurrent subscribers are supported (fan-out).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import structlog

log = structlog.get_logger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

EventType = Literal[
    "state",           # one of: idle, listening, thinking, speaking, quiet, wake, sleep
    "transcript",      # user STT partial or final
    "response_chunk",  # agent text chunk (streaming)
    "response_end",    # agent stream complete
    "memory_recall",   # memory hit (optional, for ambient effects)
    "error",           # something went wrong
]

# Payload schemas (informal — kept as constants for client/server alignment):
#
# state:          {"value": "listening"|"thinking"|"speaking"|"idle"|"quiet"|"wake"|"sleep",
#                  "channel": "voice"|"text"|"system"}
# transcript:     {"text": str, "final": bool, "channel": "voice"|"text"}
# response_chunk: {"text": str, "channel": "voice"|"text"}
# response_end:   {"channel": "voice"|"text", "episode_id": int|None}
# memory_recall:  {"kind": "fact"|"episode"|"event"|"attachment", "person_name": str|None}
# error:          {"message": str, "where": str}

PAYLOAD_SCHEMA_DOCS: dict[str, dict[str, object]] = {
    "state": {"value": "str", "channel": "str"},
    "transcript": {"text": "str", "final": "bool", "channel": "str"},
    "response_chunk": {"text": "str", "channel": "str"},
    "response_end": {"channel": "str", "episode_id": "int|None"},
    "memory_recall": {"kind": "str", "person_name": "str|None"},
    "error": {"message": "str", "where": "str"},
}

# ── Event dataclass ───────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class Event:
    """An immutable, hashable event published to the bus.

    Attributes
    ----------
    type:
        One of the EventType literals above.
    payload:
        JSON-serializable dict; schema depends on *type* (see PAYLOAD_SCHEMA_DOCS).
    ts:
        ``time.monotonic()`` at publish time. Useful for debouncing on the client.
        NOT a wall-clock timestamp.
    """

    type: EventType
    # payload is excluded from __hash__ because dicts are not hashable.
    # Identity is determined by (type, ts) which is sufficient for deduplication
    # and set membership checks in practice.
    payload: dict = field(hash=False)  # type: ignore[type-arg]
    ts: float


# ── EventBus ──────────────────────────────────────────────────────────────────


import asyncio  # noqa: E402 — placed after dataclass to keep type block clean


class EventBus:
    """Async fan-out pub-sub bus.

    Every subscriber receives every event independently (no shared offset).

    Concurrency
    -----------
    - Designed for a single asyncio event loop. NOT thread-safe.
    - ``publish()`` is synchronous and never blocks; it puts to each
      subscriber's bounded ``asyncio.Queue``. On overflow, the oldest item is
      dropped and a structlog warning is emitted.
    - ``subscribe()`` is an async generator that yields ``Event`` objects until
      ``close()`` is called or the generator is garbage-collected.
    - Multiple concurrent subscribers are fine.

    Lifecycle
    ---------
    - Use the module-level singleton via ``get_default_bus()`` in production.
    - Tests should construct their own ``EventBus()`` (or call
      ``reset_default_bus()`` to get a fresh singleton).
    """

    def __init__(self, *, queue_maxsize: int = 256) -> None:
        self._queue_maxsize = queue_maxsize
        self._subs: list[asyncio.Queue[Event | None]] = []
        self._closed = False

    # ── publish ───────────────────────────────────────────────────────────────

    def publish(self, event: Event) -> None:
        """Fan the event out to all current subscriber queues.

        Never blocks. On queue overflow, drops the oldest item and logs a
        warning at the *warn* level so operators can tune queue_maxsize.
        """
        if self._closed:
            log.debug("eventbus.publish_after_close_noop", event_type=event.type)
            return

        for q in self._subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest, insert newest.
                try:
                    dropped = q.get_nowait()
                except asyncio.QueueEmpty:
                    dropped = None
                q.put_nowait(event)
                log.warning(
                    "eventbus.subscriber_overflow_dropped_oldest",
                    dropped_event_type=dropped.type if dropped is not None else None,
                )

    def publish_state(self, value: str, *, channel: str = "system") -> None:
        """Convenience wrapper for the most common event type.

        Example::

            bus.publish_state("listening", channel="voice")
        """
        self.publish(
            Event(
                type="state",
                payload={"value": value, "channel": channel},
                ts=time.monotonic(),
            )
        )

    # ── subscribe ─────────────────────────────────────────────────────────────

    async def subscribe(self) -> AsyncIterator[Event]:  # type: ignore[override]
        """Async generator — yields every ``Event`` published after subscription.

        Late subscribers do NOT receive events published before they subscribed
        (no replay buffer in v1).

        Usage::

            async for event in bus.subscribe():
                handle(event)
        """
        q: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subs.append(q)
        try:
            while True:
                item = await q.get()
                if item is None:
                    # Sentinel from close() — stop iteration.
                    return
                yield item
        finally:
            # Clean up even if the consumer breaks out of the loop early.
            try:
                self._subs.remove(q)
            except ValueError:
                pass  # already removed (e.g. close() cleared the list)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def subscriber_count(self) -> int:
        """Return the number of active subscribers."""
        return len(self._subs)

    async def close(self) -> None:
        """Signal all subscribers to stop iteration and mark bus as closed.

        After ``close()``, ``publish()`` becomes a no-op. Future ``subscribe()``
        calls will yield nothing (the generator would receive the sentinel
        immediately on first ``get()`` — or just never get any events since
        publish is a no-op). Existing subscribers receive the ``None`` sentinel
        and exit cleanly.
        """
        if self._closed:
            return
        self._closed = True
        for q in list(self._subs):
            await q.put(None)
        # Don't clear _subs here — subscribers will self-remove via `finally`.
        log.info("eventbus.closed", subscriber_count=len(self._subs))


# ── Module-level singleton ────────────────────────────────────────────────────

_default_bus: EventBus | None = None


def get_default_bus() -> EventBus:
    """Return (creating if needed) the module-level default EventBus.

    Idempotent — repeated calls return the same instance.
    Use ``reset_default_bus()`` in tests to start with a clean bus.
    """
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def reset_default_bus() -> None:
    """Replace the module-level default bus with a fresh instance.

    Intended for tests only. Does NOT close the old bus first — call
    ``await old_bus.close()`` before resetting if subscribers are running.
    """
    global _default_bus
    _default_bus = None
