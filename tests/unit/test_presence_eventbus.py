"""Unit tests for apps/presence/eventbus.py.

Tests numbered to match the spec in the eventbus-eng mission brief:
1.  Event is frozen, slotted, hashable.
2.  publish() + single subscriber receives event.
3.  Multiple subscribers each receive every event independently (fanout).
4.  Late subscriber does NOT receive events published before subscribe.
5.  Overflow: publish more than queue_maxsize without consumer → oldest dropped.
6.  close() causes subscribers to terminate iteration cleanly.
7.  publish_state() produces a correct state event.
8.  get_default_bus() returns same instance; reset_default_bus() gives a fresh one.
9.  subscriber_count() is accurate before/after subscribe/close.
10. Publishing after close is a no-op.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from apps.presence.eventbus import (
    Event,
    EventBus,
    get_default_bus,
    reset_default_bus,
)


# ── Test 1: Event frozen, slotted, hashable ───────────────────────────────────


def test_event_frozen() -> None:
    ev = Event(type="state", payload={"value": "idle"}, ts=1.0)
    with pytest.raises((AttributeError, TypeError)):
        ev.type = "error"  # type: ignore[misc]


def test_event_slotted() -> None:
    ev = Event(type="state", payload={"value": "idle"}, ts=1.0)
    # Slotted dataclasses have __slots__; no __dict__
    assert not hasattr(ev, "__dict__")


def test_event_hashable() -> None:
    ev1 = Event(type="state", payload={"value": "idle"}, ts=1.0)
    ev2 = Event(type="state", payload={"value": "idle"}, ts=1.0)
    # payload is excluded from __hash__ (dicts are not hashable); hash is
    # based on (type, ts) so same type+ts → same hash.
    assert hash(ev1) == hash(ev2)
    # Can be stored in a set.
    s = {ev1, ev2}
    assert len(s) == 1

    # Different ts → different hash (very likely, though not guaranteed by spec)
    ev3 = Event(type="state", payload={"value": "idle"}, ts=2.0)
    assert ev3 not in s


# ── Test 2: Single subscriber receives published event ────────────────────────


@pytest.mark.asyncio
async def test_single_subscriber_receives_event() -> None:
    bus = EventBus()
    ev = Event(type="state", payload={"value": "listening", "channel": "voice"}, ts=1.0)

    received: list[Event] = []

    async def consumer() -> None:
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(consumer())
    # Yield control so consumer can register its queue.
    await asyncio.sleep(0)

    bus.publish(ev)
    await bus.close()
    await task

    assert received == [ev]


# ── Test 3: Multiple subscribers get every event (fanout) ────────────────────


@pytest.mark.asyncio
async def test_fanout_multiple_subscribers() -> None:
    bus = EventBus()
    ev = Event(type="response_chunk", payload={"text": "hello", "channel": "text"}, ts=2.0)

    received_a: list[Event] = []
    received_b: list[Event] = []

    async def consumer_a() -> None:
        async for event in bus.subscribe():
            received_a.append(event)

    async def consumer_b() -> None:
        async for event in bus.subscribe():
            received_b.append(event)

    task_a = asyncio.create_task(consumer_a())
    task_b = asyncio.create_task(consumer_b())
    await asyncio.sleep(0)

    bus.publish(ev)
    await bus.close()
    await asyncio.gather(task_a, task_b)

    assert received_a == [ev]
    assert received_b == [ev]


# ── Test 4: Late subscriber misses earlier events (no replay) ────────────────


@pytest.mark.asyncio
async def test_late_subscriber_no_replay() -> None:
    bus = EventBus()

    early_ev = Event(type="state", payload={"value": "idle", "channel": "system"}, ts=0.5)
    late_ev = Event(type="state", payload={"value": "thinking", "channel": "voice"}, ts=1.5)

    bus.publish(early_ev)  # published BEFORE any subscriber

    received: list[Event] = []

    async def late_consumer() -> None:
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(late_consumer())
    await asyncio.sleep(0)  # subscriber now registered

    bus.publish(late_ev)  # published AFTER subscriber
    await bus.close()
    await task

    # Only late_ev should be received; early_ev is gone.
    assert received == [late_ev]


# ── Test 5: Overflow drops oldest, keeps newest ──────────────────────────────


@pytest.mark.asyncio
async def test_overflow_drops_oldest() -> None:
    maxsize = 4
    bus = EventBus(queue_maxsize=maxsize)

    # Create a subscriber queue but do NOT consume from it — queue fills up.
    received: list[Event] = []

    async def slow_consumer() -> None:
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(slow_consumer())
    await asyncio.sleep(0)  # subscriber registers

    # Publish maxsize + 2 events without yielding (no consumption).
    events = [
        Event(type="transcript", payload={"text": str(i), "final": False, "channel": "voice"}, ts=float(i))
        for i in range(maxsize + 2)
    ]
    for ev in events:
        bus.publish(ev)

    # Now consume everything by closing and draining.
    await bus.close()
    await task

    # The queue could hold at most `maxsize` items at one time.
    # Oldest (i=0,1) should have been dropped; newest should be present.
    assert len(received) <= maxsize
    # The last two published events must be in received (they are the newest).
    last_two = events[-2:]
    for ev in last_two:
        assert ev in received


# ── Test 6: close() terminates subscriber iteration cleanly ──────────────────


@pytest.mark.asyncio
async def test_close_terminates_subscribers() -> None:
    bus = EventBus()
    finished: list[bool] = []

    async def consumer() -> None:
        async for _ in bus.subscribe():
            pass
        finished.append(True)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    await bus.close()
    await task

    assert finished == [True]


@pytest.mark.asyncio
async def test_close_terminates_multiple_subscribers() -> None:
    bus = EventBus()
    finished_count = 0

    async def consumer() -> None:
        nonlocal finished_count
        async for _ in bus.subscribe():
            pass
        finished_count += 1

    tasks = [asyncio.create_task(consumer()) for _ in range(3)]
    await asyncio.sleep(0)

    await bus.close()
    await asyncio.gather(*tasks)

    assert finished_count == 3


# ── Test 7: publish_state() produces correct state event ────────────────────


@pytest.mark.asyncio
async def test_publish_state_produces_state_event() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def consumer() -> None:
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    before = time.monotonic()
    bus.publish_state("listening", channel="voice")
    after = time.monotonic()

    await bus.close()
    await task

    assert len(received) == 1
    ev = received[0]
    assert ev.type == "state"
    assert ev.payload == {"value": "listening", "channel": "voice"}
    # ts should be a monotonic timestamp between before and after
    assert before <= ev.ts <= after


@pytest.mark.asyncio
async def test_publish_state_default_channel() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def consumer() -> None:
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    bus.publish_state("idle")
    await bus.close()
    await task

    assert received[0].payload["channel"] == "system"


# ── Test 8: get_default_bus() singleton; reset_default_bus() ─────────────────


def test_get_default_bus_returns_same_instance() -> None:
    reset_default_bus()  # ensure clean state
    bus1 = get_default_bus()
    bus2 = get_default_bus()
    assert bus1 is bus2


def test_reset_default_bus_gives_fresh_instance() -> None:
    reset_default_bus()
    bus1 = get_default_bus()
    reset_default_bus()
    bus2 = get_default_bus()
    assert bus1 is not bus2


def test_reset_default_bus_then_get_gives_eventbus() -> None:
    reset_default_bus()
    bus = get_default_bus()
    assert isinstance(bus, EventBus)


# ── Test 9: subscriber_count() accuracy ──────────────────────────────────────


@pytest.mark.asyncio
async def test_subscriber_count_before_after() -> None:
    bus = EventBus()
    assert bus.subscriber_count() == 0

    received: list[Event] = []

    async def consumer() -> None:
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    assert bus.subscriber_count() == 1

    task2 = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    assert bus.subscriber_count() == 2

    await bus.close()
    await asyncio.gather(task, task2)
    # After consumers exit their generators, self._subs should be empty.
    assert bus.subscriber_count() == 0


# ── Test 10: Publishing after close is a no-op ───────────────────────────────


@pytest.mark.asyncio
async def test_publish_after_close_is_noop() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def consumer() -> None:
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    ev_before = Event(type="state", payload={"value": "idle", "channel": "system"}, ts=1.0)
    ev_after = Event(type="error", payload={"message": "oops", "where": "test"}, ts=2.0)

    bus.publish(ev_before)
    await bus.close()
    await task

    # Publishing after close should not raise and should not deliver events.
    bus.publish(ev_after)  # must not raise

    assert received == [ev_before]


@pytest.mark.asyncio
async def test_publish_after_close_no_exception() -> None:
    bus = EventBus()
    await bus.close()
    # Should silently do nothing — no exception.
    ev = Event(type="state", payload={"value": "idle", "channel": "system"}, ts=0.0)
    bus.publish(ev)  # no-op, no raise


@pytest.mark.asyncio
async def test_close_idempotent() -> None:
    bus = EventBus()
    await bus.close()
    await bus.close()  # second close should be harmless


# ── Additional edge-case tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscriber_exits_early_cleans_up() -> None:
    """Subscriber that explicitly closes the async generator removes itself from _subs.

    Note: Python's async generator finalization (on implicit break) is deferred
    to the event loop's shutdown sweep (asyncio.shutdown_asyncgens). To guarantee
    immediate cleanup, the consumer must call aclose() explicitly. This test verifies
    that explicit aclose() triggers the finally-block cleanup path in subscribe().
    """
    bus = EventBus()

    async def one_shot_consumer() -> None:
        gen = bus.subscribe()
        async for _ in gen:
            await gen.aclose()  # explicit close → finally runs → self-removal
            break

    task = asyncio.create_task(one_shot_consumer())
    await asyncio.sleep(0)

    assert bus.subscriber_count() == 1

    bus.publish(Event(type="state", payload={"value": "idle", "channel": "system"}, ts=0.0))
    await task

    assert bus.subscriber_count() == 0


@pytest.mark.asyncio
async def test_publish_all_event_types() -> None:
    """Smoke-test that all EventType literals round-trip through the bus."""
    bus = EventBus()
    all_events = [
        Event("state", {"value": "idle", "channel": "system"}, ts=0.1),
        Event("transcript", {"text": "hi", "final": True, "channel": "voice"}, ts=0.2),
        Event("response_chunk", {"text": "hello", "channel": "text"}, ts=0.3),
        Event("response_end", {"channel": "voice", "episode_id": 42}, ts=0.4),
        Event("memory_recall", {"kind": "fact", "person_name": "엄마"}, ts=0.5),
        Event("error", {"message": "oops", "where": "agent"}, ts=0.6),
    ]

    received: list[Event] = []

    async def consumer() -> None:
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    for ev in all_events:
        bus.publish(ev)

    await bus.close()
    await task

    assert received == all_events
