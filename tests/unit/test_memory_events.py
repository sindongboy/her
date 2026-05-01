"""Tests for Event CRUD in MemoryStore."""

from __future__ import annotations

import pytest

from apps.memory import Event, MemoryStore


def test_add_and_get_event(store: MemoryStore) -> None:
    eid = store.add_event("birthday", "Mom birthday", "2026-05-01T10:00:00")
    ev = store.get_event(eid)
    assert ev is not None
    assert ev.id == eid
    assert ev.type == "birthday"
    assert ev.title == "Mom birthday"
    assert ev.when_at == "2026-05-01T10:00:00"
    assert ev.status == "pending"
    assert ev.person_id is None
    assert ev.recurrence is None
    assert ev.source is None


def test_add_event_with_person_and_options(store: MemoryStore) -> None:
    pid = store.add_person("Alice", relation="mother")
    eid = store.add_event(
        "reminder",
        "Doctor appointment",
        "2026-06-01T09:00:00",
        person_id=pid,
        recurrence="FREQ=YEARLY",
        source="calendar",
    )
    ev = store.get_event(eid)
    assert ev is not None
    assert ev.person_id == pid
    assert ev.recurrence == "FREQ=YEARLY"
    assert ev.source == "calendar"


def test_get_event_not_found(store: MemoryStore) -> None:
    assert store.get_event(9999) is None


def test_list_upcoming_events(store: MemoryStore) -> None:
    # Events within the next 24h from a known "now"
    now = "2026-05-01T00:00:00"
    store.add_event("a", "Soon", "2026-05-01T12:00:00")
    store.add_event("b", "Also soon", "2026-05-01T23:00:00")
    store.add_event("c", "Too far", "2026-05-03T00:00:00")
    store.add_event("d", "Already done", "2026-05-01T06:00:00", )

    events = store.list_upcoming_events(within_hours=24, now_iso=now)
    titles = [e.title for e in events]
    assert "Soon" in titles
    assert "Also soon" in titles
    assert "Too far" not in titles


def test_list_upcoming_events_only_pending(store: MemoryStore) -> None:
    now = "2026-05-01T00:00:00"
    eid = store.add_event("task", "Pending task", "2026-05-01T12:00:00")
    store.set_event_status(eid, "done")

    events = store.list_upcoming_events(within_hours=24, now_iso=now)
    assert all(e.status == "pending" for e in events)


def test_list_upcoming_events_ordered_by_when_at(store: MemoryStore) -> None:
    now = "2026-05-01T00:00:00"
    store.add_event("x", "Later", "2026-05-01T20:00:00")
    store.add_event("y", "Earlier", "2026-05-01T08:00:00")

    events = store.list_upcoming_events(within_hours=24, now_iso=now)
    when_ats = [e.when_at for e in events]
    assert when_ats == sorted(when_ats)


def test_set_event_status_transitions(store: MemoryStore) -> None:
    eid = store.add_event("task", "My task", "2026-05-01T10:00:00")
    store.set_event_status(eid, "done")
    ev = store.get_event(eid)
    assert ev is not None
    assert ev.status == "done"

    store.set_event_status(eid, "cancelled")
    ev = store.get_event(eid)
    assert ev is not None
    assert ev.status == "cancelled"


def test_set_event_status_invalid_raises(store: MemoryStore) -> None:
    eid = store.add_event("task", "Task", "2026-05-01T10:00:00")
    with pytest.raises(ValueError, match="status must be one of"):
        store.set_event_status(eid, "unknown")


def test_event_cascade_set_null_on_person_delete(store: MemoryStore) -> None:
    pid = store.add_person("Bob")
    eid = store.add_event("b", "Bob event", "2026-05-01T10:00:00", person_id=pid)
    ev_before = store.get_event(eid)
    assert ev_before is not None
    assert ev_before.person_id == pid

    store.conn.execute("DELETE FROM people WHERE id = ?", (pid,))

    ev_after = store.get_event(eid)
    assert ev_after is not None
    assert ev_after.person_id is None


def test_event_is_frozen_dataclass(store: MemoryStore) -> None:
    eid = store.add_event("t", "T", "2026-05-01T10:00:00")
    ev = store.get_event(eid)
    assert isinstance(ev, Event)
    with pytest.raises(Exception):
        ev.title = "mutated"  # type: ignore[misc]
