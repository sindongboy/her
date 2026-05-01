"""Tests for Session + Message CRUD in MemoryStore."""

from __future__ import annotations

import pytest

from apps.memory import MemoryStore, Message, Session


def test_add_and_get_session(store: MemoryStore) -> None:
    sid = store.add_session(title="첫 대화")
    s = store.get_session(sid)
    assert s is not None
    assert s.id == sid
    assert s.title == "첫 대화"
    assert s.archived_at is None
    assert s.started_at is not None
    assert s.last_active_at is not None


def test_add_session_with_explicit_started_at(store: MemoryStore) -> None:
    sid = store.add_session(started_at="2026-01-01T00:00:00")
    s = store.get_session(sid)
    assert s is not None
    assert s.started_at == "2026-01-01T00:00:00"


def test_add_session_no_title(store: MemoryStore) -> None:
    sid = store.add_session()
    s = store.get_session(sid)
    assert s is not None
    assert s.title is None
    assert s.summary is None


def test_get_session_not_found(store: MemoryStore) -> None:
    assert store.get_session(9999) is None


def test_list_recent_sessions_descending(store: MemoryStore) -> None:
    a = store.add_session(title="a", started_at="2026-01-01T00:00:00")
    b = store.add_session(title="b", started_at="2026-04-01T00:00:00")
    c = store.add_session(title="c", started_at="2026-04-30T00:00:00")
    sessions = store.list_recent_sessions(limit=10)
    titles = [s.title for s in sessions]
    assert titles == ["c", "b", "a"]
    assert {a, b, c} == {s.id for s in sessions}


def test_list_recent_sessions_limit(store: MemoryStore) -> None:
    for i in range(5):
        store.add_session(title=f"s{i}")
    assert len(store.list_recent_sessions(limit=3)) == 3


def test_set_session_title(store: MemoryStore) -> None:
    sid = store.add_session()
    store.set_session_title(sid, "새 제목")
    assert store.get_session(sid).title == "새 제목"  # type: ignore[union-attr]


def test_set_session_summary(store: MemoryStore) -> None:
    sid = store.add_session()
    store.set_session_summary(sid, "요약 변경")
    assert store.get_session(sid).summary == "요약 변경"  # type: ignore[union-attr]


def test_archive_session_excludes_from_default_list(store: MemoryStore) -> None:
    active = store.add_session(title="활성")
    archived = store.add_session(title="아카이브")
    store.archive_session(archived)

    active_ids = {s.id for s in store.list_recent_sessions()}
    assert active in active_ids
    assert archived not in active_ids

    all_ids = {s.id for s in store.list_recent_sessions(include_archived=True)}
    assert {active, archived}.issubset(all_ids)


def test_session_is_frozen(store: MemoryStore) -> None:
    sid = store.add_session()
    s = store.get_session(sid)
    assert isinstance(s, Session)
    with pytest.raises(Exception):
        s.title = "mutated"  # type: ignore[misc]


# ── messages ──────────────────────────────────────────────────────────────


def test_add_message_and_list(store: MemoryStore) -> None:
    sid = store.add_session()
    store.add_message(sid, "user", "안녕")
    store.add_message(sid, "assistant", "네, 안녕하세요!")
    msgs = store.list_messages(sid)
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[0].content == "안녕"
    assert msgs[1].role == "assistant"


def test_add_message_invalid_role_raises(store: MemoryStore) -> None:
    sid = store.add_session()
    with pytest.raises(ValueError):
        store.add_message(sid, "ghost", "x")


def test_message_bumps_session_last_active(store: MemoryStore) -> None:
    sid = store.add_session(started_at="2020-01-01T00:00:00")
    store.add_message(sid, "user", "hi")
    s = store.get_session(sid)
    assert s is not None
    # last_active_at should now be CURRENT_TIMESTAMP, not 2020-01-01
    assert not s.last_active_at.startswith("2020")


def test_recall_messages_returns_chronological_tail(store: MemoryStore) -> None:
    sid = store.add_session()
    for i in range(5):
        store.add_message(sid, "user", f"msg-{i}")
    msgs = store.recall_messages(sid, limit=3)
    assert len(msgs) == 3
    assert [m.content for m in msgs] == ["msg-2", "msg-3", "msg-4"]


def test_message_cascades_on_session_delete(store: MemoryStore) -> None:
    sid = store.add_session()
    store.add_message(sid, "user", "x")
    store.conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    assert store.list_messages(sid) == []


def test_message_is_frozen(store: MemoryStore) -> None:
    sid = store.add_session()
    mid = store.add_message(sid, "user", "x")
    msgs = store.list_messages(sid)
    assert msgs[0].id == mid
    assert isinstance(msgs[0], Message)
    with pytest.raises(Exception):
        msgs[0].content = "y"  # type: ignore[misc]
