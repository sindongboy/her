"""Tests for Attachment CRUD in MemoryStore."""

from __future__ import annotations

import sqlite3

import pytest

from apps.memory import Attachment, MemoryStore


def test_add_and_list_attachments(store: MemoryStore) -> None:
    sid = store.add_session(title="with attachment")
    aid = store.add_attachment(
        sid,
        sha256="abc123",
        path="/data/attachments/1/abc123.pdf",
        mime="application/pdf",
        ext="pdf",
        byte_size=1024,
        description="A test PDF",
    )
    attachments = store.list_attachments(sid)
    assert len(attachments) == 1
    a = attachments[0]
    assert a.id == aid
    assert a.session_id == sid
    assert a.sha256 == "abc123"
    assert a.path == "/data/attachments/1/abc123.pdf"
    assert a.mime == "application/pdf"
    assert a.ext == "pdf"
    assert a.byte_size == 1024
    assert a.description == "A test PDF"
    assert a.ingested_at is not None


def test_add_attachment_minimal(store: MemoryStore) -> None:
    sid = store.add_session()
    store.add_attachment(sid, sha256="deadbeef", path="/tmp/file.txt")
    attachments = store.list_attachments(sid)
    assert len(attachments) == 1
    assert attachments[0].mime is None
    assert attachments[0].ext is None
    assert attachments[0].byte_size is None
    assert attachments[0].description is None


def test_find_attachment_by_sha256(store: MemoryStore) -> None:
    sid = store.add_session()
    store.add_attachment(sid, sha256="sha_a", path="/a")
    store.add_attachment(sid, sha256="sha_b", path="/b")

    found = store.find_attachment_by_sha256(sid, "sha_a")
    assert found is not None
    assert found.sha256 == "sha_a"
    assert found.path == "/a"


def test_find_attachment_by_sha256_not_found(store: MemoryStore) -> None:
    sid = store.add_session()
    assert store.find_attachment_by_sha256(sid, "nonexistent") is None


def test_find_attachment_by_sha256_wrong_session(store: MemoryStore) -> None:
    s1 = store.add_session()
    s2 = store.add_session()
    store.add_attachment(s1, sha256="shared_hash", path="/s1/file")
    assert store.find_attachment_by_sha256(s2, "shared_hash") is None


def test_unique_session_sha256_constraint(store: MemoryStore) -> None:
    sid = store.add_session()
    store.add_attachment(sid, sha256="dup_hash", path="/file1")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_attachment(sid, sha256="dup_hash", path="/file2")


def test_same_sha256_different_sessions_allowed(store: MemoryStore) -> None:
    s1 = store.add_session()
    s2 = store.add_session()
    store.add_attachment(s1, sha256="same_hash", path="/s1/file")
    # Should not raise
    store.add_attachment(s2, sha256="same_hash", path="/s2/file")


def test_list_attachments_empty(store: MemoryStore) -> None:
    sid = store.add_session()
    assert store.list_attachments(sid) == []


def test_attachments_cascade_delete_with_session(store: MemoryStore) -> None:
    sid = store.add_session()
    store.add_attachment(sid, sha256="willdelete", path="/tmp/x")
    assert len(store.list_attachments(sid)) == 1

    store.conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    assert store.list_attachments(sid) == []


def test_attachment_is_frozen_dataclass(store: MemoryStore) -> None:
    sid = store.add_session()
    store.add_attachment(sid, sha256="frozen_test", path="/f")
    attachments = store.list_attachments(sid)
    assert isinstance(attachments[0], Attachment)
    with pytest.raises(Exception):
        attachments[0].sha256 = "mutated"  # type: ignore[misc]
