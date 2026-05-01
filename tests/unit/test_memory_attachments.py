"""Tests for Attachment CRUD in MemoryStore."""

from __future__ import annotations

import sqlite3

import pytest

from apps.memory import Attachment, MemoryStore


def test_add_and_list_attachments(store: MemoryStore) -> None:
    eid = store.add_episode("with attachment", primary_channel="text")
    aid = store.add_attachment(
        eid,
        sha256="abc123",
        path="/data/attachments/1/abc123.pdf",
        mime="application/pdf",
        ext="pdf",
        byte_size=1024,
        description="A test PDF",
    )
    attachments = store.list_attachments(eid)
    assert len(attachments) == 1
    a = attachments[0]
    assert a.id == aid
    assert a.episode_id == eid
    assert a.sha256 == "abc123"
    assert a.path == "/data/attachments/1/abc123.pdf"
    assert a.mime == "application/pdf"
    assert a.ext == "pdf"
    assert a.byte_size == 1024
    assert a.description == "A test PDF"
    assert a.ingested_at is not None


def test_add_attachment_minimal(store: MemoryStore) -> None:
    eid = store.add_episode(primary_channel="text")
    aid = store.add_attachment(eid, sha256="deadbeef", path="/tmp/file.txt")
    attachments = store.list_attachments(eid)
    assert len(attachments) == 1
    assert attachments[0].mime is None
    assert attachments[0].ext is None
    assert attachments[0].byte_size is None
    assert attachments[0].description is None


def test_find_attachment_by_sha256(store: MemoryStore) -> None:
    eid = store.add_episode(primary_channel="text")
    store.add_attachment(eid, sha256="sha_a", path="/a")
    store.add_attachment(eid, sha256="sha_b", path="/b")

    found = store.find_attachment_by_sha256(eid, "sha_a")
    assert found is not None
    assert found.sha256 == "sha_a"
    assert found.path == "/a"


def test_find_attachment_by_sha256_not_found(store: MemoryStore) -> None:
    eid = store.add_episode(primary_channel="text")
    assert store.find_attachment_by_sha256(eid, "nonexistent") is None


def test_find_attachment_by_sha256_wrong_episode(store: MemoryStore) -> None:
    eid1 = store.add_episode(primary_channel="text")
    eid2 = store.add_episode(primary_channel="text")
    store.add_attachment(eid1, sha256="shared_hash", path="/ep1/file")

    # Same sha256 but different episode_id => not found
    result = store.find_attachment_by_sha256(eid2, "shared_hash")
    assert result is None


def test_unique_episode_sha256_constraint(store: MemoryStore) -> None:
    eid = store.add_episode(primary_channel="text")
    store.add_attachment(eid, sha256="dup_hash", path="/file1")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_attachment(eid, sha256="dup_hash", path="/file2")


def test_same_sha256_different_episodes_allowed(store: MemoryStore) -> None:
    eid1 = store.add_episode(primary_channel="text")
    eid2 = store.add_episode(primary_channel="text")
    store.add_attachment(eid1, sha256="same_hash", path="/ep1/file")
    # Should not raise
    store.add_attachment(eid2, sha256="same_hash", path="/ep2/file")


def test_list_attachments_empty(store: MemoryStore) -> None:
    eid = store.add_episode(primary_channel="text")
    assert store.list_attachments(eid) == []


def test_attachments_cascade_delete_with_episode(store: MemoryStore) -> None:
    eid = store.add_episode(primary_channel="text")
    store.add_attachment(eid, sha256="willdelete", path="/tmp/x")
    assert len(store.list_attachments(eid)) == 1

    store.conn.execute("DELETE FROM episodes WHERE id = ?", (eid,))
    assert store.list_attachments(eid) == []


def test_attachment_is_frozen_dataclass(store: MemoryStore) -> None:
    eid = store.add_episode(primary_channel="text")
    store.add_attachment(eid, sha256="frozen_test", path="/f")
    attachments = store.list_attachments(eid)
    assert isinstance(attachments[0], Attachment)
    with pytest.raises(Exception):
        attachments[0].sha256 = "mutated"  # type: ignore[misc]
