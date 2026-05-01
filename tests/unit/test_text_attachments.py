"""Unit tests for AttachmentHandler.

Tests use either the real MemoryStore (via conftest `store` fixture) or a
lightweight FakeStore.  We prefer the real store to avoid divergence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from apps.channels.text.attachments import (
    ALLOWED_EXTS,
    MAX_BYTES,
    AttachmentError,
    AttachmentHandler,
    IngestedAttachment,
)
from apps.memory.store import MemoryStore


# ── helpers ──────────────────────────────────────────────────────────────────


def _write(tmp_path: Path, name: str, content: bytes = b"hello") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def attachments_dir(tmp_path: Path) -> Path:
    d = tmp_path / "attachments"
    d.mkdir()
    return d


@pytest.fixture
def handler(store: MemoryStore, attachments_dir: Path) -> AttachmentHandler:
    return AttachmentHandler(store, attachments_dir)


# ── happy-path tests ──────────────────────────────────────────────────────────


def test_ingest_txt_returns_ingested_attachment(
    tmp_path: Path, store: MemoryStore, handler: AttachmentHandler
) -> None:
    episode_id = store.add_episode(primary_channel="text")
    src = _write(tmp_path, "note.txt", b"hello world")

    result = handler.ingest(episode_id, src)

    assert isinstance(result, IngestedAttachment)
    assert result.ext == ".txt"
    assert result.byte_size == len(b"hello world")
    assert result.sha256 == _sha256(b"hello world")
    assert result.stored_path.exists()
    assert result.original_name == "note.txt"


def test_ingest_pdf(
    tmp_path: Path, store: MemoryStore, handler: AttachmentHandler
) -> None:
    episode_id = store.add_episode(primary_channel="text")
    src = _write(tmp_path, "doc.pdf", b"%PDF-1.4 fake")

    result = handler.ingest(episode_id, src)

    assert result.ext == ".pdf"
    assert result.stored_path.exists()


def test_ingest_records_in_db(
    tmp_path: Path, store: MemoryStore, handler: AttachmentHandler
) -> None:
    episode_id = store.add_episode(primary_channel="text")
    src = _write(tmp_path, "readme.md", b"# Hi")

    result = handler.ingest(episode_id, src, description="test desc")

    attachments = store.list_attachments(episode_id)
    assert len(attachments) == 1
    att = attachments[0]
    assert att.sha256 == result.sha256
    assert att.description == "test desc"
    assert att.ext == ".md"
    assert att.episode_id == episode_id


# ── deduplication ─────────────────────────────────────────────────────────────


def test_ingest_same_file_twice_same_episode_dedupes_stored_file(
    tmp_path: Path, store: MemoryStore, handler: AttachmentHandler
) -> None:
    """Same content ingested twice in same episode: file stored once, DB row once.

    The schema enforces UNIQUE(episode_id, sha256), so the second ingest is a
    no-op at the DB level.  The file on disk is also not re-copied.
    """
    episode_id = store.add_episode(primary_channel="text")
    content = b"duplicate content"
    src = _write(tmp_path, "dup.txt", content)

    r1 = handler.ingest(episode_id, src)
    r2 = handler.ingest(episode_id, src)

    # Same stored path (deduped on disk)
    assert r1.stored_path == r2.stored_path

    # DB has exactly one row (deduped by UNIQUE constraint logic)
    attachments = store.list_attachments(episode_id)
    assert len(attachments) == 1
    assert attachments[0].sha256 == _sha256(content)


def test_ingest_same_content_different_episode_creates_two_rows(
    tmp_path: Path, store: MemoryStore, handler: AttachmentHandler
) -> None:
    """Same file in two different episodes → separate DB rows."""
    ep1 = store.add_episode(primary_channel="text")
    ep2 = store.add_episode(primary_channel="text")
    src = _write(tmp_path, "shared.txt", b"shared")

    handler.ingest(ep1, src)
    handler.ingest(ep2, src)

    assert len(store.list_attachments(ep1)) == 1
    assert len(store.list_attachments(ep2)) == 1


# ── rejection tests ───────────────────────────────────────────────────────────


def test_reject_exe_extension(
    tmp_path: Path, store: MemoryStore, handler: AttachmentHandler
) -> None:
    episode_id = store.add_episode(primary_channel="text")
    src = _write(tmp_path, "virus.exe", b"MZ")

    with pytest.raises(AttachmentError, match="ext_not_allowed"):
        handler.ingest(episode_id, src)


def test_reject_zip_extension(
    tmp_path: Path, store: MemoryStore, handler: AttachmentHandler
) -> None:
    episode_id = store.add_episode(primary_channel="text")
    src = _write(tmp_path, "archive.zip", b"PK")

    with pytest.raises(AttachmentError, match="ext_not_allowed"):
        handler.ingest(episode_id, src)


def test_reject_oversized_file(
    tmp_path: Path, store: MemoryStore, handler: AttachmentHandler
) -> None:
    episode_id = store.add_episode(primary_channel="text")
    # Create a file slightly over MAX_BYTES using seek-and-write (no real data)
    big = tmp_path / "big.txt"
    with big.open("wb") as f:
        f.seek(MAX_BYTES)  # seek past limit
        f.write(b"\x00")   # write one byte → file size = MAX_BYTES + 1

    with pytest.raises(AttachmentError, match="too_large"):
        handler.ingest(episode_id, big)


def test_reject_missing_file(
    tmp_path: Path, store: MemoryStore, handler: AttachmentHandler
) -> None:
    episode_id = store.add_episode(primary_channel="text")
    missing = tmp_path / "ghost.txt"

    with pytest.raises(AttachmentError, match="file_not_found"):
        handler.ingest(episode_id, missing)


# ── allowed extensions sanity ─────────────────────────────────────────────────


def test_all_allowed_exts_accepted(
    tmp_path: Path, store: MemoryStore, handler: AttachmentHandler
) -> None:
    episode_id = store.add_episode(primary_channel="text")
    for ext in sorted(ALLOWED_EXTS):
        src = _write(tmp_path, f"file{ext}", b"data")
        result = handler.ingest(episode_id, src)
        assert result.ext == ext
