"""Attachment ingestion for the text channel.

Validates, deduplicates, stores, and persists attachment metadata.
Per CLAUDE.md §6.2: only whitelisted extensions, max 25 MB.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path

import structlog

from apps.memory.store import MemoryStore

logger = structlog.get_logger(__name__)

# Whitelisted extensions per CLAUDE.md §6.2
ALLOWED_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".pdf", ".txt", ".md", ".ics", ".eml"}
)

# 25 MB per CLAUDE.md §6.2
MAX_BYTES: int = 25 * 1024 * 1024

_CHUNK_SIZE = 64 * 1024  # 64 KB chunks for sha256 streaming read


class AttachmentError(Exception):
    """Raised when an attachment cannot be ingested."""


@dataclass(slots=True, frozen=True)
class IngestedAttachment:
    sha256: str
    mime: str | None
    ext: str
    byte_size: int
    stored_path: Path
    original_name: str


class AttachmentHandler:
    """Validates, stores, and records attachments for a given episode."""

    def __init__(self, store: MemoryStore, attachments_dir: Path) -> None:
        self._store = store
        self._attachments_dir = attachments_dir

    def ingest(
        self,
        episode_id: int,
        source_path: Path,
        *,
        description: str | None = None,
    ) -> IngestedAttachment:
        """Validate, copy, deduplicate, and persist an attachment.

        Args:
            episode_id: The episode this attachment belongs to.
            source_path: Absolute path to the source file.
            description: Optional human-readable description.

        Returns:
            IngestedAttachment with metadata about the stored file.

        Raises:
            AttachmentError: For missing files, disallowed extensions, or size violations.
        """
        # 1) File must exist and be a regular file
        if not source_path.exists() or not source_path.is_file():
            raise AttachmentError(f"file_not_found: {source_path}")

        # 2) Extension check (lowercase, must be in whitelist)
        ext = source_path.suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise AttachmentError(
                f"ext_not_allowed: '{ext}' is not in the allowed list "
                f"{sorted(ALLOWED_EXTS)}"
            )

        # 3) Size check
        byte_size = source_path.stat().st_size
        if byte_size > MAX_BYTES:
            raise AttachmentError(
                f"too_large: {byte_size} bytes exceeds limit of {MAX_BYTES} bytes "
                f"({MAX_BYTES // (1024 * 1024)} MB)"
            )

        # 4) Compute sha256 via streamed read (avoid loading full file into memory)
        sha256_hex = _sha256_file(source_path)

        # 5) MIME detection — re-confirm against extension
        guessed_mime, _ = mimetypes.guess_type(str(source_path))
        ext_mime, _ = mimetypes.guess_type(f"file{ext}")
        if guessed_mime and ext_mime and guessed_mime != ext_mime:
            logger.warning(
                "mime_ext_mismatch",
                path=str(source_path),
                guessed_mime=guessed_mime,
                ext_mime=ext_mime,
            )
        mime = guessed_mime or ext_mime

        # 6) Destination path: attachments_dir / <episode_id> / <sha256><ext>
        dest_dir = self._attachments_dir / str(episode_id)
        dest = dest_dir / f"{sha256_hex}{ext}"

        # 7) Deduplicate on disk: if dest exists with same sha256, skip copy
        if dest.exists():
            logger.info(
                "attachment_file_dedupe",
                episode_id=episode_id,
                sha256=sha256_hex[:8],
                dest=str(dest),
            )
        else:
            # 8) mkdir -p and copy with metadata
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest)
            logger.info(
                "attachment_stored",
                episode_id=episode_id,
                sha256=sha256_hex[:8],
                dest=str(dest),
                byte_size=byte_size,
            )

        # 9) Persist metadata to DB.
        # The schema enforces UNIQUE(episode_id, sha256) — if an identical file
        # was already ingested into this episode, skip the duplicate DB row.
        existing = self._store.find_attachment_by_sha256(episode_id, sha256_hex)
        if existing is not None:
            logger.info(
                "attachment_db_dedupe",
                episode_id=episode_id,
                sha256=sha256_hex[:8],
            )
        else:
            self._store.add_attachment(
                episode_id,
                sha256=sha256_hex,
                path=str(dest),
                mime=mime,
                ext=ext,
                byte_size=byte_size,
                description=description,
            )

        # 10) Return result
        return IngestedAttachment(
            sha256=sha256_hex,
            mime=mime,
            ext=ext,
            byte_size=byte_size,
            stored_path=dest,
            original_name=source_path.name,
        )


def _sha256_file(path: Path) -> str:
    """Compute hex-encoded sha256 of a file using 64 KB streaming chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()
