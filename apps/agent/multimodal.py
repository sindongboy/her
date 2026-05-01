"""Multimodal helpers — convert files to Gemini Part objects.

Supports the whitelisted extensions from CLAUDE.md §6.2:
  image: .png .jpg .jpeg  → inline bytes with image/* mime
  PDF:   .pdf             → inline bytes with application/pdf
  text:  .txt .md         → text part (UTF-8)
  cal:   .ics             → text part with [ATTACHED ICS FILE] header
  mail:  .eml             → text part with [ATTACHED EML FILE] header

Per Phase 2 policy: files ≤ 4 MB are inlined; larger files are still sent
with a warning logged (Files API is out of scope for Phase 2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Phase 2 inline size limit (warn, but still send)
_INLINE_WARN_BYTES: int = 4 * 1024 * 1024  # 4 MB

# Mapping: suffix → mime for binary attachments
_BINARY_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

# Text-based extensions: read as UTF-8 and send as a labelled text part
_TEXT_EXTS: frozenset[str] = frozenset({".txt", ".md", ".ics", ".eml"})

# Header labels for non-prose text formats
_TEXT_HEADERS: dict[str, str] = {
    ".ics": "[ATTACHED ICS FILE]",
    ".eml": "[ATTACHED EML FILE]",
}


class AttachmentError(Exception):
    """Raised when a file cannot be converted to a Gemini Part."""


def file_to_part(
    path: Path,
    *,
    mime: str | None = None,
) -> Any:
    """Read a file and return a Gemini-compatible Part object.

    For text types (.txt, .md, .ics, .eml), returns a dict suitable for
    inclusion in the contents list as a text part.  For binary types
    (.png, .jpg, .jpeg, .pdf), returns an inline_data Part using the
    google.genai SDK when available, or a dict fallback otherwise.

    Args:
        path: Absolute path to the file. Must exist and have a whitelisted ext.
        mime: Override MIME type. Auto-detected from extension when None.

    Returns:
        A Gemini Part (SDK object or dict) suitable for use in `contents`.

    Raises:
        AttachmentError: If the extension is not supported.
    """
    if not path.exists() or not path.is_file():
        raise AttachmentError(f"file_not_found: {path}")

    ext = path.suffix.lower()

    # ── text types ────────────────────────────────────────────────────────
    if ext in _TEXT_EXTS:
        text_content = path.read_text(encoding="utf-8", errors="replace")
        header = _TEXT_HEADERS.get(ext, "")
        if header:
            text_content = f"{header}\n{text_content}"
        # Return a dict that _messages_to_contents will handle.
        return {"_part_type": "text", "text": text_content}

    # ── binary types ──────────────────────────────────────────────────────
    if ext not in _BINARY_MIME and mime is None:
        raise AttachmentError(
            f"unsupported_ext: '{ext}' is not a supported attachment type"
        )

    resolved_mime = mime or _BINARY_MIME[ext]
    data = path.read_bytes()

    if len(data) > _INLINE_WARN_BYTES:
        log.warning(
            "attachment_exceeds_inline_limit",
            path=str(path),
            bytes=len(data),
            limit_bytes=_INLINE_WARN_BYTES,
            note="Phase 2 sends inline anyway; use Files API in Phase 3+ for large files",
        )

    try:
        from google.genai import types  # type: ignore[import-untyped]

        return types.Part.from_bytes(data=data, mime_type=resolved_mime)
    except ImportError:
        # Fallback dict representation when SDK not available (tests / offline).
        return {"_part_type": "inline_data", "data": data, "mime_type": resolved_mime}
    except AttributeError:
        # SDK present but Part.from_bytes may not exist in all versions.
        try:
            return types.Part(  # type: ignore[union-attr]
                inline_data=types.Blob(data=data, mime_type=resolved_mime)
            )
        except Exception:
            return {"_part_type": "inline_data", "data": data, "mime_type": resolved_mime}


def parts_from_attachment_refs(
    refs: list[Any],
) -> list[Any]:
    """Convert a list of AttachmentRef objects to Gemini Part objects.

    Invalid or missing files are skipped with a warning.
    Descriptions are NOT sent to LLM directly (anonymization boundary).
    """
    parts: list[Any] = []
    for ref in refs:
        try:
            part = file_to_part(ref.path, mime=ref.mime)
            parts.append(part)
            log.debug(
                "multimodal.part_built",
                path=str(ref.path),
                mime=ref.mime,
            )
        except AttachmentError as exc:
            log.warning("multimodal.skip_attachment", error=str(exc))
    return parts
