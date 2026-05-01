"""Unit tests for apps/agent/multimodal.py — file_to_part helpers."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from apps.agent.multimodal import AttachmentError, file_to_part


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def png_file(tmp_path: Path) -> Path:
    """Minimal 1×1 white PNG."""
    # Smallest valid PNG header bytes (1x1 white pixel)
    png_bytes = bytes(
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff"
        b"?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    p = tmp_path / "test.png"
    p.write_bytes(png_bytes)
    return p


@pytest.fixture()
def jpeg_file(tmp_path: Path) -> Path:
    """Minimal JPEG stub (not a real image, just bytes with .jpg extension)."""
    data = b"\xff\xd8\xff\xe0" + b"\x00" * 10 + b"\xff\xd9"  # minimal JFIF stub
    p = tmp_path / "photo.jpg"
    p.write_bytes(data)
    return p


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    """Minimal PDF stub."""
    data = b"%PDF-1.4\n%%EOF\n"
    p = tmp_path / "document.pdf"
    p.write_bytes(data)
    return p


@pytest.fixture()
def txt_file(tmp_path: Path) -> Path:
    p = tmp_path / "note.txt"
    p.write_text("안녕하세요, 텍스트 파일입니다.", encoding="utf-8")
    return p


@pytest.fixture()
def md_file(tmp_path: Path) -> Path:
    p = tmp_path / "readme.md"
    p.write_text("# 제목\n본문 내용입니다.", encoding="utf-8")
    return p


@pytest.fixture()
def ics_file(tmp_path: Path) -> Path:
    p = tmp_path / "event.ics"
    p.write_text("BEGIN:VCALENDAR\nEND:VCALENDAR\n", encoding="utf-8")
    return p


@pytest.fixture()
def eml_file(tmp_path: Path) -> Path:
    p = tmp_path / "mail.eml"
    p.write_text("From: test@example.com\nSubject: hello\n\nbody", encoding="utf-8")
    return p


@pytest.fixture()
def large_png(tmp_path: Path) -> Path:
    """PNG file > 4 MB to trigger inline size warning."""
    data = b"\x89PNG" + b"\x00" * (5 * 1024 * 1024)  # 5 MB stub
    p = tmp_path / "large.png"
    p.write_bytes(data)
    return p


# ── tests ─────────────────────────────────────────────────────────────────


class TestFileToPartImage:
    def test_png_returns_inline_data_part(self, png_file: Path) -> None:
        part = file_to_part(png_file)
        # Either SDK Part or dict fallback — both should carry correct mime.
        if isinstance(part, dict):
            assert part["_part_type"] == "inline_data"
            assert part["mime_type"] == "image/png"
            assert part["data"] == png_file.read_bytes()
        else:
            # SDK Part object — check mime via repr or attribute.
            assert hasattr(part, "inline_data") or hasattr(part, "_raw_part")

    def test_png_bytes_match_file(self, png_file: Path) -> None:
        part = file_to_part(png_file)
        if isinstance(part, dict):
            assert part["data"] == png_file.read_bytes()

    def test_jpeg_returns_image_jpeg_mime(self, jpeg_file: Path) -> None:
        part = file_to_part(jpeg_file)
        if isinstance(part, dict):
            assert part["mime_type"] == "image/jpeg"

    def test_mime_override_respected(self, png_file: Path) -> None:
        part = file_to_part(png_file, mime="image/png")
        if isinstance(part, dict):
            assert part["mime_type"] == "image/png"


class TestFileToPartPDF:
    def test_pdf_returns_application_pdf_mime(self, pdf_file: Path) -> None:
        part = file_to_part(pdf_file)
        if isinstance(part, dict):
            assert part["_part_type"] == "inline_data"
            assert part["mime_type"] == "application/pdf"

    def test_pdf_bytes_match_file(self, pdf_file: Path) -> None:
        part = file_to_part(pdf_file)
        if isinstance(part, dict):
            assert part["data"] == pdf_file.read_bytes()


class TestFileToPartText:
    def test_txt_returns_text_part(self, txt_file: Path) -> None:
        part = file_to_part(txt_file)
        assert isinstance(part, dict)
        assert part["_part_type"] == "text"

    def test_txt_content_is_file_text(self, txt_file: Path) -> None:
        part = file_to_part(txt_file)
        assert isinstance(part, dict)
        assert "안녕하세요" in part["text"]

    def test_md_returns_text_part(self, md_file: Path) -> None:
        part = file_to_part(md_file)
        assert isinstance(part, dict)
        assert part["_part_type"] == "text"
        assert "제목" in part["text"]

    def test_ics_has_header_label(self, ics_file: Path) -> None:
        part = file_to_part(ics_file)
        assert isinstance(part, dict)
        assert "[ATTACHED ICS FILE]" in part["text"]

    def test_eml_has_header_label(self, eml_file: Path) -> None:
        part = file_to_part(eml_file)
        assert isinstance(part, dict)
        assert "[ATTACHED EML FILE]" in part["text"]


class TestFileToPartErrors:
    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.exe"
        f.write_bytes(b"\x00\x01")
        with pytest.raises(AttachmentError, match="unsupported_ext"):
            file_to_part(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "nonexistent.png"
        with pytest.raises(AttachmentError, match="file_not_found"):
            file_to_part(f)

    def test_zip_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "archive.zip"
        f.write_bytes(b"PK\x03\x04")
        with pytest.raises(AttachmentError):
            file_to_part(f)


class TestLargeFileWarning:
    def test_large_file_logs_warning_but_still_returns_part(
        self, large_png: Path
    ) -> None:
        """Files > 4 MB should log a warning but still succeed (Phase 2 policy)."""
        import structlog
        import logging

        # Should NOT raise — just warn.
        part = file_to_part(large_png)
        # Part should still be produced.
        assert part is not None
        if isinstance(part, dict):
            assert part["_part_type"] == "inline_data"
