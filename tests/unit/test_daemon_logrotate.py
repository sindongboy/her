"""Unit tests for apps.daemon.logrotate — size-based log rotation."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.daemon.logrotate import _numbered, maybe_rotate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _content(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# maybe_rotate — basic behaviour
# ---------------------------------------------------------------------------


class TestMaybeRotateBasic:
    def test_no_op_when_file_missing(self, tmp_path: Path) -> None:
        log = tmp_path / "her.log"
        result = maybe_rotate(log, max_bytes=100, keep=5)
        assert result is False

    def test_no_op_when_below_threshold(self, tmp_path: Path) -> None:
        log = tmp_path / "her.log"
        _write(log, "small content")
        result = maybe_rotate(log, max_bytes=10_000, keep=5)
        assert result is False
        assert _content(log) == "small content"

    def test_rotates_when_at_threshold(self, tmp_path: Path) -> None:
        log = tmp_path / "her.log"
        payload = "x" * 100
        _write(log, payload)
        result = maybe_rotate(log, max_bytes=100, keep=5)
        assert result is True

    def test_active_log_empty_after_rotation(self, tmp_path: Path) -> None:
        log = tmp_path / "her.log"
        _write(log, "x" * 200)
        maybe_rotate(log, max_bytes=100, keep=5)
        assert log.exists()
        assert log.stat().st_size == 0

    def test_backup_one_has_old_contents(self, tmp_path: Path) -> None:
        log = tmp_path / "her.log"
        payload = "old content"
        _write(log, payload)
        # Force rotation by setting max_bytes = len(payload)
        maybe_rotate(log, max_bytes=len(payload.encode()), keep=5)
        assert _content(_numbered(log, 1)) == payload

    def test_idempotent_when_already_empty(self, tmp_path: Path) -> None:
        log = tmp_path / "her.log"
        log.touch()  # 0 bytes
        result = maybe_rotate(log, max_bytes=100, keep=5)
        assert result is False


# ---------------------------------------------------------------------------
# maybe_rotate — chained rotations
# ---------------------------------------------------------------------------


class TestMaybeRotateChained:
    def _do_rotation(self, log: Path, content: str) -> None:
        """Write content, rotate at that size."""
        _write(log, content)
        maybe_rotate(log, max_bytes=len(content.encode()), keep=5)

    def test_two_rotations_chain(self, tmp_path: Path) -> None:
        log = tmp_path / "her.log"
        self._do_rotation(log, "first")
        self._do_rotation(log, "second")
        # After 2 rotations: .1 = "second", .2 = "first"
        assert _content(_numbered(log, 1)) == "second"
        assert _content(_numbered(log, 2)) == "first"

    def test_oldest_dropped_at_keep_limit(self, tmp_path: Path) -> None:
        log = tmp_path / "her.log"
        for i in range(1, 8):  # 7 rotations
            self._do_rotation(log, f"content-{i}")
        # Only 5 backups retained
        assert _numbered(log, 5).exists()
        assert not _numbered(log, 6).exists()
        assert not _numbered(log, 7).exists()

    def test_exactly_keep_rotations(self, tmp_path: Path) -> None:
        log = tmp_path / "her.log"
        for i in range(1, 6):  # exactly 5 rotations
            self._do_rotation(log, f"entry-{i}")
        assert _numbered(log, 5).exists()
        assert not _numbered(log, 6).exists()

    def test_contents_shift_correctly(self, tmp_path: Path) -> None:
        log = tmp_path / "her.log"
        contents = [f"line-{i}" for i in range(1, 4)]
        for c in contents:
            self._do_rotation(log, c)
        # Most recent backup is at .1
        assert _content(_numbered(log, 1)) == "line-3"
        assert _content(_numbered(log, 2)) == "line-2"
        assert _content(_numbered(log, 3)) == "line-1"


# ---------------------------------------------------------------------------
# _numbered helper
# ---------------------------------------------------------------------------


class TestNumbered:
    def test_appends_suffix(self, tmp_path: Path) -> None:
        base = tmp_path / "her.log"
        assert _numbered(base, 1) == Path(str(base) + ".1")
        assert _numbered(base, 3) == Path(str(base) + ".3")
