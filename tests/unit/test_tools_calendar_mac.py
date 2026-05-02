"""Unit tests for apps/tools/calendar_mac.py.

All tests mock asyncio.create_subprocess_exec so no real osascript is run.
"""

from __future__ import annotations

import asyncio
import platform
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.tools.calendar_mac import (
    CalendarEvent,
    CalendarUnavailable,
    _detect_tcc_error,
    _events_cache,
    _parse_line,
    get_events,
)


@pytest.fixture(autouse=True)
def _clear_calendar_cache():
    """Clear the in-memory events cache between tests so canned stdout
    fixtures don't cross-contaminate."""
    _events_cache.clear()
    yield
    _events_cache.clear()


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_proc(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Return a fake asyncio subprocess process."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(
        return_value=(stdout.encode(), stderr.encode())
    )
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


# ── Unit: _detect_tcc_error ────────────────────────────────────────────────


class TestDetectTccError:
    def test_tcc_marker_in_stderr(self) -> None:
        assert _detect_tcc_error(
            "Not authorized to send Apple events to Calendar", returncode=1
        )

    def test_tcc_marker_1743(self) -> None:
        assert _detect_tcc_error("osascript: (-1743)", returncode=1)

    def test_execution_error(self) -> None:
        assert _detect_tcc_error("execution error: ...", returncode=1)

    def test_no_marker_returncode_nonzero(self) -> None:
        assert not _detect_tcc_error("some other error", returncode=1)

    def test_returncode_zero_ignored(self) -> None:
        assert not _detect_tcc_error(
            "Not authorized to send Apple events to Calendar", returncode=0
        )


# ── Unit: _parse_line ──────────────────────────────────────────────────────


class TestParseLine:
    def test_valid_timed_event(self) -> None:
        line = "팀 미팅|||회사|||false|||2099-05-01T14:00:00|||2099-05-01T15:00:00"
        ev = _parse_line(line)
        assert ev is not None
        assert ev.title == "팀 미팅"
        assert ev.calendar_name == "회사"
        assert ev.starts_at_iso == "2099-05-01T14:00:00"
        assert ev.ends_at_iso == "2099-05-01T15:00:00"
        assert ev.is_all_day is False

    def test_valid_allday_event(self) -> None:
        line = "워크숍|||개인|||true|||2099-05-02T00:00:00|||2099-05-03T00:00:00"
        ev = _parse_line(line)
        assert ev is not None
        assert ev.is_all_day is True

    def test_wrong_field_count_returns_none(self) -> None:
        assert _parse_line("only two|||fields") is None

    def test_empty_line_returns_none(self) -> None:
        assert _parse_line("") is None

    def test_single_field_returns_none(self) -> None:
        assert _parse_line("no separators at all") is None

    def test_empty_end_iso(self) -> None:
        line = "이벤트|||캘린더|||false|||2099-05-01T09:00:00|||"
        ev = _parse_line(line)
        assert ev is not None
        assert ev.ends_at_iso is None


# ── Integration: get_events ────────────────────────────────────────────────


_CANNED_STDOUT = (
    "팀 미팅|||회사|||false|||2099-05-01T14:00:00|||2099-05-01T15:00:00\n"
    "가족 저녁|||가족|||false|||2099-05-01T19:00:00|||2099-05-01T21:00:00\n"
    "종일 행사|||개인|||true|||2099-05-02T00:00:00|||2099-05-03T00:00:00\n"
)


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only path")
class TestGetEventsDarwin:
    @pytest.mark.asyncio
    async def test_canned_stdout_parsed_correctly(self) -> None:
        proc = _make_proc(stdout=_CANNED_STDOUT)
        with patch(
            "apps.tools.calendar_mac.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            events = await get_events(days_ahead=2)

        assert len(events) == 3
        assert events[0].title == "팀 미팅"
        assert events[0].is_all_day is False
        assert events[2].is_all_day is True

    @pytest.mark.asyncio
    async def test_empty_stdout_returns_empty_list(self) -> None:
        proc = _make_proc(stdout="")
        with patch(
            "apps.tools.calendar_mac.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            events = await get_events(days_ahead=2)
        assert events == []

    @pytest.mark.asyncio
    async def test_bad_lines_skipped_silently(self) -> None:
        stdout = (
            "bad_line_no_separators\n"
            "팀 미팅|||회사|||false|||2099-05-01T14:00:00|||2099-05-01T15:00:00\n"
            "only|||two\n"
        )
        proc = _make_proc(stdout=stdout)
        with patch(
            "apps.tools.calendar_mac.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            events = await get_events(days_ahead=2)
        assert len(events) == 1
        assert events[0].title == "팀 미팅"

    @pytest.mark.asyncio
    async def test_tcc_error_raises_calendar_unavailable(self) -> None:
        proc = _make_proc(
            stderr="Not authorized to send Apple events to Calendar",
            returncode=1,
        )
        with patch(
            "apps.tools.calendar_mac.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            with pytest.raises(CalendarUnavailable) as exc_info:
                await get_events(days_ahead=2)
        # Error message must be Korean and mention the permission instruction
        msg = str(exc_info.value)
        assert "Calendar" in msg
        assert "자동화" in msg or "개인정보" in msg or "권한" in msg

    @pytest.mark.asyncio
    async def test_timeout_raises_calendar_unavailable(self) -> None:
        async def _hang(*args: object, **kwargs: object) -> object:
            await asyncio.sleep(9999)

        proc = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)

        with patch(
            "apps.tools.calendar_mac.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            with patch(
                "apps.tools.calendar_mac.asyncio.wait_for",
                side_effect=asyncio.TimeoutError,
            ):
                with pytest.raises(CalendarUnavailable) as exc_info:
                    await get_events(days_ahead=2, timeout_s=0.001)
        assert "시간 초과" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_sorted_ascending_by_start(self) -> None:
        stdout = (
            "늦은 이벤트|||A|||false|||2099-05-01T18:00:00|||2099-05-01T19:00:00\n"
            "이른 이벤트|||B|||false|||2099-05-01T09:00:00|||2099-05-01T10:00:00\n"
        )
        proc = _make_proc(stdout=stdout)
        with patch(
            "apps.tools.calendar_mac.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            events = await get_events(days_ahead=2)
        assert events[0].title == "이른 이벤트"
        assert events[1].title == "늦은 이벤트"

    @pytest.mark.asyncio
    async def test_max_events_respected(self) -> None:
        lines = "\n".join(
            f"이벤트{i}|||캘린더|||false|||2099-05-01T{9+i:02d}:00:00|||2099-05-01T{10+i:02d}:00:00"
            for i in range(10)
        ) + "\n"
        proc = _make_proc(stdout=lines)
        with patch(
            "apps.tools.calendar_mac.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            events = await get_events(days_ahead=2, max_events=5)
        assert len(events) == 5


@pytest.mark.skipif(platform.system() == "Darwin", reason="non-Darwin path test")
class TestGetEventsNonDarwin:
    @pytest.mark.asyncio
    async def test_non_darwin_returns_empty(self) -> None:
        # On Darwin this is skipped; run manually on Linux CI
        events = await get_events(days_ahead=2)
        assert events == []


class TestGetEventsNonDarwinMocked:
    """Test non-Darwin path by mocking platform.system."""

    @pytest.mark.asyncio
    async def test_non_darwin_skip_returns_empty(self) -> None:
        with patch("apps.tools.calendar_mac.platform.system", return_value="Linux"):
            events = await get_events(days_ahead=2)
        assert events == []
