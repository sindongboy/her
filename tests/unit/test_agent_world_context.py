"""Unit tests for apps/agent/world_context.py."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.agent.world_context import (
    WorldStateCache,
    _build_block,
    get_world_state_block,
    init_world_state_cache,
)
from apps.tools.weather import WeatherSnapshot, WeatherUnavailable


# ── Fixtures ───────────────────────────────────────────────────────────────

@dataclass
class _FakeSettings:
    location_name: str = "서울"
    location_lat: float = 37.5665
    location_lon: float = 126.9780
    enable_search_grounding: bool = True


def _make_snapshot(temp: float = 20.0, code: int = 0, text: str = "맑음") -> WeatherSnapshot:
    return WeatherSnapshot(
        temperature_c=temp,
        feels_like_c=temp - 2.0,
        humidity_pct=55,
        weather_code=code,
        weather_text_ko=text,
        wind_kmh=8.0,
        fetched_at_iso=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        location_name="서울",
    )


# ── Tests: WorldStateCache ─────────────────────────────────────────────────

class TestWorldStateCacheNoRefresh:
    def test_get_block_before_refresh_has_time(self) -> None:
        """Before start(), get_block() returns at least the time line."""
        settings = _FakeSettings()
        cache = WorldStateCache(settings, weather_fn=AsyncMock(return_value=_make_snapshot()))
        block = cache.get_block()

        assert "[현재 상태]" in block
        assert "지금:" in block

    def test_get_block_before_refresh_no_weather(self) -> None:
        """Before any refresh, the weather data line is absent."""
        settings = _FakeSettings()
        cache = WorldStateCache(settings, weather_fn=AsyncMock(return_value=_make_snapshot()))
        block = cache.get_block()

        # The weather data line starts with "- 오늘 날씨:"
        assert "오늘 날씨:" not in block

    def test_get_block_has_location(self) -> None:
        """Location line appears after construction (no weather needed)."""
        settings = _FakeSettings(location_name="부산")
        cache = WorldStateCache(settings, weather_fn=AsyncMock(return_value=_make_snapshot()))
        block = cache.get_block()

        assert "부산" in block


class TestWorldStateCacheAfterRefresh:
    @pytest.mark.asyncio
    async def test_after_start_includes_weather(self) -> None:
        snapshot = _make_snapshot(temp=22.0, code=0, text="맑음")
        weather_fn = AsyncMock(return_value=snapshot)
        settings = _FakeSettings()

        cache = WorldStateCache(settings, weather_fn=weather_fn)
        await cache.start()
        block = cache.get_block()
        await cache.stop()

        assert "오늘 날씨:" in block
        assert "22°C" in block
        assert "맑음" in block

    @pytest.mark.asyncio
    async def test_weather_failure_block_still_has_time_and_location(self) -> None:
        """WeatherUnavailable must not crash the block — time + location survive."""
        weather_fn = AsyncMock(side_effect=WeatherUnavailable("network down"))
        settings = _FakeSettings(location_name="서울")

        cache = WorldStateCache(settings, weather_fn=weather_fn)
        await cache.start()
        block = cache.get_block()
        await cache.stop()

        assert "[현재 상태]" in block
        assert "지금:" in block
        assert "서울" in block
        assert "오늘 날씨:" not in block

    @pytest.mark.asyncio
    async def test_stop_cancels_background_task(self) -> None:
        """stop() should cancel and await the background task cleanly."""
        weather_fn = AsyncMock(return_value=_make_snapshot())
        cache = WorldStateCache(_FakeSettings(), refresh_interval_s=9999.0, weather_fn=weather_fn)
        await cache.start()
        assert cache._task is not None
        task = cache._task
        await cache.stop()
        assert task.done()

    @pytest.mark.asyncio
    async def test_start_twice_does_not_duplicate_task(self) -> None:
        weather_fn = AsyncMock(return_value=_make_snapshot())
        cache = WorldStateCache(_FakeSettings(), weather_fn=weather_fn)
        await cache.start()
        task1 = cache._task
        await cache.start()  # second call — should no-op
        task2 = cache._task
        assert task1 is task2
        await cache.stop()


# ── Tests: _build_block ────────────────────────────────────────────────────

class TestBuildBlock:
    def test_no_settings_no_weather(self) -> None:
        block = _build_block(settings=None, weather=None)
        assert "[현재 상태]" in block
        assert "지금:" in block
        assert "오늘 날씨:" not in block
        assert "위치:" not in block

    def test_settings_no_weather(self) -> None:
        block = _build_block(settings=_FakeSettings(location_name="인천"), weather=None)
        assert "인천" in block
        assert "오늘 날씨:" not in block

    def test_settings_with_weather(self) -> None:
        w = _make_snapshot(temp=15.0, code=61, text="비")
        block = _build_block(settings=_FakeSettings(location_name="제주"), weather=w)
        assert "제주" in block
        assert "오늘 날씨:" in block
        assert "비" in block
        assert "15°C" in block

    def test_disclaimer_line_present(self) -> None:
        block = _build_block(settings=None, weather=None)
        assert "추측 금지" in block


# ── Tests: module-level helpers ────────────────────────────────────────────

class TestModuleLevelHelpers:
    def test_get_world_state_block_before_init(self) -> None:
        """Falls back to time-only block if no cache initialised."""
        import apps.agent.world_context as wc
        old_cache = wc._default_cache
        wc._default_cache = None
        try:
            block = get_world_state_block()
            assert "[현재 상태]" in block
        finally:
            wc._default_cache = old_cache

    def test_init_world_state_cache_sets_default(self) -> None:
        import apps.agent.world_context as wc
        settings = _FakeSettings()
        cache = init_world_state_cache(settings, weather_fn=AsyncMock(return_value=_make_snapshot()))
        assert wc._default_cache is cache


# ── Calendar fixtures & helpers ────────────────────────────────────────────


from dataclasses import dataclass as _dc, replace as _replace
from datetime import date as _date

from apps.tools.calendar_mac import CalendarEvent, CalendarUnavailable


@_dc
class _FakeSettingsWithCal:
    location_name: str = "서울"
    location_lat: float = 37.5665
    location_lon: float = 126.9780
    enable_search_grounding: bool = True
    enable_calendar: bool = True
    calendar_lookahead_days: int = 2
    calendar_max_events: int = 8


def _make_today_event(title: str = "팀 미팅", cal: str = "회사") -> CalendarEvent:
    today_str = _date.today().isoformat()
    return CalendarEvent(
        title=title,
        calendar_name=cal,
        starts_at_iso=f"{today_str}T14:00:00",
        ends_at_iso=f"{today_str}T15:00:00",
        is_all_day=False,
    )


def _make_tomorrow_event(title: str = "워크숍", cal: str = "개인") -> CalendarEvent:
    tomorrow = _date.fromordinal(_date.today().toordinal() + 1)
    return CalendarEvent(
        title=title,
        calendar_name=cal,
        starts_at_iso=f"{tomorrow.isoformat()}T00:00:00",
        ends_at_iso=None,
        is_all_day=True,
    )


# ── Tests: calendar integration in WorldStateCache ─────────────────────────


class TestWorldStateCacheCalendar:
    @pytest.mark.asyncio
    async def test_calendar_events_appear_in_block(self) -> None:
        """After start(), get_block() includes today and tomorrow sections."""
        today_ev = _make_today_event("팀 미팅", "회사")
        tomorrow_ev = _make_tomorrow_event("워크숍", "개인")
        events_fn = AsyncMock(return_value=[today_ev, tomorrow_ev])
        weather_fn = AsyncMock(return_value=_make_snapshot())
        settings = _FakeSettingsWithCal()

        cache = WorldStateCache(settings, weather_fn=weather_fn, events_fn=events_fn)
        await cache.start()
        block = cache.get_block()
        await cache.stop()

        assert "오늘 일정" in block
        assert "내일 일정" in block
        assert "팀 미팅" in block
        assert "워크숍" in block

    @pytest.mark.asyncio
    async def test_calendar_unavailable_shows_permission_notice(self) -> None:
        """CalendarUnavailable → block shows '(Calendar 권한 미허용)'."""
        events_fn = AsyncMock(side_effect=CalendarUnavailable("권한 없음"))
        weather_fn = AsyncMock(return_value=_make_snapshot())
        settings = _FakeSettingsWithCal()

        cache = WorldStateCache(settings, weather_fn=weather_fn, events_fn=events_fn)
        await cache.start()
        block = cache.get_block()
        await cache.stop()

        assert "Calendar 권한 미허용" in block
        # Must not crash — time + location still present
        assert "지금:" in block

    @pytest.mark.asyncio
    async def test_enable_calendar_false_no_calendar_section(self) -> None:
        """enable_calendar=False → calendar refresh task never spawned; no calendar section."""
        events_fn = AsyncMock(return_value=[_make_today_event()])
        weather_fn = AsyncMock(return_value=_make_snapshot())

        @_dc
        class _NoCal:
            location_name: str = "서울"
            location_lat: float = 37.5665
            location_lon: float = 126.9780
            enable_search_grounding: bool = True
            enable_calendar: bool = False
            calendar_lookahead_days: int = 2
            calendar_max_events: int = 8

        settings = _NoCal()
        cache = WorldStateCache(settings, weather_fn=weather_fn, events_fn=events_fn)
        await cache.start()
        block = cache.get_block()
        await cache.stop()

        # Calendar section must be absent
        assert "오늘 일정" not in block
        # Calendar refresh task should never have been spawned
        assert cache._cal_task is None
        # events_fn should not have been called
        events_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_truncation_with_extra_suffix(self) -> None:
        """More events than max_events → '(외 N건 더)' appears in block."""
        today_iso = _date.today().isoformat()
        many_events = [
            CalendarEvent(
                title=f"이벤트{i}",
                calendar_name="테스트",
                starts_at_iso=f"{today_iso}T{9+i:02d}:00:00",
                ends_at_iso=f"{today_iso}T{10+i:02d}:00:00",
                is_all_day=False,
            )
            for i in range(12)
        ]
        events_fn = AsyncMock(return_value=many_events)
        weather_fn = AsyncMock(return_value=_make_snapshot())

        @_dc
        class _SmallMax:
            location_name: str = "서울"
            location_lat: float = 37.5665
            location_lon: float = 126.9780
            enable_search_grounding: bool = True
            enable_calendar: bool = True
            calendar_lookahead_days: int = 2
            calendar_max_events: int = 5

        settings = _SmallMax()
        cache = WorldStateCache(settings, weather_fn=weather_fn, events_fn=events_fn)
        await cache.start()
        block = cache.get_block()
        await cache.stop()

        assert "외" in block and "건 더" in block

    @pytest.mark.asyncio
    async def test_no_events_shows_없음(self) -> None:
        """Empty event list → '오늘 일정] 없음' in block."""
        events_fn = AsyncMock(return_value=[])
        weather_fn = AsyncMock(return_value=_make_snapshot())
        settings = _FakeSettingsWithCal()

        cache = WorldStateCache(settings, weather_fn=weather_fn, events_fn=events_fn)
        await cache.start()
        block = cache.get_block()
        await cache.stop()

        assert "없음" in block


# ── Tests: _build_block with calendar ─────────────────────────────────────


class TestBuildBlockCalendar:
    def test_calendar_section_in_build_block(self) -> None:
        today_ev = _make_today_event("세미나", "학교")
        block = _build_block(
            settings=_FakeSettingsWithCal(),
            weather=None,
            events_today=[today_ev],
            events_tomorrow=[],
            calendar_error=None,
        )
        assert "세미나" in block
        assert "오늘 일정" in block

    def test_calendar_error_in_build_block(self) -> None:
        block = _build_block(
            settings=_FakeSettingsWithCal(),
            weather=None,
            events_today=[],
            events_tomorrow=[],
            calendar_error="Permission denied",
        )
        assert "Calendar 권한 미허용" in block

    def test_no_settings_no_calendar_section(self) -> None:
        """When settings=None (fallback), calendar section must not appear."""
        block = _build_block(
            settings=None,
            weather=None,
            events_today=[_make_today_event()],
            events_tomorrow=[],
            calendar_error=None,
        )
        assert "오늘 일정" not in block
