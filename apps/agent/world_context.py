"""World-state context block — time, location, weather, and calendar events
injected into the system prompt each turn.

Provides a background-refreshed cache so AgentCore._build_system_prompt stays
synchronous while still serving fresh weather and calendar data.

Usage::
    cache = init_world_state_cache(settings)
    await cache.start()          # spawns background refresh tasks
    block = get_world_state_block()   # sync, fast — read by _build_system_prompt
    ...
    await cache.stop()           # cancels refresh tasks on shutdown

CLAUDE.md references:
  §2.1  Time/date accuracy in voice responses
  §3.2  httpx already in deps
  §1.2  Calendar use-case (know the user's actual schedule)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, date
from typing import Any

import structlog

from apps.tools.weather import WeatherSnapshot, WeatherUnavailable, get_current_weather
from apps.tools.calendar_mac import CalendarEvent, CalendarUnavailable, get_events

log = structlog.get_logger(__name__)

_KO_WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")


def _now_line() -> str:
    """Return the current time/date as a Korean-friendly string."""
    now = datetime.now().astimezone()
    weekday = _KO_WEEKDAYS[now.weekday()]
    tz = now.strftime("%Z") or "Local"
    return f"{now.strftime('%Y-%m-%d')} ({weekday}요일) {now.strftime('%H:%M')} {tz}"


def _date_label(d: date) -> str:
    """Return Korean date label like '2026-05-01 금요일'."""
    weekday = _KO_WEEKDAYS[d.weekday()]
    return f"{d.strftime('%Y-%m-%d')} {weekday}요일"


def _event_line(event: CalendarEvent) -> str:
    """Format a single calendar event for the world-state block."""
    if event.is_all_day:
        return f"  - 종일 {event.title}"
    # HH:MM only — strip seconds
    time_part = event.starts_at_iso[11:16] if len(event.starts_at_iso) >= 16 else event.starts_at_iso
    return f"  - {time_part} {event.title} ({event.calendar_name})"


def _build_calendar_section(
    events_today: list[CalendarEvent],
    events_tomorrow: list[CalendarEvent],
    calendar_error: str | None,
    max_events: int,
    today: date,
    tomorrow: date,
) -> str:
    """Return Korean calendar section lines for the world-state block."""
    lines: list[str] = []

    today_label = _date_label(today)
    if calendar_error:
        lines.append(f"[오늘 일정] (Calendar 권한 미허용)")
        return "\n".join(lines)

    if not events_today:
        lines.append(f"[오늘 일정] 없음")
    else:
        lines.append(f"[오늘 일정 ({today_label})]")
        shown = events_today[:max_events]
        for ev in shown:
            lines.append(_event_line(ev))
        extra = len(events_today) - len(shown)
        if extra > 0:
            lines.append(f"  (외 {extra}건 더)")

    tomorrow_label = _date_label(tomorrow)
    if not events_tomorrow:
        lines.append(f"[내일 일정] 없음")
    else:
        lines.append(f"[내일 일정 ({tomorrow_label})]")
        shown = events_tomorrow[:max_events]
        for ev in shown:
            lines.append(_event_line(ev))
        extra = len(events_tomorrow) - len(shown)
        if extra > 0:
            lines.append(f"  (외 {extra}건 더)")

    return "\n".join(lines)


class WorldStateCache:
    """Background-refreshed cache of the world-state block.

    AgentCore calls get_block() synchronously on every prompt build.
    Background asyncio tasks fetch weather and calendar on independent
    schedules and update the cached values.

    If weather or calendar is unavailable, the block still includes
    time + location (and a graceful notice for calendar).
    """

    def __init__(
        self,
        settings: Any,
        refresh_interval_s: float = 600.0,
        calendar_refresh_interval_s: float = 300.0,
        weather_fn: Callable[..., Any] = get_current_weather,
        events_fn: Callable[..., Any] = get_events,
    ) -> None:
        self._settings = settings
        self._refresh_interval_s = refresh_interval_s
        self._calendar_refresh_interval_s = calendar_refresh_interval_s
        self._weather_fn = weather_fn
        self._events_fn = events_fn

        self._last_weather: WeatherSnapshot | None = None
        self._events_today: list[CalendarEvent] = []
        self._events_tomorrow: list[CalendarEvent] = []
        self._calendar_error: str | None = None  # last permission/error message

        self._task: asyncio.Task[None] | None = None
        self._cal_task: asyncio.Task[None] | None = None

    def get_block(self) -> str:
        """Return the current world-state block (sync, fast).

        Always includes an up-to-date time line. Weather and calendar use
        last-known values; if none yet, their sections are omitted or show
        graceful notices.
        """
        return _build_block(
            settings=self._settings,
            weather=self._last_weather,
            events_today=self._events_today,
            events_tomorrow=self._events_tomorrow,
            calendar_error=self._calendar_error,
        )

    async def start(self) -> None:
        """Start background refresh tasks. Safe to call multiple times."""
        if self._task is None or self._task.done():
            await self._refresh_weather_once()
            self._task = asyncio.create_task(
                self._weather_refresh_loop(), name="world_state_refresh"
            )

        enable_cal: bool = getattr(self._settings, "enable_calendar", True)
        if enable_cal and (self._cal_task is None or self._cal_task.done()):
            await self._refresh_calendar_once()
            self._cal_task = asyncio.create_task(
                self._calendar_refresh_loop(), name="world_state_calendar_refresh"
            )

    async def stop(self) -> None:
        """Cancel background refresh tasks and wait for them to exit."""
        for task_attr in ("_task", "_cal_task"):
            task: asyncio.Task[None] | None = getattr(self, task_attr)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self, task_attr, None)

    # ── Weather ──────────────────────────────────────────────────────────────

    async def _refresh_weather_once(self) -> None:
        """Fetch weather once; silently downgrade on failure."""
        s = self._settings
        try:
            lat: float = getattr(s, "location_lat", 37.5665)
            lon: float = getattr(s, "location_lon", 126.9780)
            name: str = getattr(s, "location_name", "서울")
            weather = await asyncio.wait_for(
                self._weather_fn(lat=lat, lon=lon, location_name=name),
                timeout=3.0,
            )
            self._last_weather = weather
            log.debug("world_state.weather_refreshed", location=name, temp=weather.temperature_c)
        except (WeatherUnavailable, asyncio.TimeoutError, Exception) as exc:
            log.warning("world_state.weather_unavailable", error=str(exc))

    async def _weather_refresh_loop(self) -> None:
        """Periodic weather refresh; exits on CancelledError."""
        try:
            while True:
                await asyncio.sleep(self._refresh_interval_s)
                await self._refresh_weather_once()
        except asyncio.CancelledError:
            pass

    # ── Calendar ─────────────────────────────────────────────────────────────

    async def _refresh_calendar_once(self) -> None:
        """Fetch calendar events once; silently downgrade on failure."""
        s = self._settings
        days_ahead: int = getattr(s, "calendar_lookahead_days", 2)
        max_events: int = getattr(s, "calendar_max_events", 8)
        try:
            all_events = await asyncio.wait_for(
                self._events_fn(days_ahead=days_ahead, max_events=max_events * 2),
                timeout=30.0,
            )
            today = date.today()
            tomorrow_date = date.fromordinal(today.toordinal() + 1)

            today_events: list[CalendarEvent] = []
            tomorrow_events: list[CalendarEvent] = []
            for ev in all_events:
                ev_date = date.fromisoformat(ev.starts_at_iso[:10])
                if ev_date == today:
                    today_events.append(ev)
                elif ev_date == tomorrow_date:
                    tomorrow_events.append(ev)

            self._events_today = today_events
            self._events_tomorrow = tomorrow_events
            self._calendar_error = None
            log.debug(
                "world_state.calendar_refreshed",
                today_count=len(today_events),
                tomorrow_count=len(tomorrow_events),
            )
        except CalendarUnavailable as exc:
            log.warning("world_state.calendar_unavailable", error=str(exc)[:80])
            self._calendar_error = str(exc)
        except (asyncio.TimeoutError, Exception) as exc:
            log.warning("world_state.calendar_error", error=str(exc))
            self._calendar_error = str(exc)

    async def _calendar_refresh_loop(self) -> None:
        """Periodic calendar refresh; exits on CancelledError."""
        try:
            while True:
                await asyncio.sleep(self._calendar_refresh_interval_s)
                await self._refresh_calendar_once()
        except asyncio.CancelledError:
            pass


# ── Module-level singleton ────────────────────────────────────────────────────

_default_cache: WorldStateCache | None = None


def init_world_state_cache(
    settings: Any,
    *,
    refresh_interval_s: float = 600.0,
    calendar_refresh_interval_s: float = 300.0,
    weather_fn: Callable[..., Any] = get_current_weather,
    events_fn: Callable[..., Any] = get_events,
) -> WorldStateCache:
    """Create (and store as default) a WorldStateCache for the given settings.

    Call ``await cache.start()`` after this to begin background refreshes.
    """
    global _default_cache
    cache = WorldStateCache(
        settings,
        refresh_interval_s=refresh_interval_s,
        calendar_refresh_interval_s=calendar_refresh_interval_s,
        weather_fn=weather_fn,
        events_fn=events_fn,
    )
    _default_cache = cache
    return cache


def get_world_state_block() -> str:
    """Sync read of the current world-state block.

    Returns just the time block if no cache has been initialised yet.
    """
    if _default_cache is not None:
        return _default_cache.get_block()
    # Fallback: no cache — return time only (shouldn't happen in production)
    return _build_block(
        settings=None,
        weather=None,
        events_today=[],
        events_tomorrow=[],
        calendar_error=None,
    )


# ── Block builder ─────────────────────────────────────────────────────────────


def _build_block(
    settings: Any,
    weather: WeatherSnapshot | None,
    events_today: list[CalendarEvent] | None = None,
    events_tomorrow: list[CalendarEvent] | None = None,
    calendar_error: str | None = None,
) -> str:
    """Compose the [현재 상태] system-prompt addendum."""
    lines: list[str] = ["\n\n[현재 상태]", f"- 지금: {_now_line()}"]

    if settings is not None:
        name: str = getattr(settings, "location_name", "서울")
        lat: float = getattr(settings, "location_lat", 37.5665)
        lon: float = getattr(settings, "location_lon", 126.9780)
        lines.append(f"- 위치: {name} ({lat:.2f}, {lon:.2f})")

    if weather is not None:
        lines.append(
            f"- 오늘 날씨: {weather.weather_text_ko}, "
            f"{weather.temperature_c:.0f}°C "
            f"(체감 {weather.feels_like_c:.0f}°C), "
            f"습도 {weather.humidity_pct}%"
        )

    lines.append("- 위 정보를 시간·날씨 질문에 사용하세요. 추측 금지.")

    # Calendar section — only included when enable_calendar is True (or settings=None)
    enable_cal: bool = getattr(settings, "enable_calendar", True) if settings is not None else False
    if enable_cal or settings is None:
        # When settings=None (fallback), skip calendar block entirely
        if settings is not None:
            max_events: int = getattr(settings, "calendar_max_events", 8)
            today = date.today()
            tomorrow_date = date.fromordinal(today.toordinal() + 1)
            cal_section = _build_calendar_section(
                events_today=events_today or [],
                events_tomorrow=events_tomorrow or [],
                calendar_error=calendar_error,
                max_events=max_events,
                today=today,
                tomorrow=tomorrow_date,
            )
            lines.append(cal_section)

    return "\n".join(lines)
