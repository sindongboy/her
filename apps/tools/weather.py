"""Open-Meteo weather tool — free, no API key required.

Fetches current weather conditions for a given lat/lon and returns a
WeatherSnapshot. Results are cached in-memory for _CACHE_TTL_S seconds
(default 600 = 10 min) to avoid hammering the free API on every agent turn.

CLAUDE.md references:
  §4    Tool Registry — weather is an external integration
  §3.2  httpx already in deps
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

_CACHE_TTL_S: float = 600.0  # 10 minutes

# (rounded_lat, rounded_lon) → (fetched_monotonic, snapshot)
_cache: dict[tuple[float, float], tuple[float, "WeatherSnapshot"]] = {}

# Per-coord locks to prevent thundering herd
_coord_locks: dict[tuple[float, float], asyncio.Lock] = {}
_locks_lock = asyncio.Lock()

# WMO weather code → Korean text (common codes; unknown codes fall back to generic label)
_WMO_KO: dict[int, str] = {
    0: "맑음",
    1: "대체로 맑음",
    2: "구름 조금",
    3: "흐림",
    45: "안개",
    48: "안개",
    51: "이슬비",
    53: "이슬비",
    55: "이슬비",
    56: "얼어붙는 이슬비",
    57: "얼어붙는 이슬비",
    61: "비",
    63: "비",
    65: "폭우",
    66: "얼어붙는 비",
    67: "얼어붙는 비",
    71: "눈",
    73: "눈",
    75: "폭설",
    77: "싸락눈",
    80: "소나기",
    81: "소나기",
    82: "강한 소나기",
    85: "눈 소나기",
    86: "눈 소나기",
    95: "천둥번개",
    96: "우박 동반 천둥번개",
    99: "우박 동반 천둥번개",
}

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherUnavailable(Exception):
    """Raised when the weather fetch fails (network error, timeout, bad response)."""


@dataclass(slots=True, frozen=True)
class WeatherSnapshot:
    temperature_c: float
    feels_like_c: float
    humidity_pct: int
    weather_code: int
    weather_text_ko: str
    wind_kmh: float
    fetched_at_iso: str
    location_name: str


def _wmo_to_ko(code: int) -> str:
    """Map WMO weather code to Korean description."""
    return _WMO_KO.get(code, f"날씨 코드 {code}")


def _round_coord(v: float) -> float:
    return round(v, 2)


def _parse_response(data: dict[str, Any], location_name: str) -> WeatherSnapshot:
    """Parse Open-Meteo response dict into a WeatherSnapshot."""
    current = data.get("current", {})
    temp = float(current.get("temperature_2m", 0.0))
    feels = float(current.get("apparent_temperature", temp))
    humidity = int(current.get("relative_humidity_2m", 0))
    code = int(current.get("weather_code", 0))
    wind = float(current.get("wind_speed_10m", 0.0))
    fetched_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    return WeatherSnapshot(
        temperature_c=temp,
        feels_like_c=feels,
        humidity_pct=humidity,
        weather_code=code,
        weather_text_ko=_wmo_to_ko(code),
        wind_kmh=wind,
        fetched_at_iso=fetched_at,
        location_name=location_name,
    )


async def _get_coord_lock(key: tuple[float, float]) -> asyncio.Lock:
    """Return (creating if needed) the per-coord asyncio.Lock."""
    async with _locks_lock:
        if key not in _coord_locks:
            _coord_locks[key] = asyncio.Lock()
        return _coord_locks[key]


async def get_current_weather(
    *,
    lat: float,
    lon: float,
    location_name: str = "",
    timeout_s: float = 3.0,
    client: object | None = None,
) -> WeatherSnapshot:
    """Fetch current weather from Open-Meteo, with in-memory cache.

    Caches result for _CACHE_TTL_S seconds keyed on (lat, lon) rounded to
    2 decimals. Concurrent callers for the same coord share one HTTP request
    via a per-coord asyncio.Lock.

    On network error / timeout → raises WeatherUnavailable.
    """
    key = (_round_coord(lat), _round_coord(lon))
    now = time.monotonic()

    # Fast path: cache hit without acquiring any lock
    cached = _cache.get(key)
    if cached is not None and (now - cached[0]) < _CACHE_TTL_S:
        log.debug("weather.cache_hit", location=location_name)
        return cached[1]

    coord_lock = await _get_coord_lock(key)

    async with coord_lock:
        # Re-check under lock (another coro may have populated it while we waited)
        cached = _cache.get(key)
        if cached is not None and (now - cached[0]) < _CACHE_TTL_S:
            log.debug("weather.cache_hit_under_lock", location=location_name)
            return cached[1]

        # Perform the fetch
        snapshot = await _do_fetch(lat, lon, location_name, timeout_s, client)
        _cache[key] = (time.monotonic(), snapshot)
        return snapshot


async def _do_fetch(
    lat: float,
    lon: float,
    location_name: str,
    timeout_s: float,
    client: object | None,
) -> WeatherSnapshot:
    """Perform the actual HTTP GET to Open-Meteo and parse the result."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "weather_code,wind_speed_10m"
        ),
        "timezone": "auto",
    }
    try:
        if client is not None:
            # Injected test client (httpx.AsyncClient-compatible)
            resp = await client.get(  # type: ignore[union-attr]
                _OPEN_METEO_URL, params=params, timeout=timeout_s
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        else:
            async with httpx.AsyncClient() as http:
                resp = await http.get(_OPEN_METEO_URL, params=params, timeout=timeout_s)
                resp.raise_for_status()
                data = resp.json()
    except httpx.TimeoutException as exc:
        raise WeatherUnavailable(f"Open-Meteo timeout after {timeout_s}s") from exc
    except (httpx.NetworkError, httpx.HTTPStatusError) as exc:
        raise WeatherUnavailable(f"Open-Meteo fetch failed: {exc}") from exc
    except WeatherUnavailable:
        raise
    except Exception as exc:
        raise WeatherUnavailable(f"Unexpected weather error: {exc}") from exc

    log.debug(
        "weather.fetched",
        location=location_name,
        code=data.get("current", {}).get("weather_code"),
    )
    return _parse_response(data, location_name)
