"""Unit tests for apps/tools/weather.py."""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.tools.weather import (
    WeatherSnapshot,
    WeatherUnavailable,
    _CACHE_TTL_S,
    _cache,
    _coord_locks,
    _round_coord,
    _wmo_to_ko,
    get_current_weather,
)


# ── Helpers ────────────────────────────────────────────────────────────────

_SAMPLE_RESPONSE: dict[str, Any] = {
    "current": {
        "temperature_2m": 18.5,
        "apparent_temperature": 16.2,
        "relative_humidity_2m": 65,
        "weather_code": 3,
        "wind_speed_10m": 12.4,
    }
}

_LAT, _LON = 37.5665, 126.9780


def _make_mock_client(data: dict[str, Any] | None = None, raise_exc: Exception | None = None) -> MagicMock:
    """Return an async httpx.AsyncClient-compatible mock."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=data or _SAMPLE_RESPONSE)

    client = MagicMock()
    if raise_exc is not None:
        client.get = AsyncMock(side_effect=raise_exc)
    else:
        client.get = AsyncMock(return_value=mock_resp)
    return client


def _clear_cache() -> None:
    """Remove test coords from caches between tests."""
    key = (_round_coord(_LAT), _round_coord(_LON))
    _cache.pop(key, None)
    _coord_locks.pop(key, None)


# ── Tests: parsing ─────────────────────────────────────────────────────────

class TestParsing:
    @pytest.mark.asyncio
    async def test_basic_snapshot(self) -> None:
        _clear_cache()
        client = _make_mock_client()
        w = await get_current_weather(lat=_LAT, lon=_LON, location_name="서울", client=client)

        assert isinstance(w, WeatherSnapshot)
        assert w.temperature_c == pytest.approx(18.5)
        assert w.feels_like_c == pytest.approx(16.2)
        assert w.humidity_pct == 65
        assert w.weather_code == 3
        assert w.wind_kmh == pytest.approx(12.4)
        assert w.location_name == "서울"
        assert "T" in w.fetched_at_iso  # ISO format

    @pytest.mark.asyncio
    async def test_weather_text_korean(self) -> None:
        _clear_cache()
        client = _make_mock_client()
        w = await get_current_weather(lat=_LAT, lon=_LON, location_name="서울", client=client)
        assert w.weather_text_ko == "흐림"  # WMO code 3


# ── Tests: WMO code mapping ────────────────────────────────────────────────

class TestWmoMapping:
    def test_clear_sky(self) -> None:
        assert _wmo_to_ko(0) == "맑음"

    def test_rain(self) -> None:
        assert _wmo_to_ko(61) == "비"

    def test_heavy_rain(self) -> None:
        assert _wmo_to_ko(65) == "폭우"

    def test_snow(self) -> None:
        assert _wmo_to_ko(71) == "눈"

    def test_thunderstorm(self) -> None:
        assert _wmo_to_ko(95) == "천둥번개"

    def test_fog(self) -> None:
        assert _wmo_to_ko(45) == "안개"

    def test_unknown_code_falls_back(self) -> None:
        result = _wmo_to_ko(999)
        assert "999" in result


# ── Tests: caching ─────────────────────────────────────────────────────────

class TestCaching:
    @pytest.mark.asyncio
    async def test_two_calls_within_ttl_one_http_request(self) -> None:
        _clear_cache()
        client = _make_mock_client()

        await get_current_weather(lat=_LAT, lon=_LON, location_name="서울", client=client)
        await get_current_weather(lat=_LAT, lon=_LON, location_name="서울", client=client)

        assert client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_after_ttl_refetch(self) -> None:
        _clear_cache()
        client = _make_mock_client()
        key = (_round_coord(_LAT), _round_coord(_LON))

        await get_current_weather(lat=_LAT, lon=_LON, location_name="서울", client=client)

        # Manually expire the cache entry
        _cache[key] = (time.monotonic() - _CACHE_TTL_S - 1, _cache[key][1])

        await get_current_weather(lat=_LAT, lon=_LON, location_name="서울", client=client)

        assert client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_location_name_in_snapshot(self) -> None:
        _clear_cache()
        client = _make_mock_client()
        w = await get_current_weather(lat=_LAT, lon=_LON, location_name="테스트도시", client=client)
        assert w.location_name == "테스트도시"


# ── Tests: error handling ──────────────────────────────────────────────────

class TestErrors:
    @pytest.mark.asyncio
    async def test_network_timeout_raises_unavailable(self) -> None:
        import httpx

        _clear_cache()
        client = _make_mock_client(raise_exc=httpx.TimeoutException("timed out"))

        with pytest.raises(WeatherUnavailable):
            await get_current_weather(lat=_LAT, lon=_LON, client=client)

    @pytest.mark.asyncio
    async def test_network_error_raises_unavailable(self) -> None:
        import httpx

        _clear_cache()
        client = _make_mock_client(raise_exc=httpx.NetworkError("connection refused"))

        with pytest.raises(WeatherUnavailable):
            await get_current_weather(lat=_LAT, lon=_LON, client=client)

    @pytest.mark.asyncio
    async def test_http_status_error_raises_unavailable(self) -> None:
        import httpx

        _clear_cache()
        mock_req = MagicMock()
        mock_resp_obj = MagicMock()
        mock_resp_obj.status_code = 500
        client = _make_mock_client(
            raise_exc=httpx.HTTPStatusError("500", request=mock_req, response=mock_resp_obj)
        )

        with pytest.raises(WeatherUnavailable):
            await get_current_weather(lat=_LAT, lon=_LON, client=client)
