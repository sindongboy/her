"""Tests for apps.tools.stocks — ticker routing + provider fallbacks."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from apps.tools import stocks


class TestClassify:
    @pytest.mark.parametrize(
        "ticker,expected",
        [
            ("AAPL",       ("global", "AAPL")),
            ("aapl",       ("global", "AAPL")),
            ("BRK.B",      ("global", "BRK.B")),
            ("005930",     ("kr", "005930")),
            ("005930.KS",  ("kr", "005930")),
            ("005930.ks",  ("kr", "005930")),
            ("000660.KQ",  ("kr", "000660")),
            ("123456",     ("kr", "123456")),
            ("12345",      ("global", "12345")),  # 5 digits — not KR
            ("TSLA.KS",    ("global", "TSLA.KS")),  # alphas + .KS not KR
        ],
    )
    def test_classify(self, ticker: str, expected: tuple[str, str]) -> None:
        assert stocks.classify(ticker) == expected


class TestGetQuotesRouting:
    def setup_method(self) -> None:
        # Reset cache between tests so cached errors don't leak.
        stocks._quote_cache.clear()
        stocks._kiwoom_token = None

    def test_no_keys_yields_per_ticker_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        monkeypatch.delenv("KIWOOM_APP_KEY", raising=False)
        monkeypatch.delenv("KIWOOM_APP_SECRET", raising=False)
        out = asyncio.run(stocks.get_quotes(["AAPL", "005930.KS"]))
        assert len(out) == 2
        assert all("error" in q for q in out)

    def test_finnhub_used_for_global_ticker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FINNHUB_API_KEY", "X")
        monkeypatch.delenv("KIWOOM_APP_KEY", raising=False)
        monkeypatch.delenv("KIWOOM_APP_SECRET", raising=False)

        async def fake_finnhub(ticker: str, key: str) -> dict[str, Any]:
            return {
                "ticker": ticker, "name": "Apple Inc.",
                "price": 213.4, "change": 1.2, "change_pct": 0.56,
                "currency": "USD", "source": "finnhub",
            }

        with patch.object(stocks, "_fetch_finnhub", side_effect=fake_finnhub):
            out = asyncio.run(stocks.get_quotes(["AAPL"]))
        assert out[0]["source"] == "finnhub"
        assert out[0]["price"] == 213.4

    def test_polygon_fallback_when_finnhub_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.setenv("POLYGON_API_KEY", "Y")

        async def fake_polygon(ticker: str, key: str) -> dict[str, Any]:
            return {
                "ticker": ticker, "name": "",
                "price": 100.0, "change": 0.0, "change_pct": 0.0,
                "currency": "USD", "source": "polygon",
            }

        with patch.object(stocks, "_fetch_polygon", side_effect=fake_polygon):
            out = asyncio.run(stocks.get_quotes(["NVDA"]))
        assert out[0]["source"] == "polygon"

    def test_kiwoom_used_for_kr_ticker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KIWOOM_APP_KEY", "K")
        monkeypatch.setenv("KIWOOM_APP_SECRET", "S")

        async def fake_kiwoom(ticker: str, code: str) -> dict[str, Any]:
            assert code == "005930"
            return {
                "ticker": ticker, "name": "삼성전자",
                "price": 78900, "change": 200, "change_pct": 0.25,
                "currency": "KRW", "source": "kiwoom",
            }

        with patch.object(stocks, "_fetch_kiwoom", side_effect=fake_kiwoom):
            out = asyncio.run(stocks.get_quotes(["005930.KS"]))
        assert out[0]["source"] == "kiwoom"
        assert out[0]["currency"] == "KRW"
        assert out[0]["name"] == "삼성전자"

    def test_caches_within_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINNHUB_API_KEY", "X")

        call_count = {"n": 0}

        async def fake(ticker: str, key: str) -> dict[str, Any]:
            call_count["n"] += 1
            return {"ticker": ticker, "name": "", "price": 1.0,
                    "change": 0, "change_pct": 0,
                    "currency": "USD", "source": "finnhub"}

        with patch.object(stocks, "_fetch_finnhub", side_effect=fake):
            asyncio.run(stocks.get_quotes(["MSFT"]))
            asyncio.run(stocks.get_quotes(["MSFT"]))  # cached
        assert call_count["n"] == 1

    def test_per_ticker_failure_does_not_drop_others(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FINNHUB_API_KEY", "X")
        monkeypatch.setenv("KIWOOM_APP_KEY", "K")
        monkeypatch.setenv("KIWOOM_APP_SECRET", "S")

        async def fake_finnhub(ticker: str, key: str) -> dict[str, Any]:
            raise RuntimeError("boom")

        async def fake_kiwoom(ticker: str, code: str) -> dict[str, Any]:
            return {"ticker": ticker, "name": "삼성전자", "price": 78900,
                    "change": 0, "change_pct": 0, "currency": "KRW",
                    "source": "kiwoom"}

        with patch.object(stocks, "_fetch_finnhub", side_effect=fake_finnhub), \
             patch.object(stocks, "_fetch_polygon", side_effect=fake_finnhub), \
             patch.object(stocks, "_fetch_kiwoom", side_effect=fake_kiwoom):
            out = asyncio.run(stocks.get_quotes(["AAPL", "005930"]))

        out_by_ticker = {q["ticker"]: q for q in out}
        assert "error" in out_by_ticker["AAPL"]
        assert out_by_ticker["005930"]["source"] == "kiwoom"
