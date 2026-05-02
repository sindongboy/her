"""Tests for apps.tools.news (Tavily wrapper + news-intent heuristic)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.tools import news


@pytest.fixture(autouse=True)
def _clear_cache():
    news._cache.clear()
    yield
    news._cache.clear()


class TestLooksLikeNewsQuery:
    @pytest.mark.parametrize("msg", [
        "테슬라 최근 뉴스 알려줘",
        "삼성전자 최근 동향 어때",
        "what's happening with NVDA",
        "latest news on Apple",
    ])
    def test_match(self, msg: str) -> None:
        assert news.looks_like_news_query(msg)

    @pytest.mark.parametrize("msg", [
        "어머니 생신 선물 추천해줘",
        "오늘 날씨 어때?",
        "what is your name",
    ])
    def test_no_match(self, msg: str) -> None:
        assert not news.looks_like_news_query(msg)


class TestSearchNews:
    def test_returns_empty_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        out = asyncio.run(news.search_news("any query"))
        assert out == []

    def test_returns_empty_on_api_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "X")

        class FailingClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def post(self, *a, **k):
                raise RuntimeError("boom")

        with patch("apps.tools.news.httpx.AsyncClient", return_value=FailingClient()):
            out = asyncio.run(news.search_news("AAPL"))
        assert out == []

    def test_normalises_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "X")

        body = {
            "results": [
                {
                    "title": "Tesla beats earnings",
                    "url": "https://www.example.com/tesla",
                    "content": "...",
                    "published_date": "2026-05-01T10:00:00Z",
                    "score": 0.91,
                },
                {
                    "title": "NVDA up 5%",
                    "url": "https://reuters.com/nvda",
                    "content": "...",
                },
            ]
        }
        resp = MagicMock()
        resp.json.return_value = body
        resp.raise_for_status = MagicMock()

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.post = AsyncMock(return_value=resp)

        with patch("apps.tools.news.httpx.AsyncClient", return_value=client):
            out = asyncio.run(news.search_news("test query"))

        assert len(out) == 2
        assert out[0]["title"] == "Tesla beats earnings"
        assert out[0]["source"] == "example.com"
        assert out[0]["url"].startswith("https://")
        assert out[1]["source"] == "reuters.com"

    def test_caches_within_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "X")

        resp = MagicMock()
        resp.json.return_value = {"results": []}
        resp.raise_for_status = MagicMock()

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        post_mock = AsyncMock(return_value=resp)
        client.post = post_mock

        with patch("apps.tools.news.httpx.AsyncClient", return_value=client):
            asyncio.run(news.search_news("AAPL"))
            asyncio.run(news.search_news("AAPL"))  # cached

        assert post_mock.await_count == 1


class TestSearchStockNews:
    def test_strips_kr_suffix_from_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        async def fake_search(query, *, max_results, days, topic="news"):
            captured["query"] = query
            return []

        with patch.object(news, "search_news", side_effect=fake_search):
            asyncio.run(news.search_stock_news("005930.KS"))

        assert "005930" in captured["query"]
        assert ".KS" not in captured["query"]
        assert "stock news" in captured["query"]


class TestFormatForPrompt:
    def test_empty_returns_empty_string(self) -> None:
        assert news.format_for_prompt([]) == ""

    def test_renders_block_with_label(self) -> None:
        items = [
            {
                "title": "Title A",
                "url": "https://x.com/a",
                "content": "Body of article A.",
                "published_date": "2026-04-30T10:00:00Z",
            }
        ]
        block = news.format_for_prompt(items, label="최근 뉴스")
        assert "[최근 뉴스 (Tavily)]" in block
        assert "Title A" in block
        assert "https://x.com/a" in block
        assert "2026-04-30" in block
