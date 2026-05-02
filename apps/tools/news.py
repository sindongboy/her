"""Tavily-backed news search.

Used by:
  - /api/widgets/stock-news → frontend 주식 뉴스 widget
  - AgentCore.maybe_inject_news → adds a news block to the system prompt
    when the user message looks like a news-seeking query.

In-memory TTL cache so widget polls and chat turns don't re-hit Tavily for
the same query inside a short window.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"
_CACHE_TTL_S = 5 * 60.0  # 5 min — news doesn't change minute-to-minute
_HTTP_TIMEOUT_S = 8.0

# (query, max_results, days, topic) -> (fetched_monotonic, results)
_cache: dict[tuple[str, int, int, str], tuple[float, list[dict[str, Any]]]] = {}
_lock = asyncio.Lock()


# ── Triggers used by the agent to decide whether to inject news ────────

_NEWS_TRIGGERS: tuple[str, ...] = (
    "뉴스",
    "최근 동향",
    "최근 어떻",
    "최근에 무슨",
    "최근 소식",
    "what's happening",
    "what is happening",
    "recent news",
    "latest news",
)


def looks_like_news_query(message: str) -> bool:
    lower = message.lower()
    return any(t in lower for t in _NEWS_TRIGGERS)


# ── Public API ─────────────────────────────────────────────────────────


async def search_news(
    query: str,
    *,
    max_results: int = 5,
    days: int = 7,
    topic: str = "news",
    bypass_cache: bool = False,
) -> list[dict[str, Any]]:
    """Tavily search. Returns a normalised list of news items.

    Each item: {title, url, content, published_date?, score?, source?}
    Empty list on missing key / API failure (fail-quiet).

    When *bypass_cache* is True, the cached entry for this query is
    discarded and a fresh API call is made — used by the widget's
    manual refresh button.
    """
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        return []

    cache_key = (query.strip(), max_results, days, topic)
    now = time.monotonic()

    if bypass_cache:
        _cache.pop(cache_key, None)
    else:
        cached = _cache.get(cache_key)
        if cached and (now - cached[0]) < _CACHE_TTL_S:
            return cached[1]

    async with _lock:
        cached = _cache.get(cache_key)
        if cached and (time.monotonic() - cached[0]) < _CACHE_TTL_S:
            return cached[1]
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as cli:
                r = await cli.post(
                    _TAVILY_URL,
                    json={
                        "api_key": key,
                        "query": query,
                        "topic": topic,
                        "max_results": max_results,
                        "days": days,
                        "include_answer": False,
                        "include_raw_content": False,
                    },
                )
                r.raise_for_status()
                body = r.json()
        except Exception as exc:
            log.warning("tavily.search_failed", query=query, error=str(exc))
            return []

        results = body.get("results") or []
        norm: list[dict[str, Any]] = []
        for item in results:
            norm.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                    "content": str(item.get("content") or "").strip(),
                    "published_date": item.get("published_date"),
                    "score": item.get("score"),
                    "source": _domain_from_url(item.get("url") or ""),
                }
            )
        _cache[cache_key] = (time.monotonic(), norm)
        log.info("tavily.search.ok", query=query, count=len(norm))
        return norm


async def search_stock_news(
    ticker: str,
    *,
    max_results: int = 3,
    days: int = 7,
    bypass_cache: bool = False,
) -> list[dict[str, Any]]:
    """Convenience wrapper — searches for news about a stock ticker."""
    # Strip Korean/global suffixes for the search query so headlines find
    # the company by ticker symbol or name.
    bare = ticker.replace(".KS", "").replace(".KQ", "").upper()
    return await search_news(
        f"{bare} stock news",
        max_results=max_results,
        days=days,
        bypass_cache=bypass_cache,
    )


# ── Helpers ────────────────────────────────────────────────────────────


def _domain_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        from urllib.parse import urlparse

        host = urlparse(url).netloc
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def format_for_prompt(items: list[dict[str, Any]], *, label: str = "최근 뉴스") -> str:
    """Render a small text block to splice into the agent system prompt."""
    if not items:
        return ""
    lines: list[str] = [f"[{label} (Tavily)]"]
    for it in items:
        date = (it.get("published_date") or "")[:10]
        suffix = f" ({date})" if date else ""
        snippet = (it.get("content") or "").replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        lines.append(f"- {it.get('title', '(제목 없음)')}{suffix}")
        if snippet:
            lines.append(f"  {snippet}")
        if it.get("url"):
            lines.append(f"  {it['url']}")
    return "\n".join(lines)
