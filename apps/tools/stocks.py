"""Stocks tool — quotes routed by ticker shape:
  - Korean tickers (6-digit numeric, optionally with .KS/.KQ) → Kiwoom REST API
  - Everything else → Finnhub (Polygon used as fallback when key missing)

Public surface:
    get_quotes(tickers: list[str]) -> list[Quote (dict)]

Each returned dict has:
    {ticker, name, price, change, change_pct, currency, source, error?}

In-memory cache with TTL keeps API calls bounded across rapid widget refreshes.
Keys are read from env: FINNHUB_API_KEY, POLYGON_API_KEY,
KIWOOM_APP_KEY, KIWOOM_APP_SECRET, KIWOOM_BASE_URL (optional).

When no provider key is configured for a given ticker shape, the quote
entry contains `error` so the UI can show a "키 미설정" hint without
breaking the rest of the row.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

# ── cache ──────────────────────────────────────────────────────────────

_CACHE_TTL_S = 60.0
_quote_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = asyncio.Lock()

# Kiwoom OAuth token cache: (token, expires_at_monotonic)
_kiwoom_token: tuple[str, float] | None = None
_kiwoom_token_lock = asyncio.Lock()


# ── ticker classification ──────────────────────────────────────────────

_KR_NUMERIC = re.compile(r"^\d{6}$")
_KR_SUFFIX = re.compile(r"^(\d{6})\.(KS|KQ)$", re.IGNORECASE)


def classify(ticker: str) -> tuple[str, str]:
    """Return (market, code) — market in {'kr', 'global'}.

    'kr' covers KOSPI/KOSDAQ tickers; the 6-digit code is returned without
    the .KS/.KQ suffix so it can be passed straight to Kiwoom.
    """
    t = ticker.strip().upper()
    if _KR_NUMERIC.fullmatch(t):
        return "kr", t
    m = _KR_SUFFIX.fullmatch(t)
    if m:
        return "kr", m.group(1)
    return "global", t


# ── public API ─────────────────────────────────────────────────────────


async def search_symbols(query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    """Symbol/company-name search via Finnhub. Cross-market (US, KR with .KS,
    JP with .T, etc.). Returns up to *limit* matches as dicts:
        {symbol, display_symbol, name, type}
    Empty list when no key, empty query, or API failure.
    """
    q = query.strip()
    if not q:
        return []
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not finnhub_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=4.0) as cli:
            r = await cli.get(
                "https://finnhub.io/api/v1/search",
                params={"q": q, "token": finnhub_key},
            )
            r.raise_for_status()
            body = r.json()
    except Exception as exc:
        log.warning("stocks.search_failed", query=q, error=str(exc))
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in body.get("result") or []:
        sym = str(item.get("symbol") or "").strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(
            {
                "symbol": sym,
                "display_symbol": str(item.get("displaySymbol") or sym),
                "name": str(item.get("description") or "").strip(),
                "type": str(item.get("type") or "").strip(),
            }
        )
        if len(out) >= limit:
            break
    return out


async def get_quotes(tickers: list[str]) -> list[dict[str, Any]]:
    """Fetch quotes for many tickers, dispatching per-provider.

    Concurrent per ticker; cached per ticker with TTL. Failures yield an
    `error` field on the row but don't drop other tickers.
    """
    seen: dict[str, dict[str, Any]] = {}
    coros: list[tuple[str, asyncio.Task[dict[str, Any]]]] = []
    now = time.monotonic()

    for raw in tickers:
        t = raw.strip().upper()
        if not t or t in seen:
            continue
        cached = _quote_cache.get(t)
        if cached and (now - cached[0]) < _CACHE_TTL_S:
            seen[t] = cached[1]
            continue
        coros.append((t, asyncio.create_task(_fetch_one(t))))

    for t, task in coros:
        try:
            quote = await task
        except Exception as exc:
            log.warning("stocks.fetch_failed", ticker=t, error=str(exc))
            quote = {"ticker": t, "error": str(exc)}
        seen[t] = quote
        async with _cache_lock:
            _quote_cache[t] = (time.monotonic(), quote)

    return [seen[t.strip().upper()] for t in tickers if t.strip()]


# ── per-provider fetch ─────────────────────────────────────────────────


async def _fetch_one(ticker: str) -> dict[str, Any]:
    market, code = classify(ticker)
    if market == "kr":
        return await _fetch_kiwoom(ticker, code)
    return await _fetch_global(ticker)


async def _fetch_global(ticker: str) -> dict[str, Any]:
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    polygon_key = os.environ.get("POLYGON_API_KEY", "").strip()

    if finnhub_key:
        try:
            return await _fetch_finnhub(ticker, finnhub_key)
        except Exception as exc:
            log.warning("stocks.finnhub_failed", ticker=ticker, error=str(exc))
            if not polygon_key:
                return {"ticker": ticker, "error": f"finnhub: {exc}"}

    if polygon_key:
        try:
            return await _fetch_polygon(ticker, polygon_key)
        except Exception as exc:
            log.warning("stocks.polygon_failed", ticker=ticker, error=str(exc))
            return {"ticker": ticker, "error": f"polygon: {exc}"}

    return {"ticker": ticker, "error": "FINNHUB_API_KEY / POLYGON_API_KEY 미설정"}


async def _fetch_finnhub(ticker: str, key: str) -> dict[str, Any]:
    """Finnhub /quote endpoint. Free-tier covers US + many global symbols."""
    async with httpx.AsyncClient(timeout=4.0) as cli:
        r = await cli.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": ticker, "token": key},
        )
        r.raise_for_status()
        body = r.json()

    price = float(body.get("c") or 0)
    if price == 0:
        # Finnhub returns 0 for unknown symbols — treat as error
        raise ValueError(f"unknown symbol or no data: {ticker}")
    change = float(body.get("d") or 0)
    change_pct = float(body.get("dp") or 0)

    name = await _finnhub_name(ticker, key)

    return {
        "ticker": ticker,
        "name": name,
        "price": price,
        "change": change,
        "change_pct": change_pct,
        "currency": "USD",
        "source": "finnhub",
    }


async def _finnhub_name(ticker: str, key: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=3.0) as cli:
            r = await cli.get(
                "https://finnhub.io/api/v1/stock/profile2",
                params={"symbol": ticker, "token": key},
            )
            if r.status_code == 200:
                return (r.json() or {}).get("name") or ""
    except Exception:
        pass
    return ""


async def _fetch_polygon(ticker: str, key: str) -> dict[str, Any]:
    """Polygon previous-day close (works on all tiers)."""
    async with httpx.AsyncClient(timeout=4.0) as cli:
        r = await cli.get(
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev",
            params={"apiKey": key, "adjusted": "true"},
        )
        r.raise_for_status()
        body = r.json()
    results = body.get("results") or []
    if not results:
        raise ValueError(f"no data for {ticker}")
    row = results[0]
    close = float(row.get("c") or 0)
    open_ = float(row.get("o") or 0)
    change = close - open_
    change_pct = (change / open_ * 100.0) if open_ else 0.0
    return {
        "ticker": ticker,
        "name": "",
        "price": close,
        "change": change,
        "change_pct": change_pct,
        "currency": "USD",
        "source": "polygon",
    }


# ── Kiwoom ─────────────────────────────────────────────────────────────


def _kiwoom_base_url() -> str:
    return os.environ.get("KIWOOM_BASE_URL", "https://api.kiwoom.com").rstrip("/")


async def _kiwoom_get_token() -> str:
    """OAuth client_credentials. Cached until ~5 min before expiry."""
    global _kiwoom_token
    now = time.monotonic()
    if _kiwoom_token and now < _kiwoom_token[1]:
        return _kiwoom_token[0]

    appkey = os.environ.get("KIWOOM_APP_KEY", "").strip()
    secret = os.environ.get("KIWOOM_APP_SECRET", "").strip()
    if not appkey or not secret:
        raise ValueError("KIWOOM_APP_KEY / KIWOOM_APP_SECRET 미설정")

    async with _kiwoom_token_lock:
        if _kiwoom_token and time.monotonic() < _kiwoom_token[1]:
            return _kiwoom_token[0]
        async with httpx.AsyncClient(timeout=5.0) as cli:
            r = await cli.post(
                f"{_kiwoom_base_url()}/oauth2/token",
                json={
                    "grant_type": "client_credentials",
                    "appkey": appkey,
                    "secretkey": secret,
                },
                headers={"Content-Type": "application/json;charset=UTF-8"},
            )
            r.raise_for_status()
            body = r.json()
        token = body.get("token") or body.get("access_token")
        if not token:
            raise ValueError(f"kiwoom token missing in response: {body}")
        expires_in = float(body.get("expires_dt") or body.get("expires_in") or 3600)
        # The expires_dt format from Kiwoom is sometimes a stringified datetime
        # not seconds; clamp to 1h to stay safe.
        if expires_in > 86400 or expires_in < 60:
            expires_in = 3600
        _kiwoom_token = (token, time.monotonic() + expires_in - 300)
        return token


async def _fetch_kiwoom(ticker: str, code: str) -> dict[str, Any]:
    """Kiwoom 주식기본정보요청 (api-id: ka10001)."""
    try:
        token = await _kiwoom_get_token()
    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}

    async with httpx.AsyncClient(timeout=5.0) as cli:
        r = await cli.post(
            f"{_kiwoom_base_url()}/api/dostk/stkinfo",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json;charset=UTF-8",
                "api-id": "ka10001",
            },
            json={"stk_cd": code},
        )
        r.raise_for_status()
        body = r.json()

    rt = body.get("return_code")
    if rt not in (0, "0", None):
        msg = body.get("return_msg") or f"kiwoom error code {rt}"
        return {"ticker": ticker, "error": msg}

    name = body.get("stk_nm") or ""
    price = _kiwoom_num(body.get("cur_prc"))
    change = _kiwoom_num(body.get("pred_pre"))
    change_pct = _kiwoom_num(body.get("flu_rt"))

    if price is None:
        return {"ticker": ticker, "error": "Kiwoom 응답에 시세가 없어요"}

    return {
        "ticker": ticker,
        "name": name,
        "price": price,
        "change": change or 0,
        "change_pct": change_pct or 0,
        "currency": "KRW",
        "source": "kiwoom",
    }


def _kiwoom_num(v: Any) -> float | None:
    """Kiwoom returns numbers as strings, often with leading +/- and signs."""
    if v is None or v == "":
        return None
    try:
        if isinstance(v, str):
            v = v.replace(",", "").lstrip("+")
        return float(v)
    except (TypeError, ValueError):
        return None
