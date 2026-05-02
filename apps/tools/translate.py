"""Lightweight Korean translation for news widget content.

Uses gemini-2.5-flash-lite — cheapest current Gemini model and plenty
for headline + snippet translation. One batched call per Tavily fetch
(all items at once) so the per-refresh cost stays in the 0.001 USD
range.

Failure modes (return original items, never raise):
  - GEMINI_API_KEY missing
  - genai client init fails
  - JSON parse fails
  - timeout
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_TRANSLATE_MODEL = "gemini-2.5-flash-lite"
_TIMEOUT_S = 6.0
_MAX_CONTENT_CHARS = 320  # cap each snippet to keep prompts bounded

_PROMPT = """\
다음 영문 뉴스 헤드라인과 본문 스니펫을 자연스러운 한국어로 번역해 주세요.
- 원문이 이미 한국어이면 그대로 두세요 (번역 금지).
- 사람 이름·회사 이름·티커 (TSLA, AAPL 등) 는 그대로 둡니다.
- 헤드라인은 신문 톤, 본문은 평범한 정보 톤.
- 번역 외 다른 텍스트 금지.

JSON 입력:
{payload}

응답: 엄격한 JSON 배열. 입력의 id 와 정확히 매칭되도록 출력.
스키마: [{{"id": <int>, "title": "...", "content": "..."}}, ...]
"""


_client_cache: object | None = None


def _client() -> object | None:
    """Lazy-init genai client. Returns None if API key missing."""
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from google import genai

        _client_cache = genai.Client(api_key=api_key)
    except Exception as exc:
        log.warning("translate.client_init_failed", error=str(exc))
        return None
    return _client_cache


async def translate_news_to_korean(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate the title + content fields on each news item to Korean.

    Returns a new list. Original items unchanged. Adds back any fields
    that aren't part of the translation (url, source, published_date,
    score) untouched.
    """
    if not items:
        return items
    cli = _client()
    if cli is None:
        return items

    payload: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        payload.append(
            {
                "id": i,
                "title": (it.get("title") or "")[:240],
                "content": (it.get("content") or "")[:_MAX_CONTENT_CHARS],
            }
        )

    prompt = _PROMPT.format(payload=json.dumps(payload, ensure_ascii=False))

    try:
        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["id", "title", "content"],
                },
            },
        )
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                cli.models.generate_content,  # type: ignore[attr-defined]
                model=_TRANSLATE_MODEL,
                contents=prompt,
                config=config,
            ),
            timeout=_TIMEOUT_S,
        )
        raw = resp.text or "[]"
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
        log.warning("translate.failed", error=str(exc), count=len(items))
        return items

    try:
        translated = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("translate.parse_failed", error=str(exc), raw=str(raw)[:200])
        return items

    by_id: dict[int, dict[str, Any]] = {}
    if isinstance(translated, list):
        for entry in translated:
            try:
                by_id[int(entry["id"])] = entry
            except (KeyError, TypeError, ValueError):
                continue

    out: list[dict[str, Any]] = []
    for i, original in enumerate(items):
        merged = dict(original)
        t = by_id.get(i)
        if t:
            if t.get("title"):
                merged["title"] = t["title"]
            if t.get("content"):
                merged["content"] = t["content"]
        out.append(merged)
    log.info("translate.ok", count=len(out))
    return out
