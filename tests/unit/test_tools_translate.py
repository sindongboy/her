"""Tests for apps.tools.translate."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apps.tools import translate


@pytest.fixture(autouse=True)
def _reset_client():
    translate._client_cache = None
    yield
    translate._client_cache = None


def _items(*pairs: tuple[str, str]) -> list[dict[str, Any]]:
    return [
        {"title": t, "content": c, "url": f"https://x/{i}", "source": "x"}
        for i, (t, c) in enumerate(pairs)
    ]


def test_returns_unchanged_when_empty() -> None:
    out = asyncio.run(translate.translate_news_to_korean([]))
    assert out == []


def test_returns_unchanged_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    items = _items(("Foo", "Bar"))
    out = asyncio.run(translate.translate_news_to_korean(items))
    assert out == items


def test_returns_unchanged_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "X")

    fake = MagicMock()
    fake.models.generate_content.side_effect = TimeoutError()
    translate._client_cache = fake

    items = _items(("A", "B"))
    out = asyncio.run(translate.translate_news_to_korean(items))
    assert out == items


def test_returns_unchanged_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "X")

    resp = MagicMock(); resp.text = "not json {{"
    fake = MagicMock(); fake.models.generate_content.return_value = resp
    translate._client_cache = fake

    items = _items(("A", "B"))
    out = asyncio.run(translate.translate_news_to_korean(items))
    assert out == items


def test_merges_translation_back_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "X")

    payload = json.dumps([
        {"id": 0, "title": "테슬라 실적 호조", "content": "본문 번역 1"},
        {"id": 1, "title": "엔비디아 5% 상승", "content": "본문 번역 2"},
    ])
    resp = MagicMock(); resp.text = payload
    fake = MagicMock(); fake.models.generate_content.return_value = resp
    translate._client_cache = fake

    items = _items(("Tesla beats earnings", "Body 1"), ("NVDA up 5%", "Body 2"))
    out = asyncio.run(translate.translate_news_to_korean(items))

    assert out[0]["title"] == "테슬라 실적 호조"
    assert out[0]["content"] == "본문 번역 1"
    assert out[0]["url"] == "https://x/0"  # url preserved
    assert out[0]["source"] == "x"
    assert out[1]["title"] == "엔비디아 5% 상승"


def test_skips_missing_id_in_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the model only translates some entries, untouched items keep originals."""
    monkeypatch.setenv("GEMINI_API_KEY", "X")

    payload = json.dumps([{"id": 0, "title": "번역됨", "content": "본문"}])
    resp = MagicMock(); resp.text = payload
    fake = MagicMock(); fake.models.generate_content.return_value = resp
    translate._client_cache = fake

    items = _items(("Original 0", "Body 0"), ("Original 1", "Body 1"))
    out = asyncio.run(translate.translate_news_to_korean(items))

    assert out[0]["title"] == "번역됨"
    assert out[1]["title"] == "Original 1"
    assert out[1]["content"] == "Body 1"
