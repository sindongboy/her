"""Tests for apps.agent.recall.filter_by_relevance."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from apps.agent.recall import RecallContext, filter_by_relevance


def _ctx(*, facts=(), events=(), notes=(), sessions=()) -> RecallContext:
    return RecallContext(
        facts=list(facts),
        upcoming_events=list(events),
        notes=list(notes),
        sessions=list(sessions),
        attachments=[],
        attachment_ids=[],
    )


def _mock_flash(payload: dict[str, Any]) -> MagicMock:
    m = MagicMock()
    m.generate.return_value = json.dumps(payload)
    return m


def test_returns_unchanged_when_flash_is_none() -> None:
    ctx = _ctx(facts=[(1, "어머니", "좋아한다", "케이크")])
    out = asyncio.run(filter_by_relevance(ctx, "anything", None))
    assert out is ctx


def test_returns_unchanged_when_empty() -> None:
    ctx = _ctx()
    out = asyncio.run(filter_by_relevance(ctx, "x", _mock_flash({})))
    assert out is ctx


def test_keeps_only_matching_ids() -> None:
    ctx = _ctx(
        facts=[
            (1, "어머니", "좋아한다", "단호박 케이크"),
            (2, "어머니", "알러지", "땅콩"),
        ],
        events=[(10, "어머니 생신", "2099-06-15")],
        notes=[(20, "매주 금요일 외식")],
        sessions=[(30, "이전 어머니 생신 선물 고민", 0.9)],
    )
    flash = _mock_flash({
        "keep_facts": [1],          # cake yes
        "keep_events": [10],
        "keep_notes": [],           # not relevant
        "keep_sessions": [30],
    })
    out = asyncio.run(filter_by_relevance(ctx, "어머니 생신 선물", flash))
    assert [f[0] for f in out.facts] == [1]
    assert [e[0] for e in out.upcoming_events] == [10]
    assert out.notes == []
    assert [s[0] for s in out.sessions] == [30]


def test_falls_open_on_invalid_json() -> None:
    ctx = _ctx(facts=[(1, "어머니", "좋아한다", "케이크")])
    flash = MagicMock()
    flash.generate.return_value = "not json {{{"
    out = asyncio.run(filter_by_relevance(ctx, "x", flash))
    # fall open — original ctx returned
    assert out.facts == ctx.facts


def test_strips_json_code_fence() -> None:
    ctx = _ctx(
        facts=[(1, "어머니", "좋아한다", "케이크")],
        notes=[(2, "메모 1")],
    )
    flash = MagicMock()
    flash.generate.return_value = (
        '```json\n{"keep_facts":[1],"keep_events":[],"keep_notes":[],"keep_sessions":[]}\n```'
    )
    out = asyncio.run(filter_by_relevance(ctx, "x", flash))
    assert [f[0] for f in out.facts] == [1]
    assert out.notes == []


def test_attachments_always_pass_through() -> None:
    ctx = RecallContext(
        facts=[(1, "x", "y", "z")],
        upcoming_events=[],
        notes=[],
        sessions=[],
        attachments=[(99, "/tmp/a.txt", "어머니 사진")],
        attachment_ids=[99],
    )
    flash = _mock_flash({"keep_facts": [], "keep_events": [], "keep_notes": [], "keep_sessions": []})
    out = asyncio.run(filter_by_relevance(ctx, "x", flash))
    # attachments preserved even if facts dropped
    assert out.attachments == ctx.attachments
    assert out.attachment_ids == ctx.attachment_ids
    assert out.facts == []


def test_filter_drops_unrelated_person_facts() -> None:
    """Real-world scenario: user asks about a birthday. Recall pulls every
    fact about that person. Filter should drop facts unrelated to birthday."""
    ctx = _ctx(
        facts=[
            (1, "어머니", "좋아한다", "단호박 케이크"),  # relevant for gift
            (2, "어머니", "직업", "교사"),               # not relevant
            (3, "어머니", "혈액형", "A형"),               # not relevant
        ],
    )
    flash = _mock_flash({
        "keep_facts": [1],
        "keep_events": [],
        "keep_notes": [],
        "keep_sessions": [],
    })
    out = asyncio.run(filter_by_relevance(ctx, "어머니 생신 선물 추천", flash))
    assert {f[0] for f in out.facts} == {1}
