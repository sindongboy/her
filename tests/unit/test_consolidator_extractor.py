"""Tests for apps.consolidator.extractor."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from apps.consolidator.extractor import (
    BATCH_SIZE,
    ExtractionResult,
    extract_facts_and_events,
    _build_sessions_text,
    _parse_extraction_response,
)
from apps.memory.store import Session


def _make_session(sid: int = 1, summary: str = "테스트 대화") -> Session:
    return Session(
        id=sid,
        started_at="2026-04-30T10:00:00+00:00",
        last_active_at="2026-04-30T10:00:00+00:00",
        title=None,
        summary=summary,
        archived_at=None,
    )


def _fake_client(responses: list[str]) -> MagicMock:
    client = MagicMock()
    client._client = MagicMock()

    call_count = {"n": 0}

    def _generate_content(*args, **kwargs):  # noqa: ANN001
        idx = call_count["n"]
        call_count["n"] += 1
        resp = MagicMock()
        resp.text = responses[idx % len(responses)] if responses else ""
        return resp

    client._client.models.generate_content.side_effect = _generate_content
    client.generate.side_effect = lambda msgs: (
        responses[call_count["n"] - 1] if responses else ""
    )
    return client


# ── _parse_extraction_response ────────────────────────────────────────────


def test_parse_valid_facts() -> None:
    payload = json.dumps(
        {
            "facts": [
                {
                    "subject_person_name": "어머니",
                    "predicate": "좋아한다",
                    "object": "단호박 케이크",
                    "confidence": 0.9,
                }
            ],
            "events": [],
            "notes": [],
        }
    )
    facts, events, notes = _parse_extraction_response(payload)
    assert len(facts) == 1
    f = facts[0]
    assert f.subject_person_name == "어머니"
    assert f.predicate == "좋아한다"
    assert f.object == "단호박 케이크"
    assert f.confidence == pytest.approx(0.9)
    assert events == []
    assert notes == []


def test_parse_valid_events() -> None:
    payload = json.dumps(
        {
            "facts": [],
            "events": [
                {
                    "person_name": "아내",
                    "type": "appointment",
                    "title": "치과 예약",
                    "when_at": "2026-05-10T14:00:00",
                    "recurrence": None,
                }
            ],
            "notes": [],
        }
    )
    _, events, _ = _parse_extraction_response(payload)
    assert len(events) == 1
    e = events[0]
    assert e.person_name == "아내"
    assert e.type == "appointment"
    assert e.title == "치과 예약"
    assert e.when_at == "2026-05-10T14:00:00"
    assert e.recurrence is None


def test_parse_valid_notes() -> None:
    payload = json.dumps(
        {
            "facts": [],
            "events": [],
            "notes": [
                {"content": "매주 금요일 외식하기로 함", "tags": ["routine"]},
                {"content": "아내 워크숍 일정 확인", "tags": []},
            ],
        }
    )
    _, _, notes = _parse_extraction_response(payload)
    assert len(notes) == 2
    assert notes[0].content == "매주 금요일 외식하기로 함"
    assert notes[0].tags == ["routine"]
    assert notes[1].tags == []


def test_parse_empty_response() -> None:
    payload = json.dumps({"facts": [], "events": [], "notes": []})
    facts, events, notes = _parse_extraction_response(payload)
    assert facts == [] and events == [] and notes == []


def test_parse_skips_missing_required_fields() -> None:
    payload = json.dumps(
        {
            "facts": [
                {"subject_person_name": "아빠", "predicate": "직업", "confidence": 0.9}
            ],
            "events": [{"person_name": "엄마", "type": "birthday", "title": "엄마 생일"}],
            "notes": [{"content": "   ", "tags": []}],
        }
    )
    facts, events, notes = _parse_extraction_response(payload)
    assert facts == []
    assert events == []
    assert notes == []


def test_parse_confidence_clamped() -> None:
    payload = json.dumps(
        {
            "facts": [{"predicate": "x", "object": "y", "confidence": 1.5}],
            "events": [],
            "notes": [],
        }
    )
    facts, _, _ = _parse_extraction_response(payload)
    assert facts[0].confidence == pytest.approx(1.0)


# ── _build_sessions_text ──────────────────────────────────────────────────


def test_build_sessions_text_includes_session_id_and_content() -> None:
    s = _make_session(7, "어머니 케이크 이야기")
    blob = _build_sessions_text([(s, "사용자: 안녕\n비서: 네")])
    assert "session_id=7" in blob
    assert "사용자: 안녕" in blob
    assert "비서: 네" in blob


def test_build_sessions_text_handles_empty_content() -> None:
    s = _make_session()
    blob = _build_sessions_text([(s, "")])
    assert "내용 없음" in blob


# ── extract_facts_and_events ──────────────────────────────────────────────


def test_extract_canned_response() -> None:
    sessions = [(_make_session(1, "어머니 단호박 이야기"), "사용자: 케이크 좋아하셔")]
    payload = json.dumps(
        {
            "facts": [
                {
                    "subject_person_name": "어머니",
                    "predicate": "좋아한다",
                    "object": "단호박 케이크",
                    "confidence": 0.85,
                }
            ],
            "events": [],
            "notes": [{"content": "케이크 단호박 비율 메모", "tags": ["food"]}],
        }
    )
    client = _fake_client([payload])
    result: ExtractionResult = asyncio.run(
        extract_facts_and_events(sessions, client=client)
    )
    assert len(result.facts) == 1
    assert result.facts[0].predicate == "좋아한다"
    assert len(result.notes) == 1
    assert result.errors == []


def test_extract_empty_sessions() -> None:
    client = _fake_client([])
    result = asyncio.run(extract_facts_and_events([], client=client))
    assert result.facts == [] and result.events == [] and result.notes == []
    client._client.models.generate_content.assert_not_called()


def test_extract_malformed_json_retries_then_errors() -> None:
    client = _fake_client(["not valid json", "still bad {"])
    sessions = [(_make_session(), "내용")]
    result = asyncio.run(extract_facts_and_events(sessions, client=client))
    assert len(result.errors) == 1
    assert "JSON" in result.errors[0]
    assert result.facts == []


def test_extract_malformed_first_then_valid() -> None:
    valid = json.dumps(
        {
            "facts": [{"predicate": "좋아한다", "object": "초콜릿", "confidence": 0.9}],
            "events": [],
            "notes": [],
        }
    )
    call_count = {"n": 0}
    client = MagicMock()
    client._client = MagicMock()

    def _gen(*args, **kwargs):  # noqa: ANN001
        call_count["n"] += 1
        resp = MagicMock()
        resp.text = "BAD JSON HERE" if call_count["n"] == 1 else valid
        return resp

    client._client.models.generate_content.side_effect = _gen

    sessions = [(_make_session(), "내용")]
    result = asyncio.run(extract_facts_and_events(sessions, client=client))
    assert len(result.facts) == 1
    assert result.errors == []


def test_extract_batches_multiple_sessions() -> None:
    n = BATCH_SIZE + 3
    sessions = [(_make_session(i, f"요약 {i}"), f"사용자: msg {i}") for i in range(n)]

    payload = json.dumps(
        {
            "facts": [
                {
                    "subject_person_name": "아빠",
                    "predicate": "직업",
                    "object": "의사",
                    "confidence": 0.9,
                }
            ],
            "events": [],
            "notes": [],
        }
    )
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    client = _fake_client([payload] * n_batches)

    result = asyncio.run(extract_facts_and_events(sessions, client=client))
    assert len(result.facts) == n_batches
    assert result.errors == []
