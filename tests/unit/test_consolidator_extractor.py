"""Tests for apps.consolidator.extractor."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from apps.consolidator.extractor import (
    BATCH_SIZE,
    ExtractionResult,
    ExtractedEvent,
    ExtractedFact,
    extract_facts_and_events,
    _build_episodes_text,
    _parse_extraction_response,
)
from apps.memory.store import Episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(ep_id: int = 1, summary: str = "테스트 대화") -> Episode:
    return Episode(
        id=ep_id,
        when_at="2026-04-30T10:00:00+00:00",
        summary=summary,
        primary_channel="text",
    )


def _fake_client(responses: list[str]) -> MagicMock:
    """Build a fake GeminiClient that returns canned JSON strings."""
    client = MagicMock()
    client._client = MagicMock()

    call_count = {"n": 0}

    def _generate_content(*args, **kwargs):  # noqa: ANN001
        idx = call_count["n"]
        call_count["n"] += 1
        resp = MagicMock()
        resp.text = responses[idx % len(responses)]
        return resp

    client._client.models.generate_content.side_effect = _generate_content
    # Also set up .generate() fallback
    client.generate.side_effect = lambda msgs: responses[call_count["n"] - 1]
    return client


# ---------------------------------------------------------------------------
# _parse_extraction_response
# ---------------------------------------------------------------------------


def test_parse_valid_facts():
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
        }
    )
    facts, events = _parse_extraction_response(payload)
    assert len(facts) == 1
    f = facts[0]
    assert f.subject_person_name == "어머니"
    assert f.predicate == "좋아한다"
    assert f.object == "단호박 케이크"
    assert f.confidence == pytest.approx(0.9)
    assert events == []


def test_parse_valid_events():
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
        }
    )
    facts, events = _parse_extraction_response(payload)
    assert len(events) == 1
    e = events[0]
    assert e.person_name == "아내"
    assert e.type == "appointment"
    assert e.title == "치과 예약"
    assert e.when_at == "2026-05-10T14:00:00"
    assert e.recurrence is None
    assert facts == []


def test_parse_empty_response():
    payload = json.dumps({"facts": [], "events": []})
    facts, events = _parse_extraction_response(payload)
    assert facts == []
    assert events == []


def test_parse_fact_with_null_subject():
    payload = json.dumps(
        {
            "facts": [
                {
                    "subject_person_name": None,
                    "predicate": "날씨",
                    "object": "맑음",
                    "confidence": 0.8,
                }
            ],
            "events": [],
        }
    )
    facts, _ = _parse_extraction_response(payload)
    assert len(facts) == 1
    assert facts[0].subject_person_name is None


def test_parse_skips_missing_required_fields():
    payload = json.dumps(
        {
            "facts": [
                # missing "object"
                {
                    "subject_person_name": "아빠",
                    "predicate": "직업",
                    "confidence": 0.9,
                }
            ],
            "events": [
                # missing "when_at"
                {"person_name": "엄마", "type": "birthday", "title": "엄마 생일"}
            ],
        }
    )
    facts, events = _parse_extraction_response(payload)
    assert facts == []
    assert events == []


def test_parse_confidence_clamped():
    payload = json.dumps(
        {
            "facts": [
                {
                    "predicate": "x",
                    "object": "y",
                    "confidence": 1.5,  # over 1.0
                }
            ],
            "events": [],
        }
    )
    facts, _ = _parse_extraction_response(payload)
    assert facts[0].confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# extract_facts_and_events (integration of batch + parse)
# ---------------------------------------------------------------------------


def test_extract_canned_response():
    episodes = [_make_episode(1, "어머니가 단호박 케이크를 좋아한다고 하셨어요.")]
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
        }
    )
    client = _fake_client([payload])
    result: ExtractionResult = asyncio.run(
        extract_facts_and_events(episodes, client=client)
    )
    assert len(result.facts) == 1
    assert result.facts[0].predicate == "좋아한다"
    assert result.errors == []


def test_extract_empty_episodes():
    client = _fake_client([])
    result = asyncio.run(
        extract_facts_and_events([], client=client)
    )
    assert result.facts == []
    assert result.events == []
    assert result.errors == []
    # LLM should not have been called
    client._client.models.generate_content.assert_not_called()


def test_extract_malformed_json_retries_then_errors():
    """First call returns malformed JSON; second also bad → error reported."""
    client = _fake_client(["not valid json", "still bad {"])
    episodes = [_make_episode()]
    result = asyncio.run(
        extract_facts_and_events(episodes, client=client)
    )
    # Should have accumulated an error, not raised
    assert len(result.errors) == 1
    assert "JSON" in result.errors[0]
    assert result.facts == []


def test_extract_malformed_first_then_valid():
    """First call bad JSON, second (retry) valid → succeeds."""
    valid_payload = json.dumps(
        {
            "facts": [
                {"predicate": "좋아한다", "object": "초콜릿", "confidence": 0.9}
            ],
            "events": [],
        }
    )
    # Need to track calls manually since retry happens inside _extract_batch
    call_count = {"n": 0}
    client = MagicMock()
    client._client = MagicMock()

    def _gen(*args, **kwargs):  # noqa: ANN001
        call_count["n"] += 1
        resp = MagicMock()
        if call_count["n"] == 1:
            resp.text = "BAD JSON HERE"
        else:
            resp.text = valid_payload
        return resp

    client._client.models.generate_content.side_effect = _gen

    episodes = [_make_episode()]
    result = asyncio.run(
        extract_facts_and_events(episodes, client=client)
    )
    assert len(result.facts) == 1
    assert result.errors == []


def test_extract_batches_multiple_episodes():
    """More than BATCH_SIZE episodes → multiple LLM calls, results merged."""
    n_episodes = BATCH_SIZE + 3
    episodes = [_make_episode(i, f"요약 {i}") for i in range(n_episodes)]

    single_fact = {
        "subject_person_name": "아빠",
        "predicate": "직업",
        "object": "의사",
        "confidence": 0.9,
    }
    payload = json.dumps({"facts": [single_fact], "events": []})

    # Need separate response per batch call
    n_batches = (n_episodes + BATCH_SIZE - 1) // BATCH_SIZE
    client = _fake_client([payload] * n_batches)

    result = asyncio.run(
        extract_facts_and_events(episodes, client=client)
    )
    # Each of the 2 batches returns 1 fact → total 2
    assert len(result.facts) == n_batches
    assert result.errors == []
