"""End-to-end tests for apps.consolidator.runner."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from apps.consolidator.runner import ConsolidationReport, run_consolidation
from apps.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _fake_client(facts: list[dict] | None = None, events: list[dict] | None = None) -> MagicMock:
    """Build mock GeminiClient returning canned extraction JSON."""
    payload = json.dumps({"facts": facts or [], "events": events or []})
    client = MagicMock()
    client._client = MagicMock()
    resp = MagicMock()
    resp.text = payload
    client._client.models.generate_content.return_value = resp
    return client


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "test.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_consolidation_no_episodes(store: MemoryStore, tmp_path: Path) -> None:
    """Empty DB → report with zero episodes_processed, no errors."""
    report = asyncio.run(
        run_consolidation(
            store,
            log_dir=tmp_path / "logs",
            dry_run=True,
        )
    )
    assert isinstance(report, ConsolidationReport)
    assert report.episodes_processed == 0
    assert report.facts_promoted == 0
    assert report.events_added == 0
    assert report.errors == []


def test_run_consolidation_dry_run_skips_llm(store: MemoryStore, tmp_path: Path) -> None:
    """Dry-run mode: episodes processed but no LLM call."""
    now = _now()
    for i in range(3):
        store.add_episode(
            summary=f"어머니와 대화 {i}",
            primary_channel="text",
            when_at=_iso(now - timedelta(hours=i + 1)),
        )

    client = _fake_client(
        facts=[
            {
                "subject_person_name": "어머니",
                "predicate": "좋아한다",
                "object": "단호박",
                "confidence": 0.9,
            }
        ]
    )

    report = asyncio.run(
        run_consolidation(
            store,
            now_iso=_iso(now),
            log_dir=tmp_path / "logs",
            client=client,
            dry_run=True,
        )
    )
    assert report.episodes_processed == 3
    # LLM must NOT have been called
    client._client.models.generate_content.assert_not_called()
    assert report.facts_promoted == 0  # no LLM → no facts


def test_run_consolidation_promotes_facts(store: MemoryStore, tmp_path: Path) -> None:
    """Happy path: 3 recent episodes, mocked LLM returns 2 high-confidence facts."""
    now = _now()
    pid = store.add_person("어머니")

    for i in range(3):
        store.add_episode(
            summary=f"대화 요약 {i}",
            primary_channel="text",
            when_at=_iso(now - timedelta(hours=i + 1)),
        )

    client = _fake_client(
        facts=[
            {
                "subject_person_name": "어머니",
                "predicate": "좋아한다",
                "object": "단호박 케이크",
                "confidence": 0.9,
            },
            {
                "subject_person_name": "어머니",
                "predicate": "알러지",
                "object": "땅콩",
                "confidence": 0.85,
            },
        ]
    )

    report = asyncio.run(
        run_consolidation(
            store,
            now_iso=_iso(now),
            log_dir=tmp_path / "logs",
            client=client,
        )
    )
    assert report.episodes_processed == 3
    assert report.facts_extracted == 2
    assert report.facts_promoted == 2
    assert report.facts_archived == 0
    assert report.events_added == 0
    assert report.errors == []

    active = store.list_active_facts(pid)
    assert len(active) == 2


def test_run_consolidation_archives_conflict(store: MemoryStore, tmp_path: Path) -> None:
    """Conflicting fact → old archived, new added."""
    now = _now()
    pid = store.add_person("어머니")
    old_fact_id = store.add_fact(pid, "좋아한다", "단호박 케이크", confidence=0.9)

    store.add_episode(
        summary="어머니가 이제 초콜릿 케이크를 좋아한다고 하셨어요.",
        primary_channel="text",
        when_at=_iso(now - timedelta(hours=1)),
    )

    client = _fake_client(
        facts=[
            {
                "subject_person_name": "어머니",
                "predicate": "좋아한다",
                "object": "초콜릿 케이크",  # different object → conflict
                "confidence": 0.88,
            }
        ]
    )

    report = asyncio.run(
        run_consolidation(
            store,
            now_iso=_iso(now),
            log_dir=tmp_path / "logs",
            client=client,
        )
    )
    assert report.facts_promoted == 1
    assert report.facts_archived == 1
    assert report.errors == []

    active = store.list_active_facts(pid)
    assert len(active) == 1
    assert active[0].object == "초콜릿 케이크"

    # Old fact exists but archived
    row = store.conn.execute(
        "SELECT archived_at FROM facts WHERE id = ?", (old_fact_id,)
    ).fetchone()
    assert row["archived_at"] is not None


def test_run_consolidation_skips_old_episodes(store: MemoryStore, tmp_path: Path) -> None:
    """Episodes older than lookback_hours are excluded."""
    now = _now()
    # Old episode (30 hours ago — outside 24h window)
    store.add_episode(
        summary="오래된 대화",
        primary_channel="text",
        when_at=_iso(now - timedelta(hours=30)),
    )
    # Recent episode (2 hours ago)
    store.add_episode(
        summary="최근 대화",
        primary_channel="text",
        when_at=_iso(now - timedelta(hours=2)),
    )

    client = _fake_client(facts=[], events=[])

    report = asyncio.run(
        run_consolidation(
            store,
            now_iso=_iso(now),
            lookback_hours=24,
            log_dir=tmp_path / "logs",
            client=client,
        )
    )
    assert report.episodes_processed == 1  # only recent


def test_run_consolidation_adds_events(store: MemoryStore, tmp_path: Path) -> None:
    now = _now()
    store.add_episode(
        summary="다음 주 치과 예약 잡았어.",
        primary_channel="text",
        when_at=_iso(now - timedelta(hours=1)),
    )

    client = _fake_client(
        events=[
            {
                "person_name": None,
                "type": "appointment",
                "title": "치과 예약",
                "when_at": "2026-05-10T14:00:00",
                "recurrence": None,
            }
        ]
    )

    report = asyncio.run(
        run_consolidation(
            store,
            now_iso=_iso(now),
            log_dir=tmp_path / "logs",
            client=client,
        )
    )
    assert report.events_added == 1
    assert report.errors == []


def test_run_consolidation_writes_log_file(store: MemoryStore, tmp_path: Path) -> None:
    now = _now()
    store.add_episode(
        summary="테스트 대화",
        primary_channel="text",
        when_at=_iso(now - timedelta(hours=1)),
    )
    log_dir = tmp_path / "consolidation_log"

    report = asyncio.run(
        run_consolidation(
            store,
            now_iso=_iso(now),
            log_dir=log_dir,
            dry_run=True,
        )
    )

    # Log file exists
    date_str = report.ran_at[:10]
    log_file = log_dir / f"{date_str}.json"
    assert log_file.exists()

    # Parseable JSON
    data = json.loads(log_file.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 1
    entry = data[-1]
    assert entry["episodes_processed"] == report.episodes_processed
    assert "ran_at" in entry


def test_run_consolidation_errors_on_partial_failure(store: MemoryStore, tmp_path: Path) -> None:
    """Low-confidence or unknown-person facts don't raise; report stays clean."""
    now = _now()
    store.add_episode(
        summary="미지의 사람과 대화",
        primary_channel="text",
        when_at=_iso(now - timedelta(hours=1)),
    )

    # Returns a fact for unknown person (no person in DB)
    client = _fake_client(
        facts=[
            {
                "subject_person_name": "모르는사람",
                "predicate": "직업",
                "object": "의사",
                "confidence": 0.95,
            }
        ]
    )

    report = asyncio.run(
        run_consolidation(
            store,
            now_iso=_iso(now),
            log_dir=tmp_path / "logs",
            client=client,
        )
    )
    # Unknown person → skipped, not an error
    assert report.facts_promoted == 0
    assert report.errors == []


def test_consolidation_report_is_dataclass(store: MemoryStore, tmp_path: Path) -> None:
    report = asyncio.run(
        run_consolidation(store, log_dir=tmp_path / "logs", dry_run=True)
    )
    assert hasattr(report, "ran_at")
    assert hasattr(report, "episodes_processed")
    assert hasattr(report, "facts_extracted")
    assert hasattr(report, "facts_promoted")
    assert hasattr(report, "facts_archived")
    assert hasattr(report, "events_added")
    assert hasattr(report, "errors")
