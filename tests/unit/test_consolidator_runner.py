"""End-to-end tests for apps.consolidator.runner."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from apps.consolidator.runner import ConsolidationReport, run_consolidation
from apps.memory.store import MemoryStore


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _fake_client(
    facts: list[dict] | None = None,
    events: list[dict] | None = None,
    notes: list[dict] | None = None,
) -> MagicMock:
    payload = json.dumps(
        {"facts": facts or [], "events": events or [], "notes": notes or []}
    )
    client = MagicMock()
    client._client = MagicMock()
    resp = MagicMock()
    resp.text = payload
    client._client.models.generate_content.return_value = resp
    return client


def _seed_recent_session(store: MemoryStore, now: datetime, hours_ago: int) -> int:
    """Insert a session with last_active_at = now - hours_ago."""
    when = _iso(now - timedelta(hours=hours_ago))
    cur = store.conn.execute(
        "INSERT INTO sessions (started_at, last_active_at, summary) VALUES (?, ?, ?)",
        (when, when, f"세션 {hours_ago}h ago"),
    )
    sid = int(cur.lastrowid or 0)
    store.add_message(sid, "user", f"메시지 (sid={sid})")
    # add_message bumps last_active_at to CURRENT_TIMESTAMP, so reset.
    store.conn.execute(
        "UPDATE sessions SET last_active_at = ? WHERE id = ?", (when, sid)
    )
    return sid


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(tmp_path / "test.db")
    try:
        yield s
    finally:
        s.close()


def test_run_consolidation_no_sessions(store: MemoryStore, tmp_path: Path) -> None:
    report = asyncio.run(
        run_consolidation(store, log_dir=tmp_path / "logs", dry_run=True)
    )
    assert isinstance(report, ConsolidationReport)
    assert report.sessions_processed == 0
    assert report.facts_promoted == 0
    assert report.events_added == 0
    assert report.notes_added == 0
    assert report.errors == []


def test_run_consolidation_dry_run_skips_llm(store: MemoryStore, tmp_path: Path) -> None:
    now = _now()
    for i in range(3):
        _seed_recent_session(store, now, hours_ago=i + 1)

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
    assert report.sessions_processed == 3
    client._client.models.generate_content.assert_not_called()
    assert report.facts_promoted == 0


def test_run_consolidation_promotes_facts(store: MemoryStore, tmp_path: Path) -> None:
    now = _now()
    pid = store.add_person("어머니")
    for i in range(3):
        _seed_recent_session(store, now, hours_ago=i + 1)

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
    assert report.sessions_processed == 3
    assert report.facts_extracted == 2
    assert report.facts_promoted == 2
    assert report.facts_archived == 0
    assert report.events_added == 0
    assert report.errors == []
    assert len(store.list_active_facts(pid)) == 2


def test_run_consolidation_archives_conflict(store: MemoryStore, tmp_path: Path) -> None:
    now = _now()
    pid = store.add_person("어머니")
    old_fact_id = store.add_fact(pid, "좋아한다", "단호박 케이크", confidence=0.9)
    _seed_recent_session(store, now, hours_ago=1)

    client = _fake_client(
        facts=[
            {
                "subject_person_name": "어머니",
                "predicate": "좋아한다",
                "object": "초콜릿 케이크",
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

    row = store.conn.execute(
        "SELECT archived_at FROM facts WHERE id = ?", (old_fact_id,)
    ).fetchone()
    assert row["archived_at"] is not None


def test_run_consolidation_skips_old_sessions(store: MemoryStore, tmp_path: Path) -> None:
    now = _now()
    _seed_recent_session(store, now, hours_ago=30)  # outside 24h
    _seed_recent_session(store, now, hours_ago=2)   # within 24h

    client = _fake_client()
    report = asyncio.run(
        run_consolidation(
            store,
            now_iso=_iso(now),
            lookback_hours=24,
            log_dir=tmp_path / "logs",
            client=client,
        )
    )
    assert report.sessions_processed == 1


def test_run_consolidation_adds_events(store: MemoryStore, tmp_path: Path) -> None:
    now = _now()
    _seed_recent_session(store, now, hours_ago=1)

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


def test_run_consolidation_adds_notes(store: MemoryStore, tmp_path: Path) -> None:
    now = _now()
    _seed_recent_session(store, now, hours_ago=1)

    client = _fake_client(
        notes=[
            {"content": "매주 금요일 외식하기로 결정", "tags": ["routine"]},
            {"content": "워크숍 일정 챙기기", "tags": ["todo"]},
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
    assert report.notes_added == 2
    notes = store.list_notes()
    contents = [n.content for n in notes]
    assert "매주 금요일 외식하기로 결정" in contents
    assert "워크숍 일정 챙기기" in contents


def test_run_consolidation_writes_log_file(store: MemoryStore, tmp_path: Path) -> None:
    now = _now()
    _seed_recent_session(store, now, hours_ago=1)
    log_dir = tmp_path / "consolidation_log"

    report = asyncio.run(
        run_consolidation(
            store,
            now_iso=_iso(now),
            log_dir=log_dir,
            dry_run=True,
        )
    )

    log_file = log_dir / f"{report.ran_at[:10]}.json"
    assert log_file.exists()

    data = json.loads(log_file.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 1
    entry = data[-1]
    assert entry["sessions_processed"] == report.sessions_processed
    assert "ran_at" in entry


def test_run_consolidation_unknown_person_skipped_not_errored(
    store: MemoryStore, tmp_path: Path
) -> None:
    now = _now()
    _seed_recent_session(store, now, hours_ago=1)

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
    assert report.facts_promoted == 0
    assert report.errors == []


def test_consolidation_report_dataclass_fields(
    store: MemoryStore, tmp_path: Path
) -> None:
    report = asyncio.run(
        run_consolidation(store, log_dir=tmp_path / "logs", dry_run=True)
    )
    for attr in (
        "ran_at",
        "sessions_processed",
        "facts_extracted",
        "facts_promoted",
        "facts_archived",
        "events_added",
        "notes_added",
        "errors",
    ):
        assert hasattr(report, attr), f"missing {attr}"
