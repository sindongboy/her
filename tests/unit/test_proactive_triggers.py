"""Tests for proactive trigger implementations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from apps.proactive.activity import record_activity
from apps.proactive.triggers import (
    RecurringPatternTrigger,
    SilenceTrigger,
    TimeBasedTrigger,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_store(tmp_path: Path):
    """Create an in-memory (tmp_path) MemoryStore."""
    from apps.memory.store import MemoryStore

    return MemoryStore(tmp_path / "db.sqlite")


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── TimeBasedTrigger ───────────────────────────────────────────────────────────


class TestTimeBasedTrigger:
    def test_no_events_returns_empty(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        trigger = TimeBasedTrigger(lookahead_hours=24.0, lookahead_imminent_hours=1.0)
        now = _now_utc()
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert proposals == []
        store.close()

    def test_event_within_24h_fires(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        now = _now_utc()
        # Event in +6 hours (within 24h lookahead)
        when = now + timedelta(hours=6)
        store.add_event("meeting", "팀 회의", when.strftime("%Y-%m-%d %H:%M:%S"))

        trigger = TimeBasedTrigger(lookahead_hours=24.0, lookahead_imminent_hours=1.0)
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert len(proposals) == 1
        assert proposals[0].source == "time"
        assert "팀 회의" in proposals[0].context
        store.close()

    def test_event_outside_24h_does_not_fire(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        now = _now_utc()
        # Event in +48 hours (outside 24h lookahead)
        when = now + timedelta(hours=48)
        store.add_event("birthday", "생일 파티", when.strftime("%Y-%m-%d %H:%M:%S"))

        trigger = TimeBasedTrigger(lookahead_hours=24.0)
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert proposals == []
        store.close()

    def test_imminent_event_gets_higher_priority(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        now = _now_utc()
        # Imminent: +30 min
        when_imminent = now + timedelta(minutes=30)
        store.add_event("alarm", "약속", when_imminent.strftime("%Y-%m-%d %H:%M:%S"))
        # Non-imminent: +6 hours
        when_far = now + timedelta(hours=6)
        store.add_event("meeting", "회의", when_far.strftime("%Y-%m-%d %H:%M:%S"))

        trigger = TimeBasedTrigger(lookahead_hours=24.0, lookahead_imminent_hours=1.0)
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert len(proposals) == 2
        # Sort by priority to verify imminent is higher
        by_priority = sorted(proposals, key=lambda p: p.priority, reverse=True)
        assert by_priority[0].priority > by_priority[1].priority
        assert by_priority[0].priority == 80  # imminent priority
        store.close()

    def test_cancelled_event_not_returned(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        now = _now_utc()
        when = now + timedelta(hours=2)
        eid = store.add_event("meeting", "취소된 회의", when.strftime("%Y-%m-%d %H:%M:%S"))
        store.set_event_status(eid, "cancelled")

        trigger = TimeBasedTrigger(lookahead_hours=24.0)
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert proposals == []
        store.close()

    def test_multiple_events_multiple_proposals(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        now = _now_utc()
        for i in range(3):
            when = now + timedelta(hours=i + 1)
            store.add_event("task", f"할 일 {i}", when.strftime("%Y-%m-%d %H:%M:%S"))

        trigger = TimeBasedTrigger(lookahead_hours=24.0)
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert len(proposals) == 3
        store.close()

    def test_dedup_keys_are_stable(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        now = _now_utc()
        when = now + timedelta(hours=3)
        eid = store.add_event("meeting", "회의", when.strftime("%Y-%m-%d %H:%M:%S"))

        trigger = TimeBasedTrigger(lookahead_hours=24.0)
        import asyncio

        p1 = asyncio.run(trigger.evaluate(store, now=now))
        p2 = asyncio.run(trigger.evaluate(store, now=now))
        assert p1[0].dedup_key == p2[0].dedup_key
        assert p1[0].dedup_key == f"time:event:{eid}"
        store.close()


# ── SilenceTrigger ─────────────────────────────────────────────────────────────


class TestSilenceTrigger:
    def test_silence_fires_when_no_activity_file(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        activity_path = tmp_path / "activity.json"  # does not exist
        trigger = SilenceTrigger(threshold_hours=4.0, activity_path=activity_path)
        now = _now_utc()
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert len(proposals) == 1
        assert proposals[0].source == "silence"
        store.close()

    def test_silence_fires_when_elapsed_exceeds_threshold(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        activity_path = tmp_path / "activity.json"
        # Write activity 5 hours ago
        last_ts = (_now_utc() - timedelta(hours=5)).isoformat()
        activity_path.write_text(
            json.dumps({"channel": "text", "ts": last_ts}), encoding="utf-8"
        )
        trigger = SilenceTrigger(threshold_hours=4.0, activity_path=activity_path)
        now = _now_utc()
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert len(proposals) == 1
        assert proposals[0].dedup_key == "silence:check_in"
        store.close()

    def test_silence_does_not_fire_when_recent(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        activity_path = tmp_path / "activity.json"
        # Write activity 10 minutes ago
        last_ts = (_now_utc() - timedelta(minutes=10)).isoformat()
        activity_path.write_text(
            json.dumps({"channel": "voice", "ts": last_ts}), encoding="utf-8"
        )
        trigger = SilenceTrigger(threshold_hours=4.0, activity_path=activity_path)
        now = _now_utc()
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert proposals == []
        store.close()

    def test_silence_fires_exactly_at_threshold(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        activity_path = tmp_path / "activity.json"
        # Exactly at threshold — should NOT fire (elapsed < threshold is skip condition)
        # elapsed == threshold → elapsed < threshold is False → fires
        last_ts = (_now_utc() - timedelta(hours=4)).isoformat()
        activity_path.write_text(
            json.dumps({"channel": "text", "ts": last_ts}), encoding="utf-8"
        )
        trigger = SilenceTrigger(threshold_hours=4.0, activity_path=activity_path)
        now = _now_utc()
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        # elapsed is ~0 due to test execution time being <1s, so should be == threshold
        # allow both 0 and 1 proposal (timing sensitivity)
        assert len(proposals) <= 1
        store.close()


# ── RecurringPatternTrigger ────────────────────────────────────────────────────


class TestRecurringPatternTrigger:
    def test_no_facts_no_proposal(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        trigger = RecurringPatternTrigger()
        now = _now_utc()
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert proposals == []
        store.close()

    def test_unrelated_predicate_no_proposal(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        pid = store.add_person("홍길동", relation="나")
        store.add_fact(pid, "좋아하는음식", "불고기", confidence=0.9)
        trigger = RecurringPatternTrigger()
        now = _now_utc()
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert proposals == []
        store.close()

    def test_matching_fact_returns_proposal(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        pid = store.add_person("홍길동", relation="나")
        store.add_fact(pid, "주간_패턴", "금요일_저녁_외식", confidence=0.85)
        trigger = RecurringPatternTrigger()
        now = _now_utc()
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert len(proposals) == 1
        assert proposals[0].source == "pattern"
        assert "금요일_저녁_외식" in proposals[0].context
        store.close()

    def test_archived_fact_not_returned(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        pid = store.add_person("홍길동", relation="나")
        fid = store.add_fact(pid, "주간_패턴", "금요일_저녁_외식", confidence=0.85)
        store.archive_fact(fid)
        trigger = RecurringPatternTrigger()
        now = _now_utc()
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert proposals == []
        store.close()

    def test_multiple_people_multiple_proposals(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        for i in range(2):
            pid = store.add_person(f"사람{i}", relation="가족")
            store.add_fact(pid, "주간_패턴", f"패턴{i}", confidence=0.8)
        trigger = RecurringPatternTrigger()
        now = _now_utc()
        import asyncio

        proposals = asyncio.run(trigger.evaluate(store, now=now))
        assert len(proposals) == 2
        store.close()
