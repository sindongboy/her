"""Proactive trigger implementations.

Each trigger evaluates a condition against the memory store and returns
a list of TriggerProposal objects describing what to say and why.

CLAUDE.md §2.4 trigger sources:
  1. Time-based: upcoming event in events table within lookahead window.
  2. Silence: no interaction for threshold_hours.
  3. Recurring pattern: stub — looks for facts with predicate='주간_패턴'.
  4. External: placeholder — not implemented in v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import structlog

from apps.memory.store import MemoryStore
from apps.proactive.activity import read_last_activity

log = structlog.get_logger(__name__)


# ── TriggerProposal ───────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class TriggerProposal:
    """A trigger's proposed proactive utterance.

    The engine ranks, dedupes, and applies daily limits before firing.
    """

    source: str          # "time" | "silence" | "pattern" | "external"
    priority: int        # 0..100; higher fires first when proposals collide
    context: str         # Korean text describing the situation (fed to LLM)
    dedup_key: str       # stable string — same key won't fire twice in N hours


# ── Trigger protocol ──────────────────────────────────────────────────────────


class Trigger(Protocol):
    """Protocol every trigger must satisfy."""

    async def evaluate(
        self, store: MemoryStore, *, now: datetime
    ) -> list[TriggerProposal]:
        """Return zero or more proposals given the current store state."""
        ...


# ── TimeBasedTrigger ─────────────────────────────────────────────────────────


class TimeBasedTrigger:
    """Fire when an event in `events` is within `lookahead_hours` of now.

    Imminent events (within `lookahead_imminent_hours`) get higher priority.
    """

    def __init__(
        self,
        *,
        lookahead_hours: float = 24.0,
        lookahead_imminent_hours: float = 1.0,
    ) -> None:
        self._lookahead_hours = lookahead_hours
        self._lookahead_imminent_hours = lookahead_imminent_hours

    async def evaluate(
        self, store: MemoryStore, *, now: datetime
    ) -> list[TriggerProposal]:
        """Query the store for upcoming events; return one proposal per event.

        Note: SQLite store is single-threaded; call directly on event loop.
        """
        now_iso = now.strftime("%Y-%m-%d %H:%M:%S")
        try:
            events = store.list_upcoming_events(
                within_hours=int(self._lookahead_hours),
                now_iso=now_iso,
            )
        except Exception as exc:
            log.warning("time_trigger.store_error", error=str(exc))
            return []

        proposals: list[TriggerProposal] = []
        for event in events:
            try:
                event_dt = datetime.fromisoformat(event.when_at.replace("Z", "+00:00"))
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
                delta_h = (event_dt - now).total_seconds() / 3600.0
            except (ValueError, TypeError):
                delta_h = self._lookahead_hours  # treat as non-imminent

            is_imminent = 0 < delta_h <= self._lookahead_imminent_hours
            priority = 80 if is_imminent else 50

            context = (
                f"곧 예정된 일정이 있습니다: '{event.title}' "
                f"({event.when_at}). "
                f"{'1시간 이내로 임박했습니다.' if is_imminent else '24시간 이내 일정입니다.'}"
                " 사용자에게 자연스럽게 미리 알려주세요."
            )
            dedup_key = f"time:event:{event.id}"

            proposals.append(
                TriggerProposal(
                    source="time",
                    priority=priority,
                    context=context,
                    dedup_key=dedup_key,
                )
            )

        return proposals


# ── SilenceTrigger ────────────────────────────────────────────────────────────


class SilenceTrigger:
    """Fire when last interaction was longer than `threshold_hours` ago.

    Reads ``~/.her/activity.json`` written by the activity module.
    """

    def __init__(
        self,
        *,
        threshold_hours: float = 4.0,
        activity_path: object | None = None,  # Path | None — kept loose for tests
    ) -> None:
        self._threshold_hours = threshold_hours
        self._activity_path = activity_path

    async def evaluate(
        self, store: MemoryStore, *, now: datetime
    ) -> list[TriggerProposal]:
        """Return a silence proposal if no interaction within threshold."""
        result = read_last_activity(path=self._activity_path)
        if result is None:
            # No activity on record — treat as silence exceeded
            elapsed_h = self._threshold_hours + 1.0
        else:
            _channel, last_dt = result
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            elapsed_h = (now - last_dt).total_seconds() / 3600.0

        if elapsed_h < self._threshold_hours:
            return []

        log.debug("silence_trigger.fired", elapsed_hours=round(elapsed_h, 2))
        return [
            TriggerProposal(
                source="silence",
                priority=20,
                context=(
                    f"사용자와 마지막 대화 후 약 {elapsed_h:.0f}시간이 지났습니다. "
                    "부드럽게 안부를 여쭤보거나 오늘 하루 어떠셨는지 물어봐 주세요."
                ),
                dedup_key="silence:check_in",
            )
        ]


# ── RecurringPatternTrigger ───────────────────────────────────────────────────


class RecurringPatternTrigger:
    """v1 stub: look for facts with predicate='주간_패턴' and parse object.

    In v1, only fires when such a fact exists for any person. The pattern
    matching logic is intentionally minimal — Phase 4 v2 will refine.
    """

    async def evaluate(
        self, store: MemoryStore, *, now: datetime
    ) -> list[TriggerProposal]:
        """Check all people for recurring pattern facts.

        Note: SQLite store is single-threaded; call directly on event loop.
        """
        try:
            people = store.list_people()
        except Exception as exc:
            log.warning("pattern_trigger.store_error", error=str(exc))
            return []

        proposals: list[TriggerProposal] = []
        for person in people:
            try:
                facts = store.list_active_facts(person.id)
            except Exception as exc:
                log.warning("pattern_trigger.facts_error", error=str(exc))
                continue

            for fact in facts:
                if fact.predicate != "주간_패턴":
                    continue
                # Pattern format: "금요일_저녁_외식" or freeform string
                context = (
                    f"반복 패턴이 감지되었습니다: {person.name}님과 관련된 "
                    f"'{fact.object}' 패턴이 있습니다. "
                    "이와 관련해 사용자에게 자연스럽게 물어봐 주세요."
                )
                dedup_key = f"pattern:fact:{fact.id}"
                proposals.append(
                    TriggerProposal(
                        source="pattern",
                        priority=40,
                        context=context,
                        dedup_key=dedup_key,
                    )
                )

        return proposals
