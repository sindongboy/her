"""Confidence filtering, conflict resolution, and memory promotion.

Converts extracted facts/events into persisted memory entries.
Per CLAUDE.md §5.3: archive conflicting old facts (never delete).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from apps.consolidator.extractor import ExtractedEvent, ExtractedFact
from apps.memory.store import MemoryStore, Person

log = structlog.get_logger(__name__)

CONFIDENCE_THRESHOLD = 0.7


@dataclass(slots=True, frozen=True)
class PromotionOutcome:
    action: str  # see _ACTION constants below
    new_fact_id: int | None
    archived_fact_ids: list[int]


# Action constants
_ACTION_SKIPPED_LOW = "skipped_low_confidence"
_ACTION_SKIPPED_UNKNOWN = "skipped_unknown_person"
_ACTION_SKIPPED_AMBIGUOUS = "skipped_ambiguous_person"
_ACTION_SKIPPED_DUPLICATE = "skipped_duplicate"
_ACTION_ADDED = "added"
_ACTION_ARCHIVED_AND_ADDED = "archived_and_added"
_ACTION_SKIPPED_MISSING_FIELDS = "skipped_missing_fields"
_ACTION_EVENT_ADDED = "event_added"


def _resolve_person(
    name: str | None,
    people: list[Person],
) -> tuple[int | None, str]:
    """Resolve a name string to a person_id.

    Returns (person_id, action_if_failed).
    action_if_failed is empty string on success.
    """
    if name is None:
        return None, _ACTION_SKIPPED_UNKNOWN

    name_lower = name.strip().lower()
    if not name_lower:
        return None, _ACTION_SKIPPED_UNKNOWN

    # 1. Exact match (case-insensitive)
    exact = [p for p in people if p.name.lower() == name_lower]
    if len(exact) == 1:
        return exact[0].id, ""
    if len(exact) > 1:
        log.warning("promoter.ambiguous_person", name=name_lower, count=len(exact))
        return None, _ACTION_SKIPPED_AMBIGUOUS

    # 2. Substring fallback — only if uniquely matches one person
    subs = [p for p in people if name_lower in p.name.lower() or p.name.lower() in name_lower]
    if len(subs) == 1:
        return subs[0].id, ""
    if len(subs) > 1:
        log.warning("promoter.ambiguous_person_substring", name=name_lower, count=len(subs))
        return None, _ACTION_SKIPPED_AMBIGUOUS

    return None, _ACTION_SKIPPED_UNKNOWN


def promote_fact(
    store: MemoryStore,
    fact: ExtractedFact,
    *,
    source_episode_id: int | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> PromotionOutcome:
    """Promote a single extracted fact to semantic memory.

    Rules (CLAUDE.md §5.3):
    - confidence < threshold → skip
    - unknown/ambiguous person → skip
    - same subject+predicate+object → no-op (duplicate)
    - same subject+predicate, different object → archive old + add new
    - otherwise → add new fact
    """
    # 1. Confidence filter
    if fact.confidence < confidence_threshold:
        log.debug(
            "promoter.skipped_low_confidence",
            predicate=fact.predicate,
            confidence=fact.confidence,
        )
        return PromotionOutcome(
            action=_ACTION_SKIPPED_LOW,
            new_fact_id=None,
            archived_fact_ids=[],
        )

    # 2. Resolve person
    people = store.list_people()
    person_id, fail_action = _resolve_person(fact.subject_person_name, people)
    if person_id is None:
        log.debug(
            "promoter.skipped_person",
            name=fact.subject_person_name,
            action=fail_action,
        )
        return PromotionOutcome(
            action=fail_action,
            new_fact_id=None,
            archived_fact_ids=[],
        )

    # 3. Check for existing active facts with same subject + predicate
    existing = store.list_active_facts(person_id)
    same_predicate = [f for f in existing if f.predicate == fact.predicate]

    # Duplicate check: exact same predicate + object already exists
    exact_dupes = [f for f in same_predicate if f.object == fact.object]
    if exact_dupes:
        log.debug(
            "promoter.duplicate",
            person_id=person_id,
            predicate=fact.predicate,
        )
        return PromotionOutcome(
            action=_ACTION_SKIPPED_DUPLICATE,
            new_fact_id=None,
            archived_fact_ids=[],
        )

    # Conflict: same predicate but different object → archive old, add new
    archived_ids: list[int] = []
    if same_predicate:
        for old_fact in same_predicate:
            store.archive_fact(old_fact.id)
            archived_ids.append(old_fact.id)
            log.info(
                "promoter.archived_conflict",
                person_id=person_id,
                predicate=fact.predicate,
                old_fact_id=old_fact.id,
            )

    # 4. Add the new fact
    new_id = store.add_fact(
        subject_person_id=person_id,
        predicate=fact.predicate,
        object=fact.object,
        confidence=fact.confidence,
        source_episode_id=source_episode_id,
    )
    action = _ACTION_ARCHIVED_AND_ADDED if archived_ids else _ACTION_ADDED
    log.info(
        "promoter.added_fact",
        person_id=person_id,
        predicate=fact.predicate,
        new_fact_id=new_id,
        action=action,
    )
    return PromotionOutcome(
        action=action,
        new_fact_id=new_id,
        archived_fact_ids=archived_ids,
    )


def promote_event(
    store: MemoryStore,
    event: ExtractedEvent,
    *,
    errors: list[str] | None = None,
) -> int | None:
    """Add an extracted event to the events table.

    Returns the new event_id on success, None on skip.
    No conflict resolution for events — just add.
    """
    if errors is None:
        errors = []

    # Validate required fields
    if not event.type or not event.title or not event.when_at:
        log.debug(
            "promoter.skipped_event_missing_fields",
            type=event.type,
            title=event.title,
            when_at=event.when_at,
        )
        return None

    # Resolve person (optional for events)
    person_id: int | None = None
    if event.person_name:
        people = store.list_people()
        resolved_id, fail_action = _resolve_person(event.person_name, people)
        if resolved_id is not None:
            person_id = resolved_id
        else:
            log.debug(
                "promoter.event_person_not_resolved",
                name=event.person_name,
                action=fail_action,
            )
            # Continue with person_id=None (event still stored)

    try:
        event_id = store.add_event(
            type=event.type,
            title=event.title,
            when_at=event.when_at,
            person_id=person_id,
            recurrence=event.recurrence,
            source="consolidator",
        )
        log.info(
            "promoter.added_event",
            event_id=event_id,
            person_id=person_id,
            title=event.title,
        )
        return event_id
    except Exception as exc:
        msg = f"Failed to add event '{event.title}': {exc}"
        log.error("promoter.event_add_error", error=msg)
        errors.append(msg)
        return None
