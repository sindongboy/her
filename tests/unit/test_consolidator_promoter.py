"""Tests for apps.consolidator.promoter."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.consolidator.extractor import ExtractedEvent, ExtractedFact
from apps.consolidator.promoter import (
    _ACTION_ADDED,
    _ACTION_ARCHIVED_AND_ADDED,
    _ACTION_SKIPPED_DUPLICATE,
    _ACTION_SKIPPED_LOW,
    _ACTION_SKIPPED_UNKNOWN,
    promote_event,
    promote_fact,
)
from apps.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "test.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# promote_fact
# ---------------------------------------------------------------------------


def test_low_confidence_skipped(store: MemoryStore) -> None:
    store.add_person("어머니")
    fact = ExtractedFact(
        subject_person_name="어머니",
        predicate="좋아한다",
        object="단호박 케이크",
        confidence=0.6,
    )
    outcome = promote_fact(store, fact)
    assert outcome.action == _ACTION_SKIPPED_LOW
    assert outcome.new_fact_id is None
    assert outcome.archived_fact_ids == []


def test_high_confidence_unknown_person_skipped(store: MemoryStore) -> None:
    # No person added → resolution fails
    fact = ExtractedFact(
        subject_person_name="외할머니",
        predicate="직업",
        object="교사",
        confidence=0.9,
    )
    outcome = promote_fact(store, fact)
    assert outcome.action == _ACTION_SKIPPED_UNKNOWN
    assert outcome.new_fact_id is None


def test_null_person_name_skipped_unknown(store: MemoryStore) -> None:
    fact = ExtractedFact(
        subject_person_name=None,
        predicate="좋아한다",
        object="커피",
        confidence=0.8,
    )
    outcome = promote_fact(store, fact)
    assert outcome.action == _ACTION_SKIPPED_UNKNOWN


def test_existing_person_new_predicate_added(store: MemoryStore) -> None:
    pid = store.add_person("어머니")
    fact = ExtractedFact(
        subject_person_name="어머니",
        predicate="좋아한다",
        object="단호박 케이크",
        confidence=0.8,
    )
    outcome = promote_fact(store, fact)
    assert outcome.action == _ACTION_ADDED
    assert outcome.new_fact_id is not None
    assert outcome.archived_fact_ids == []

    # Fact is stored correctly
    facts = store.list_active_facts(pid)
    assert len(facts) == 1
    assert facts[0].predicate == "좋아한다"
    assert facts[0].object == "단호박 케이크"
    assert facts[0].confidence == pytest.approx(0.8)


def test_conflict_archives_old_and_adds_new(store: MemoryStore) -> None:
    pid = store.add_person("어머니")
    old_id = store.add_fact(pid, "좋아한다", "단호박 케이크", confidence=0.9)

    new_fact = ExtractedFact(
        subject_person_name="어머니",
        predicate="좋아한다",
        object="초콜릿 케이크",
        confidence=0.85,
    )
    outcome = promote_fact(store, new_fact)
    assert outcome.action == _ACTION_ARCHIVED_AND_ADDED
    assert old_id in outcome.archived_fact_ids
    assert outcome.new_fact_id is not None

    active = store.list_active_facts(pid)
    assert len(active) == 1
    assert active[0].object == "초콜릿 케이크"

    # Old fact is archived, not deleted
    row = store.conn.execute(
        "SELECT archived_at FROM facts WHERE id = ?", (old_id,)
    ).fetchone()
    assert row["archived_at"] is not None


def test_duplicate_same_predicate_and_object_skipped(store: MemoryStore) -> None:
    pid = store.add_person("어머니")
    store.add_fact(pid, "좋아한다", "단호박 케이크", confidence=0.9)

    same_fact = ExtractedFact(
        subject_person_name="어머니",
        predicate="좋아한다",
        object="단호박 케이크",
        confidence=0.85,
    )
    outcome = promote_fact(store, same_fact)
    assert outcome.action == _ACTION_SKIPPED_DUPLICATE
    assert outcome.new_fact_id is None

    # Still only one fact in the store
    assert len(store.list_active_facts(pid)) == 1


def test_person_name_case_insensitive_match(store: MemoryStore) -> None:
    pid = store.add_person("어머니")
    fact = ExtractedFact(
        subject_person_name="어머니",  # exact match
        predicate="알러지",
        object="땅콩",
        confidence=0.95,
    )
    outcome = promote_fact(store, fact)
    assert outcome.action == _ACTION_ADDED


def test_confidence_exactly_at_threshold_accepted(store: MemoryStore) -> None:
    store.add_person("아내")
    fact = ExtractedFact(
        subject_person_name="아내",
        predicate="직업",
        object="의사",
        confidence=0.7,  # exactly at threshold
    )
    outcome = promote_fact(store, fact, confidence_threshold=0.7)
    assert outcome.action == _ACTION_ADDED


def test_confidence_just_below_threshold_skipped(store: MemoryStore) -> None:
    store.add_person("아내")
    fact = ExtractedFact(
        subject_person_name="아내",
        predicate="직업",
        object="의사",
        confidence=0.699,
    )
    outcome = promote_fact(store, fact, confidence_threshold=0.7)
    assert outcome.action == _ACTION_SKIPPED_LOW


# ---------------------------------------------------------------------------
# promote_event
# ---------------------------------------------------------------------------


def test_event_with_valid_fields_added(store: MemoryStore) -> None:
    event = ExtractedEvent(
        person_name=None,
        type="appointment",
        title="치과 예약",
        when_at="2026-05-10T14:00:00",
        recurrence=None,
    )
    errors: list[str] = []
    event_id = promote_event(store, event, errors=errors)
    assert event_id is not None
    assert errors == []
    e = store.get_event(event_id)
    assert e is not None
    assert e.title == "치과 예약"
    assert e.type == "appointment"
    assert e.person_id is None


def test_event_with_known_person_resolved(store: MemoryStore) -> None:
    pid = store.add_person("아내")
    event = ExtractedEvent(
        person_name="아내",
        type="birthday",
        title="아내 생일",
        when_at="2026-07-15T00:00:00",
        recurrence=None,
    )
    event_id = promote_event(store, event)
    assert event_id is not None
    e = store.get_event(event_id)
    assert e is not None
    assert e.person_id == pid


def test_event_with_missing_when_at_skipped(store: MemoryStore) -> None:
    event = ExtractedEvent(
        person_name=None,
        type="trip",
        title="제주도 여행",
        when_at="",  # empty → skip
        recurrence=None,
    )
    event_id = promote_event(store, event)
    assert event_id is None


def test_event_with_missing_title_skipped(store: MemoryStore) -> None:
    event = ExtractedEvent(
        person_name=None,
        type="appointment",
        title="",  # empty title
        when_at="2026-05-10T10:00:00",
        recurrence=None,
    )
    event_id = promote_event(store, event)
    assert event_id is None


def test_event_with_unknown_person_still_added(store: MemoryStore) -> None:
    """Person not in DB → event still stored with person_id=None."""
    event = ExtractedEvent(
        person_name="할머니",
        type="trip",
        title="할머니 여행",
        when_at="2026-06-01T09:00:00",
        recurrence=None,
    )
    event_id = promote_event(store, event)
    assert event_id is not None
    e = store.get_event(event_id)
    assert e is not None
    assert e.person_id is None  # unknown person → None, but event still saved


def test_event_with_recurrence(store: MemoryStore) -> None:
    event = ExtractedEvent(
        person_name=None,
        type="appointment",
        title="주간 미팅",
        when_at="2026-05-01T09:00:00",
        recurrence="FREQ=WEEKLY",
    )
    event_id = promote_event(store, event)
    assert event_id is not None
    e = store.get_event(event_id)
    assert e is not None
    assert e.recurrence == "FREQ=WEEKLY"


# ── promote_note ─────────────────────────────────────────────────────────


from apps.consolidator.extractor import ExtractedNote
from apps.consolidator.promoter import promote_note


def test_promote_note_added(store: MemoryStore) -> None:
    note = ExtractedNote(content="매주 금요일 외식하기", tags=["routine"])
    nid = promote_note(store, note)
    assert nid is not None
    notes = store.list_notes()
    assert len(notes) == 1
    assert notes[0].content == "매주 금요일 외식하기"
    assert notes[0].tags == ["routine"]


def test_promote_note_with_source_session(store: MemoryStore) -> None:
    sid = store.add_session()
    note = ExtractedNote(content="결정사항", tags=[])
    nid = promote_note(store, note, source_session_id=sid)
    assert nid is not None
    notes = store.list_notes()
    assert notes[0].source_session_id == sid


def test_promote_note_skips_empty_content(store: MemoryStore) -> None:
    note = ExtractedNote(content="   ", tags=["tag"])
    assert promote_note(store, note) is None
    assert store.list_notes() == []


def test_promote_note_dedup_same_content(store: MemoryStore) -> None:
    note = ExtractedNote(content="같은 내용", tags=[])
    nid1 = promote_note(store, note)
    nid2 = promote_note(store, note)
    assert nid1 is not None
    assert nid2 is None
    assert len(store.list_notes()) == 1
