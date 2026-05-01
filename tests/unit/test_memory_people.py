from __future__ import annotations

from apps.memory import MemoryStore


def test_add_and_get_person_round_trips_all_fields(store: MemoryStore) -> None:
    pid = store.add_person(
        name="아내",
        relation="spouse",
        birthday="1990-05-12",
        preferences={"food": ["김치찌개"], "music": "재즈"},
    )
    assert pid > 0

    p = store.get_person(pid)
    assert p is not None
    assert p.name == "아내"
    assert p.relation == "spouse"
    assert p.birthday == "1990-05-12"
    assert p.preferences == {"food": ["김치찌개"], "music": "재즈"}


def test_add_person_minimal(store: MemoryStore) -> None:
    pid = store.add_person(name="아들")

    p = store.get_person(pid)
    assert p is not None
    assert p.name == "아들"
    assert p.relation is None
    assert p.birthday is None
    assert p.preferences == {}


def test_get_person_missing_returns_none(store: MemoryStore) -> None:
    assert store.get_person(999) is None


def test_list_people_orders_by_id(store: MemoryStore) -> None:
    a = store.add_person(name="A")
    b = store.add_person(name="B")
    c = store.add_person(name="C")

    people = store.list_people()
    assert [p.id for p in people] == [a, b, c]
    assert [p.name for p in people] == ["A", "B", "C"]


def test_preferences_round_trip_preserves_korean(store: MemoryStore) -> None:
    prefs = {"음식": ["된장찌개", "김밥"], "취미": "독서"}
    pid = store.add_person(name="어머니", preferences=prefs)

    p = store.get_person(pid)
    assert p is not None
    assert p.preferences == prefs
