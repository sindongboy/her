"""scripts/seed_family.py — interactive Korean CLI to seed user + family into MemoryStore.

Usage:
  python scripts/seed_family.py [--db PATH] [--non-interactive --from-json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps.memory.store import MemoryStore  # noqa: E402


# ── data helpers ─────────────────────────────────────────────────────────────


def _parse_food_list(raw: str) -> list[str]:
    """Split comma-separated food string into cleaned list."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _seed_person(
    store: MemoryStore,
    name: str,
    relation: str,
    birthday: str | None,
    food_items: list[str],
) -> int:
    preferences: dict[str, Any] = {}
    if food_items:
        preferences["food"] = food_items
    pid = store.add_person(name, relation=relation, birthday=birthday or None, preferences=preferences)
    for food in food_items:
        store.upsert_preference(pid, "food", food)
    return pid


# ── interactive mode ──────────────────────────────────────────────────────────


def _prompt(label: str, default: str = "") -> str:
    """Prompt user and return input (stripped). Returns default on empty input."""
    suffix = f"[기본: {default}]" if default else ""
    raw = input(f"  {label}{(' ' + suffix) if suffix else ''}: ").strip()
    return raw or default


def run_interactive(store: MemoryStore) -> tuple[int, int]:
    """Run the interactive seeding flow. Returns (self_count, family_count)."""
    print("\n🌱 her — 가족 정보 입력")
    print(f"   DB 경로: {store.db_path}\n")

    # Self
    self_name = _prompt("본인 이름을 입력하세요")
    if not self_name:
        print("이름이 비어있습니다. 취소됩니다.")
        return 0, 0
    self_relation = _prompt("관계 별칭(예: '나' / 'me')", default="me")
    self_id = _seed_person(store, self_name, self_relation, None, [])
    print(f"   ✓ 본인 등록: {self_name} (id={self_id})\n")

    # Family members
    print("가족을 추가합니다. 빈 이름을 입력하면 종료됩니다.\n")
    family_count = 0
    idx = 1
    while True:
        name = input(f"[{idx}] 이름: ").strip()
        if not name:
            break
        relation = input("    관계: ").strip() or "family"
        birthday_raw = input("    생일 (YYYY-MM-DD, 빈칸 가능): ").strip()
        food_raw = input("    좋아하는 음식 (쉼표 구분, 빈칸 가능): ").strip()
        food_items = _parse_food_list(food_raw)

        fid = _seed_person(store, name, relation, birthday_raw or None, food_items)
        print(f"   ✓ 등록: {name} (id={fid})\n")
        family_count += 1
        idx += 1

    return 1, family_count


# ── non-interactive / JSON mode ───────────────────────────────────────────────


def run_from_json(store: MemoryStore, json_path: Path) -> tuple[int, int]:
    """Seed from a JSON file.

    Expected format::

        {
          "self": {"name": "홍길동", "relation": "me"},
          "family": [
            {"name": "아내", "relation": "spouse", "birthday": "1990-05-12",
             "food": ["김치찌개"]}
          ]
        }
    """
    data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))

    self_entry: dict[str, Any] = data.get("self", {})
    self_name: str = self_entry.get("name", "")
    if not self_name:
        print("[seed] ERROR: 'self.name' is required in JSON", file=sys.stderr)
        sys.exit(1)
    self_relation: str = self_entry.get("relation", "me")
    self_id = _seed_person(store, self_name, self_relation, None, [])
    print(f"[seed] self: {self_name} id={self_id}")

    family: list[dict[str, Any]] = data.get("family", [])
    for entry in family:
        name: str = entry.get("name", "")
        if not name:
            continue
        relation: str = entry.get("relation", "family")
        birthday: str | None = entry.get("birthday") or None
        food_items: list[str] = entry.get("food", [])
        fid = _seed_person(store, name, relation, birthday, food_items)
        print(f"[seed] family: {name} id={fid}")

    return 1, len(family)


# ── main ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Seed initial family data into the her memory store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default="data/db.sqlite", help="Path to SQLite DB (default: data/db.sqlite)")
    p.add_argument("--non-interactive", action="store_true", help="Read from --from-json instead of prompting")
    p.add_argument("--from-json", metavar="PATH", help="JSON file for non-interactive seeding")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(db_path)

    try:
        if args.non_interactive:
            if not args.from_json:
                print("[seed] --non-interactive requires --from-json PATH", file=sys.stderr)
                sys.exit(1)
            self_count, family_count = run_from_json(store, Path(args.from_json))
        else:
            self_count, family_count = run_interactive(store)
    except KeyboardInterrupt:
        print("\n\n취소됨.")
        sys.exit(0)
    finally:
        store.close()

    total = self_count + family_count
    if total > 0:
        print(f"\n✅ 저장 완료: 본인 {self_count}명 + 가족 {family_count}명")
    else:
        print("\n저장된 항목이 없습니다.")


if __name__ == "__main__":
    main()
