"""Entry point for running the consolidator.

    python -m apps.consolidator              # process last 24h, write log
    python -m apps.consolidator --no-llm     # dry-run; skip API call
    python -m apps.consolidator --since 48h  # custom lookback window
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def _parse_duration(s: str) -> int:
    """Parse duration string like '24h' or '48h' to integer hours."""
    s = s.strip().lower()
    if s.endswith("h"):
        return int(s[:-1])
    if s.endswith("d"):
        return int(s[:-1]) * 24
    return int(s)  # bare integer treated as hours


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m apps.consolidator",
        description="Run the her memory consolidator (Phase 4).",
    )
    p.add_argument(
        "--db",
        default=str(Path("data/db.sqlite")),
        help="Path to SQLite database (default: data/db.sqlite)",
    )
    p.add_argument(
        "--log-dir",
        default=str(Path("data/consolidation_log")),
        help="Directory for JSON consolidation logs",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Dry-run: skip LLM API calls (useful for offline testing)",
    )
    p.add_argument(
        "--since",
        default="24h",
        metavar="DURATION",
        help="Lookback window (e.g. 24h, 48h, 7d). Default: 24h",
    )
    return p


async def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Import here to avoid circular issues at package init
    from apps.memory.store import MemoryStore
    from apps.consolidator.runner import run_consolidation

    db_path = Path(args.db)
    log_dir = Path(args.log_dir)
    lookback_hours = _parse_duration(args.since)
    dry_run: bool = args.no_llm

    # Build Gemini client (unless dry-run)
    client = None
    if not dry_run:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("오류: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.", file=sys.stderr)
            print("  direnv allow 를 실행하거나 GEMINI_API_KEY 를 설정하세요.", file=sys.stderr)
            sys.exit(1)

        try:
            from apps.agent.gemini import GeminiClient
            client = GeminiClient(model_id="gemini-2.5-flash", api_key=api_key)
        except Exception as exc:
            print(f"오류: Gemini 클라이언트 초기화 실패 — {exc}", file=sys.stderr)
            sys.exit(1)

    store = MemoryStore(db_path)
    try:
        print(f"🔍 메모리 통합 시작 (최근 {lookback_hours}시간, DB: {db_path})")
        if dry_run:
            print("  [dry-run 모드: LLM 호출 없음]")

        report = await run_consolidation(
            store,
            lookback_hours=lookback_hours,
            log_dir=log_dir,
            client=client,
            dry_run=dry_run,
        )

        # Summary in Korean for the user
        print()
        print("=" * 50)
        print("통합 완료")
        print("=" * 50)
        print(f"  실행 시각  : {report.ran_at}")
        print(f"  에피소드   : {report.episodes_processed}개 처리")
        print(f"  사실 추출  : {report.facts_extracted}개 (LLM 후보)")
        print(f"  사실 승격  : {report.facts_promoted}개 (신뢰도 ≥ 0.7)")
        print(f"  사실 아카이브: {report.facts_archived}개 (이전 충돌 사실)")
        print(f"  일정 추가  : {report.events_added}개")
        if report.errors:
            print(f"  오류       : {len(report.errors)}건")
            for err in report.errors:
                print(f"    - {err}")
        print(f"  로그 저장  : {log_dir}/")
        print()

    finally:
        store.close()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
