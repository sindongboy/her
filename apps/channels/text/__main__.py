"""Entry point for the text channel REPL.

Usage:
    python -m apps.channels.text

Environment variables:
    GEMINI_API_KEY   Required for AgentCore.
    HER_DB_PATH      Path to SQLite DB (default: data/db.sqlite).
"""

from __future__ import annotations

import asyncio

from apps.channels.text.repl import run_repl


def main() -> None:
    try:
        asyncio.run(run_repl(bus=None))
    except KeyboardInterrupt:
        print("\n대화를 종료합니다. 안녕히 계세요!")


if __name__ == "__main__":
    main()
