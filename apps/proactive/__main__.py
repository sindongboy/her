"""Standalone runner for testing the ProactiveEngine.

Usage:
    python -m apps.proactive --once [--no-llm]

Options:
    --once      Run a single trigger evaluation tick then exit.
    --no-llm    Skip agent LLM call; use context string directly as utterance.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProactiveEngine standalone runner")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single evaluation tick then exit.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        dest="no_llm",
        help="Skip LLM call; use raw trigger context as utterance.",
    )
    return parser.parse_args()


class _NoLLMAgent:
    """Fake agent that returns the trigger context as the utterance directly."""

    async def respond(
        self,
        message: str,
        *,
        episode_id: object = None,
        channel: str = "voice",
        attachments: object = None,
    ) -> object:
        class _R:
            text: str

        r = _R()
        # Return first sentence of the context as the utterance
        sentences = message.split(".")
        r.text = sentences[0].strip() + "." if sentences else message
        return r


class _PrintTextChannel:
    """Simple text channel that prints proactive output to stdout."""

    async def say(self, text: str) -> None:
        print(f"[먼저] > {text}", flush=True)


async def _main(args: argparse.Namespace) -> int:
    db_path = Path(os.environ.get("HER_DB_PATH", "data/db.sqlite"))
    api_key = os.environ.get("GEMINI_API_KEY", "")

    try:
        from apps.memory.store import MemoryStore
    except ImportError as exc:
        print(f"[오류] MemoryStore import failed: {exc}", file=sys.stderr)
        return 1

    store = MemoryStore(db_path)

    if args.no_llm or not api_key:
        if not api_key:
            print(
                "[경고] GEMINI_API_KEY 없음 — --no-llm 모드로 실행합니다.",
                file=sys.stderr,
            )
        agent: object = _NoLLMAgent()
    else:
        from apps.agent.core import AgentCore
        agent = AgentCore(store, api_key=api_key)

    text_channel = _PrintTextChannel()

    from apps.proactive.engine import ProactiveConfig, ProactiveEngine

    config = ProactiveConfig(
        daily_limit=3,
        silence_threshold_hours=4.0,
        cooldown_minutes=0,  # no cooldown in test mode
        quiet_mode=False,
    )

    engine = ProactiveEngine(
        store,
        agent,  # type: ignore[arg-type]
        text_channel=text_channel,  # type: ignore[arg-type]
        config=config,
    )

    if args.once:
        print("[proactive] 단일 틱 실행...", file=sys.stderr)
        fired = await engine.trigger_once()
        if fired:
            print("[proactive] 발화 완료.", file=sys.stderr)
        else:
            print("[proactive] 발화 없음 (조건 미충족).", file=sys.stderr)
        return 0

    # Continuous mode: run until Ctrl+C
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_int() -> None:
        print("\n[proactive] 종료 중...", file=sys.stderr)
        stop_event.set()

    try:
        loop.add_signal_handler(__import__("signal").SIGINT, _handle_int)
        loop.add_signal_handler(__import__("signal").SIGTERM, _handle_int)
    except (NotImplementedError, AttributeError):
        pass  # Windows / non-Unix

    print("[proactive] 연속 모드 시작. Ctrl+C 로 종료.", file=sys.stderr)
    await engine.run(stop_event=stop_event)
    return 0


def main() -> None:
    args = _parse_args()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
