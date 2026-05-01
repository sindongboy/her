"""Entry point for the voice channel.

Usage:
    python -m apps.channels.voice

Environment variables:
    GEMINI_API_KEY   Required for AgentCore and Gemini TTS.
    HER_DB_PATH      Path to SQLite DB (default: data/db.sqlite).
    HER_STT_MODEL    faster-whisper model size (default: medium).
    HER_TTS_VOICE    Gemini TTS voice name (default: Kore).

Per CLAUDE.md §6.1 (Voice Channel, Phase 1).
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import warnings

from apps.channels.voice.channel import run_voice


def main() -> None:
    # silero-vad / faster-whisper use C++ threads + multiprocessing internally
    # that don't always release cleanly on shutdown.  Suppress the noisy
    # `resource_tracker: leaked semaphore` warning and treat SIGTERM the same
    # as Ctrl-C so `make voice` exits 0 instead of 143.
    warnings.filterwarnings("ignore", message=r".*leaked semaphore.*")

    def _graceful_exit(_signum: int, _frame: object) -> None:
        print("\n[음성 채널] 종료합니다. 안녕히 계세요!", file=sys.stderr)
        os._exit(0)

    signal.signal(signal.SIGTERM, _graceful_exit)

    try:
        asyncio.run(run_voice(bus=None))
    except KeyboardInterrupt:
        print("\n[음성 채널] 종료합니다. 안녕히 계세요!", file=sys.stderr)


if __name__ == "__main__":
    main()
