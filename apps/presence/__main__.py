"""Entrypoint for the presence server.

Run with::

    python -m apps.presence
    # or
    make presence

Architecture (Phase 3.5 v2 — cross-process bus)
-------------------------------------------------
The presence server now has two modes of operation:

1. **Server-only** (default — no flags):
   The server starts and listens on ``http://127.0.0.1:8765``.  The event bus
   is initially empty.  Channel processes in *other* terminals drive the orb by
   setting ``HER_PRESENCE_URL=http://127.0.0.1:8765`` and running::

       HER_PRESENCE_URL=http://127.0.0.1:8765 make voice
       HER_PRESENCE_URL=http://127.0.0.1:8765 make text
       # or
       make voice-with-presence    # convenience shorthand in Makefile
       make text-with-presence

   Channel processes detect ``HER_PRESENCE_URL`` and automatically use a
   ``RemoteEventBus`` to POST events to ``/publish`` on this server.

2. **Co-process mode** (``--with-text`` / ``--with-voice``):
   A channel task is spawned *inside* the same asyncio event loop so it
   shares the in-process bus directly.  Useful for single-terminal demos or
   debugging.

       make presence-text    # server + text REPL in one terminal
       make presence-voice   # server + voice channel in one terminal

Flags
-----
(no flag)
    Server only — bus stays empty until driven by an external channel process
    via ``HER_PRESENCE_URL``.

``--with-text``
    Start a text-channel REPL task in the same asyncio event loop.

``--with-voice``
    Start a voice-channel task in the same asyncio event loop (Phase 1+
    hardware required).

The two ``--with-*`` flags are mutually exclusive.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def _validate_host(host: str) -> None:
    """Raise SystemExit if host is not a loopback address."""
    allowed = {"127.0.0.1", "localhost", "::1"}
    if host not in allowed:
        raise SystemExit(
            "프레즌스 서버는 로컬 전용이에요. host 는 127.0.0.1 만 허용."
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m apps.presence",
        description="her presence server (Samantha-style orb).",
        epilog=(
            "다른 터미널에서 채널을 연결하려면:\n"
            "  HER_PRESENCE_URL=http://127.0.0.1:8765 make voice\n"
            "  HER_PRESENCE_URL=http://127.0.0.1:8765 make text\n"
            "또는 Makefile 단축키:\n"
            "  make voice-with-presence / make text-with-presence"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind address (loopback only).")
    p.add_argument("--port", default=8765, type=int, help="TCP port (default 8765).")
    p.add_argument(
        "--log-level",
        default="warning",
        help="uvicorn log level (default: warning so voice status messages aren't drowned out).",
    )

    channel_group = p.add_mutually_exclusive_group()
    channel_group.add_argument(
        "--with-text",
        action="store_true",
        default=False,
        help="Start text REPL in the same process alongside the server.",
    )
    channel_group.add_argument(
        "--with-voice",
        action="store_true",
        default=False,
        help="Start voice channel in the same process alongside the server.",
    )
    return p


async def _run_text_channel() -> None:  # pragma: no cover
    """Spawn the text REPL in the same event loop so it publishes to the bus."""
    try:
        from apps.channels.text.repl import run_repl  # type: ignore[import-untyped]
        from apps.presence import get_default_bus

        await run_repl(bus=get_default_bus())
    except ImportError:
        import structlog

        _log = structlog.get_logger(__name__)
        _log.warning("presence.__main__.text_channel_import_failed")


async def _run_voice_channel() -> None:  # pragma: no cover
    """Spawn the voice channel in the same event loop."""
    try:
        from apps.channels.voice.channel import run_voice  # type: ignore[import-untyped]
        from apps.presence import get_default_bus

        await run_voice(bus=get_default_bus())
    except ImportError:
        import structlog

        _log = structlog.get_logger(__name__)
        _log.warning("presence.__main__.voice_channel_import_failed")


def main(argv: list[str] | None = None) -> None:
    """Parse args, validate, and launch uvicorn with optional channels."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    with_text = args.with_text
    with_voice = args.with_voice

    _validate_host(args.host)

    # sounddevice / faster-whisper / torch hold C++ threads that don't always
    # release on KeyboardInterrupt — the asyncio loop sees the interrupt but
    # cleanup hangs. Force-exit on SIGTERM and on a second Ctrl-C.
    import os as _os
    import signal as _signal
    import warnings as _warnings

    _warnings.filterwarnings("ignore", message=r".*leaked semaphore.*")

    def _force_exit(_signum: int, _frame: object) -> None:
        print("\n[프레즌스] 종료합니다.", file=sys.stderr)
        _os._exit(0)

    _signal.signal(_signal.SIGTERM, _force_exit)

    from apps.presence.server import create_app

    if with_text or with_voice:
        # Run uvicorn inside an asyncio event loop so channel tasks share it.
        import uvicorn

        async def _serve() -> None:
            app = create_app()
            config = uvicorn.Config(
                app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
            )
            server = uvicorn.Server(config)

            tasks = [asyncio.create_task(server.serve())]
            if with_text:
                tasks.append(asyncio.create_task(_run_text_channel()))
            if with_voice:
                tasks.append(asyncio.create_task(_run_voice_channel()))

            # Run until first task completes (server exit or channel EOF).
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

        try:
            asyncio.run(_serve())
        except KeyboardInterrupt:
            # Second Ctrl-C or normal interrupt — exit hard so torch/sounddevice
            # threads don't hold the process up.
            print("\n[프레즌스] 종료합니다.", file=sys.stderr)
            _os._exit(0)
    else:
        # Server-only — no channels in this process.
        # External channels drive the bus via HER_PRESENCE_URL → /publish.
        import uvicorn

        print(
            f"[프레즌스 서버] http://{args.host}:{args.port} 에서 시작합니다.\n"
            "  브라우저에서 열어 orb 를 확인하세요.\n"
            "  다른 터미널에서 채널을 연결하려면:\n"
            f"    HER_PRESENCE_URL=http://{args.host}:{args.port} make voice\n"
            f"    HER_PRESENCE_URL=http://{args.host}:{args.port} make text\n"
            "  또는: make voice-with-presence / make text-with-presence",
            file=sys.stderr,
        )

        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
