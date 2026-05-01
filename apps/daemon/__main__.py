"""Entry point for the her background daemon.

Usage (direct):
    python -m apps.daemon [--db PATH] [--log-dir PATH]

Usage (via launchd):
    See infra/launchd/com.her.assistant.plist.template

Environment variables (required at runtime):
    GEMINI_API_KEY          Gemini API key (only required key)
    HER_DB_PATH             Override database path (optional)
    HER_DATA_DIR            Override data directory root (optional)
    HER_STT_MODEL           Whisper model size for STT + wake detection (optional, default: medium)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import structlog

from apps.daemon.lifecycle import PidFile, install_signal_handlers, load_settings
from apps.daemon.logrotate import maybe_rotate

# ---------------------------------------------------------------------------
# Structlog setup — file + timestamped output
# ---------------------------------------------------------------------------


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "her.log"

    # Rotate before opening if needed
    maybe_rotate(log_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m apps.daemon",
        description="her background daemon (Phase 3)",
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        type=Path,
        default=Path(os.environ.get("HER_DB_PATH", "data/db.sqlite")),
        help="Path to SQLite database (default: data/db.sqlite)",
    )
    parser.add_argument(
        "--log-dir",
        dest="log_dir",
        type=Path,
        default=Path(os.environ.get("HER_LOG_DIR", str(Path.home() / ".her" / "logs"))),
        help="Directory for log files (default: ~/.her/logs/)",
    )
    parser.add_argument(
        "--settings",
        dest="settings_path",
        type=Path,
        default=None,
        help="Path to settings.toml (default: ~/.her/settings.toml)",
    )
    parser.add_argument(
        "--pid-file",
        dest="pid_path",
        type=Path,
        default=Path.home() / ".her" / "her.pid",
        help="Path to PID file (default: ~/.her/her.pid)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Logging must come first (before any structlog calls below)
    _configure_logging(args.log_dir)

    logger = structlog.get_logger("apps.daemon.__main__")
    logger.info("daemon.booting", db=str(args.db_path), log_dir=str(args.log_dir))

    # ── Settings ────────────────────────────────────────────────────────
    settings = load_settings(args.settings_path)
    logger.info(
        "daemon.settings_loaded",
        consent=settings.mic_consent_granted,
        quiet_mode=settings.quiet_mode,
        wake_keyword=settings.wake_keyword,
    )

    # ── PidFile ──────────────────────────────────────────────────────────
    pidfile = PidFile(args.pid_path)
    try:
        pidfile.acquire()
    except Exception as exc:  # PidFileError or OS errors
        logger.error("daemon.already_running", error=str(exc))
        sys.exit(1)

    # ── Async entry ──────────────────────────────────────────────────────
    async def _run() -> None:
        from apps.daemon.runner import run_daemon

        stop_event = asyncio.Event()
        install_signal_handlers(stop_event)
        try:
            await run_daemon(
                db_path=args.db_path,
                log_dir=args.log_dir,
                settings=settings,
                stop_event=stop_event,
            )
        finally:
            pidfile.release()
            logger.info("daemon.exited_cleanly")

    try:
        asyncio.run(_run())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        logger.info("daemon.keyboard_interrupt")
    except Exception as exc:
        logger.exception("daemon.unhandled_exception", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
