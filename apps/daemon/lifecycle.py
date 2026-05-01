"""Daemon lifecycle helpers — pidfile management and signal handling.

Provides PidFile for atomic pid tracking and install_signal_handlers
for graceful asyncio shutdown on SIGTERM / SIGINT.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings shim — used when apps.settings is not yet available (test isolation)
# ---------------------------------------------------------------------------
try:
    from apps.settings import Settings, load_settings  # type: ignore[import]
except ImportError:  # pragma: no cover — removed once consent-eng lands

    from dataclasses import dataclass, field

    @dataclass(slots=True)
    class Settings:  # type: ignore[no-redef]
        mic_consent_granted: bool = False
        mic_consent_at: str | None = None
        quiet_mode: bool = False
        wake_keyword: str = "computer"
        wake_keyword_path: str | None = None

    def load_settings(path: Path | None = None) -> Settings:  # type: ignore[misc]
        """Return a default Settings when apps.settings is unavailable."""
        return Settings()


__all__ = ["PidFile", "install_signal_handlers", "Settings", "load_settings"]

# ---------------------------------------------------------------------------
# PidFile
# ---------------------------------------------------------------------------

_DEFAULT_PID_PATH = Path.home() / ".her" / "her.pid"


class PidFileError(RuntimeError):
    """Raised when a live process already owns the pidfile."""


class PidFile:
    """Manage ~/.her/her.pid with atomic create, stale-pid detection, and release.

    Usage::

        pf = PidFile(Path.home() / ".her" / "her.pid")
        pf.acquire()          # write current PID (raises if another live proc owns it)
        ...
        pf.release()          # delete pidfile on clean shutdown
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    # ------------------------------------------------------------------
    def acquire(self) -> int:
        """Write current PID to the pidfile.

        Raises PidFileError if another **live** process already owns the file.
        Overwrites stale pidfiles (process no longer exists).
        Returns the written PID (os.getpid()).
        """
        live = self.is_running()
        if live is not None:
            raise PidFileError(
                f"her daemon is already running (pid {live}). "
                "Run 'bin/her stop' first."
            )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        # Write atomically: write to tmp then rename
        tmp = self._path.with_suffix(".pid.tmp")
        tmp.write_text(str(pid))
        tmp.rename(self._path)
        logger.info("pidfile.acquired", path=str(self._path), pid=pid)
        return pid

    # ------------------------------------------------------------------
    def is_running(self) -> int | None:
        """Return the live PID from the pidfile, or None if absent / stale."""
        if not self._path.exists():
            return None
        try:
            raw = self._path.read_text().strip()
            pid = int(raw)
        except (ValueError, OSError):
            return None

        try:
            os.kill(pid, 0)  # signal 0 = probe only
        except ProcessLookupError:
            # Stale pidfile — remove it silently
            self._path.unlink(missing_ok=True)
            return None
        except PermissionError:
            # Process exists but we don't own it (edge case on macOS)
            return pid

        return pid

    # ------------------------------------------------------------------
    def release(self) -> None:
        """Delete the pidfile if it belongs to the current process."""
        try:
            raw = self._path.read_text().strip()
            if int(raw) == os.getpid():
                self._path.unlink(missing_ok=True)
                logger.info("pidfile.released", path=str(self._path))
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------


def install_signal_handlers(stop_event: asyncio.Event) -> None:
    """Register SIGTERM and SIGINT handlers that set *stop_event*.

    Must be called from within a running asyncio event loop (uses
    loop.add_signal_handler which is Unix-only).
    """
    loop = asyncio.get_running_loop()

    def _handler(sig: int) -> None:
        sig_name = signal.Signals(sig).name
        logger.info("signal.received", signal=sig_name)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handler, sig)
        except (NotImplementedError, RuntimeError) as exc:
            # Windows or non-main-thread — fall back to signal.signal
            logging.warning("Could not install asyncio signal handler for %s: %s", sig, exc)
            signal.signal(sig, lambda s, _f: _handler(s))
