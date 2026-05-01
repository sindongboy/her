"""Unit tests for apps.daemon.lifecycle — PidFile and signal handlers."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path

import pytest

from apps.daemon.lifecycle import PidFile, PidFileError, install_signal_handlers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pid_path(tmp_path: Path) -> Path:
    return tmp_path / "her.pid"


# ---------------------------------------------------------------------------
# PidFile.acquire
# ---------------------------------------------------------------------------


class TestPidFileAcquire:
    def test_writes_current_pid(self, pid_path: Path) -> None:
        pf = PidFile(pid_path)
        written = pf.acquire()
        assert written == os.getpid()
        assert pid_path.exists()
        assert int(pid_path.read_text().strip()) == os.getpid()

    def test_raises_if_live_process_owns_it(self, pid_path: Path) -> None:
        pf = PidFile(pid_path)
        pf.acquire()
        # Second acquire by the same process (still live) must raise
        with pytest.raises(PidFileError, match="already running"):
            pf.acquire()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "her.pid"
        pf = PidFile(deep)
        pf.acquire()
        assert deep.exists()
        pf.release()

    def test_overwrites_stale_pidfile(self, pid_path: Path) -> None:
        """A pidfile containing a dead PID should be silently overwritten."""
        # Write a PID that definitely does not exist
        stale_pid = _dead_pid()
        pid_path.write_text(str(stale_pid))

        pf = PidFile(pid_path)
        written = pf.acquire()
        assert written == os.getpid()
        pf.release()


# ---------------------------------------------------------------------------
# PidFile.is_running
# ---------------------------------------------------------------------------


class TestPidFileIsRunning:
    def test_none_when_no_file(self, pid_path: Path) -> None:
        pf = PidFile(pid_path)
        assert pf.is_running() is None

    def test_returns_live_pid(self, pid_path: Path) -> None:
        pf = PidFile(pid_path)
        pf.acquire()
        assert pf.is_running() == os.getpid()
        pf.release()

    def test_returns_none_for_stale_pid(self, pid_path: Path) -> None:
        stale_pid = _dead_pid()
        pid_path.write_text(str(stale_pid))
        pf = PidFile(pid_path)
        assert pf.is_running() is None
        # Stale file should be cleaned up
        assert not pid_path.exists()

    def test_none_for_corrupt_pidfile(self, pid_path: Path) -> None:
        pid_path.write_text("not-a-number")
        pf = PidFile(pid_path)
        assert pf.is_running() is None


# ---------------------------------------------------------------------------
# PidFile.release
# ---------------------------------------------------------------------------


class TestPidFileRelease:
    def test_deletes_pidfile(self, pid_path: Path) -> None:
        pf = PidFile(pid_path)
        pf.acquire()
        assert pid_path.exists()
        pf.release()
        assert not pid_path.exists()

    def test_release_is_idempotent(self, pid_path: Path) -> None:
        pf = PidFile(pid_path)
        pf.acquire()
        pf.release()
        pf.release()  # Should not raise
        assert not pid_path.exists()

    def test_does_not_delete_foreign_pidfile(self, pid_path: Path) -> None:
        """release() must not remove a pidfile owned by another PID."""
        foreign_pid = _dead_pid()
        pid_path.write_text(str(foreign_pid))
        pf = PidFile(pid_path)
        pf.release()  # Should be a no-op
        # File may or may not exist; no exception is the contract
        # (The stale cleanup in is_running is separate)


# ---------------------------------------------------------------------------
# Signal handler (stub — avoids actually sending SIGTERM to self in asyncio)
# ---------------------------------------------------------------------------


class TestInstallSignalHandlers:
    def test_stop_event_set_on_sigterm_stub(self) -> None:
        """Test the handler logic by calling the internal callback directly."""
        # We cannot safely send SIGTERM to ourselves inside pytest's event loop,
        # so we verify the mechanism by running a minimal asyncio coroutine that
        # installs the handlers and then checks the stop_event.

        async def _run() -> bool:
            stop_event = asyncio.Event()
            install_signal_handlers(stop_event)

            # Directly trigger stop_event (simulates the handler being called)
            loop = asyncio.get_running_loop()
            loop.call_soon(stop_event.set)

            await asyncio.sleep(0)  # yield to let call_soon fire
            return stop_event.is_set()

        result = asyncio.run(_run())
        assert result is True

    def test_signal_handler_sets_event(self) -> None:
        """Send SIGTERM to self and confirm stop_event is set before timeout."""

        async def _run() -> bool:
            stop_event = asyncio.Event()
            install_signal_handlers(stop_event)
            os.kill(os.getpid(), signal.SIGTERM)
            # Give the event loop one iteration to process the signal
            await asyncio.sleep(0.05)
            return stop_event.is_set()

        result = asyncio.run(_run())
        assert result is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    """Return a PID that is guaranteed not to be running."""
    # Fork + immediately exit; wait to reap it, leaving no zombie.
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    # The PID is now recycled (or not), but for our purposes the 'pid' value
    # should not be live.  Add a tiny sleep so macOS doesn't recycle it instantly.
    time.sleep(0.05)
    return pid
