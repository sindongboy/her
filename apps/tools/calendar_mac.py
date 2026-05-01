"""macOS Calendar.app integration via AppleScript (osascript).

Reads events from all calendars accessible through Calendar.app (iCloud,
Google, on-device). Requires Automation → Calendar TCC permission for the
running app (Terminal, Cursor, etc.).

CLAUDE.md references:
  §1.2  Calendar use-case (proactive scheduling)
  §2.3  Local-first — data stays on device, never uploaded
  §4    Tool Registry
  §3.2  asyncio subprocess only
"""

from __future__ import annotations

import asyncio
import platform
import sys
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

# ── AppleScript template ────────────────────────────────────────────────────

_APPLESCRIPT = """\
on iso(d)
    set y to (year of d) as integer as string
    set m to text -2 thru -1 of ("0" & ((month of d) as integer))
    set dd to text -2 thru -1 of ("0" & (day of d))
    set hh to text -2 thru -1 of ("0" & (hours of d))
    set mm to text -2 thru -1 of ("0" & (minutes of d))
    set ss to text -2 thru -1 of ("0" & (seconds of d))
    return y & "-" & m & "-" & dd & "T" & hh & ":" & mm & ":" & ss
end iso

set startD to (current date)
set hours of startD to 0
set minutes of startD to 0
set seconds of startD to 0
set endD to startD + (__DAYS__ * 86400)

set out to ""
tell application "Calendar"
    repeat with c in calendars
        try
            set evts to (every event of c whose start date >= startD and start date < endD)
            repeat with e in evts
                set ttl to summary of e
                set cn to title of c
                set ad to allday event of e as string
                set sd to my iso(start date of e)
                set ed to my iso(end date of e)
                set out to out & ttl & "|||" & cn & "|||" & ad & "|||" & sd & "|||" & ed & linefeed
            end repeat
        on error
            -- skip calendars we can't read
        end try
    end repeat
end tell
return out
"""

# Markers that indicate a TCC / permission denial
_TCC_MARKERS = (
    "Not authorized to send Apple events to Calendar",
    "(-1743)",
    "execution error",
)

_TCC_INSTRUCTION = (
    "macOS Calendar 접근 권한이 필요합니다.\n"
    "시스템 설정 → 개인정보 및 보안 → 자동화 → "
    "(터미널 또는 Cursor 또는 사용 중인 앱) → Calendar 토글을 켜세요.\n"
    "또는 Calendar.app 을 한 번 실행한 뒤 다시 시도하세요."
)


# ── Public types ────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class CalendarEvent:
    title: str
    calendar_name: str
    starts_at_iso: str  # local time, "%Y-%m-%dT%H:%M:%S"
    ends_at_iso: str | None
    is_all_day: bool


class CalendarUnavailable(Exception):
    """Raised when macOS Calendar can't be queried.

    The error message is Korean and tells the user how to fix the issue
    (TCC permission missing, non-macOS platform, timeout, etc.).
    """


# ── Internal helpers ─────────────────────────────────────────────────────────


def _detect_tcc_error(stderr: str, returncode: int) -> bool:
    """Return True if stderr / returncode indicates a TCC permission denial."""
    if returncode != 0:
        for marker in _TCC_MARKERS:
            if marker in stderr:
                return True
    return False


def _parse_line(line: str) -> CalendarEvent | None:
    """Parse a single pipe-separated output line.  Returns None on bad format."""
    parts = line.strip().split("|||")
    if len(parts) != 5:
        return None
    title, calendar_name, ad_str, start_iso, end_iso = parts
    is_all_day = ad_str.strip().lower() == "true"
    end: str | None = end_iso.strip() if end_iso.strip() else None
    return CalendarEvent(
        title=title.strip(),
        calendar_name=calendar_name.strip(),
        starts_at_iso=start_iso.strip(),
        ends_at_iso=end,
        is_all_day=is_all_day,
    )


# ── Public API ───────────────────────────────────────────────────────────────


async def get_events(
    *,
    days_ahead: int = 2,
    max_events: int = 50,
    timeout_s: float = 25.0,
) -> list[CalendarEvent]:
    """Read events from macOS Calendar.app via osascript.

    Returns events from today 00:00 (local) until +days_ahead days.
    Sorted ascending by start time.

    Raises:
        CalendarUnavailable: Non-macOS, no TCC permission,
            Calendar.app not running, or timeout.
    """
    if platform.system() != "Darwin":
        # Safe no-op on non-macOS — tests, Linux CI, etc.
        log.debug("calendar.non_darwin_skip")
        return []

    script = _APPLESCRIPT.replace("__DAYS__", str(int(days_ahead)))

    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        if proc is not None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        raise CalendarUnavailable("Calendar 조회 시간 초과 (Calendar.app 응답 없음)")

    stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
    stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""

    if _detect_tcc_error(stderr, proc.returncode):
        raise CalendarUnavailable(_TCC_INSTRUCTION)

    if proc.returncode != 0:
        # Some other osascript error — log stderr for debugging (no PII)
        log.warning("calendar.osascript_error", returncode=proc.returncode)
        raise CalendarUnavailable(f"osascript 실행 오류 (returncode={proc.returncode})")

    events: list[CalendarEvent] = []
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        event = _parse_line(raw_line)
        if event is None:
            log.debug("calendar.skip_bad_line")
            continue
        events.append(event)

    # Sort ascending by start ISO (lexicographic sort works for "YYYY-MM-DDTHH:MM:SS")
    events.sort(key=lambda e: e.starts_at_iso)

    trimmed = events[:max_events]
    log.debug("calendar.events_fetched", count=len(trimmed))
    return trimmed
