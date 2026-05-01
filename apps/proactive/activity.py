"""Activity tracker — records last user interaction time to a JSON sidecar.

Channels call ``record_activity(channel)`` on each user turn.
SilenceTrigger calls ``read_last_activity()`` to check elapsed time.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

_DEFAULT_PATH = Path.home() / ".her" / "activity.json"


def record_activity(channel: str, *, path: Path | None = None) -> None:
    """Atomic write of {channel, ts} to ~/.her/activity.json.

    Called on user input from both voice and text channels.
    Uses rename-based atomic write to avoid corrupt reads.
    """
    target = path or _DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).isoformat()
    data = json.dumps({"channel": channel, "ts": ts}, ensure_ascii=False)

    tmp = target.with_suffix(".json.tmp")
    try:
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        log.warning("activity_write_failed", error=str(exc))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def read_last_activity(*, path: Path | None = None) -> tuple[str, datetime] | None:
    """Return (channel, dt) of the last recorded user activity, or None.

    Reads ~/.her/activity.json written by ``record_activity``.
    """
    target = path or _DEFAULT_PATH
    if not target.exists():
        return None
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        channel: str = raw.get("channel", "unknown")
        ts_str: str = raw.get("ts", "")
        dt = datetime.fromisoformat(ts_str)
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return channel, dt
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        log.warning("activity_read_failed", error=str(exc))
        return None
