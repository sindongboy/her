"""Size-based log rotation using stdlib only.

Rotates when ``log_path`` reaches or exceeds ``max_bytes``:
    her.log   → her.log.1  (newest backup)
    her.log.1 → her.log.2
    ...
    her.log.{keep-1} → her.log.{keep}  (oldest backup, rest dropped)

After rotation ``her.log`` is recreated as an empty file so the
calling process can continue writing to the same path.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["maybe_rotate"]


def maybe_rotate(
    log_path: Path,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    keep: int = 5,
) -> bool:
    """Rotate ``log_path`` if its size >= ``max_bytes``.

    Parameters
    ----------
    log_path:  Absolute path to the active log file.
    max_bytes: Size threshold in bytes (default 10 MiB).
    keep:      Number of rotated backups to retain (default 5).

    Returns
    -------
    True if rotation happened, False otherwise.
    """
    if not log_path.exists():
        return False

    try:
        size = os.stat(log_path).st_size
    except OSError:
        return False

    if size < max_bytes:
        return False

    _rotate(log_path, keep=keep)
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rotate(log_path: Path, keep: int) -> None:
    """Shift existing numbered backups and move current log to .1."""
    # Drop the oldest backup that would exceed 'keep'
    oldest = log_path.with_suffix(f".log.{keep}") if log_path.suffix == ".log" else Path(
        str(log_path) + f".{keep}"
    )
    # Build numbered paths: log_path.1 … log_path.{keep}
    numbered = [_numbered(log_path, n) for n in range(1, keep + 1)]

    # Remove the oldest
    numbered[-1].unlink(missing_ok=True)

    # Shift: log.{k-1} → log.{k}, …, log.1 → log.2
    for i in range(keep - 1, 0, -1):
        src = numbered[i - 1]
        dst = numbered[i]
        if src.exists():
            shutil.move(str(src), str(dst))

    # Move current log → .1
    shutil.move(str(log_path), str(numbered[0]))

    # Recreate empty log file so callers can continue writing to the same path
    log_path.touch()


def _numbered(base: Path, n: int) -> Path:
    """Return ``base.{n}`` (e.g. her.log → her.log.1)."""
    return Path(str(base) + f".{n}")
