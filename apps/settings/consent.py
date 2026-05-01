"""Mic consent helpers — interactive first-run prompt + grant utilities.

IMPORTANT (CLAUDE.md §10): never log consent text or raw user input.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from apps.settings.store import Settings

logger = structlog.get_logger(__name__)

_CONSENT_ACCEPTED = frozenset({"동의", "yes", "y"})
_CONSENT_DENIED = frozenset({"취소", "no", "n"})

_DISCLOSURE_LINES = [
    "her 음성 비서는 대기 모드에서 마이크를 항상 듣고, 깨우는 말('컴퓨터' 등)이 들릴 때만 동작합니다.",
    "녹음/저장은 활성 대화 중에만 수행되며, 모든 데이터는 로컬에 보관됩니다 (CLAUDE.md §2.3).",
]


def is_consent_granted(settings: "Settings") -> bool:
    """Return True if mic consent has already been granted."""
    return settings.mic_consent_granted


def grant_mic_consent(
    settings: "Settings",
    *,
    when: datetime | None = None,
) -> "Settings":
    """Pure function — return a new Settings with consent fields set.

    Does NOT save to disk.  Caller is responsible for calling save_settings().
    """
    from dataclasses import replace

    ts = (when or datetime.now(timezone.utc)).isoformat()
    updated = replace(settings, mic_consent_granted=True, mic_consent_at=ts)
    logger.info("mic_consent_granted", at=ts)
    return updated


def prompt_mic_consent(
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[..., None] = print,
) -> bool:
    """Interactive Korean prompt.

    Returns True if user typed '동의' / 'yes' / 'y' (case-insensitive).
    NEVER logs the actual user response (§10 privacy).
    """
    output_fn("")
    for line in _DISCLOSURE_LINES:
        output_fn(line)
    output_fn("")

    try:
        raw = input_fn("동의하시겠습니까? (동의 / 취소): ")
    except (EOFError, KeyboardInterrupt):
        output_fn("")
        logger.info("mic_consent_prompt_cancelled")
        return False

    answer = raw.strip().lower()

    if answer in _CONSENT_ACCEPTED:
        output_fn("마이크 상시 청취에 동의하셨습니다. 설정을 저장합니다.")
        logger.info("mic_consent_prompt_result", accepted=True)
        return True

    output_fn("동의하지 않으셨습니다. 데몬 모드 없이 her 를 사용할 수 있습니다.")
    logger.info("mic_consent_prompt_result", accepted=False)
    return False
