"""Wake-word detection (VAD + faster-whisper polling) + quiet-mode utilities.

CLAUDE.md references:
  §2.3  Local-First — all wake detection runs on-device, no external calls.
  §2.4  Quiet mode — suppresses TTS; toggled by transcript phrases.
  §3.2  Wake Word row: VAD + faster-whisper 폴링 (재사용).
  §6.1  Pipeline: STT → Agent → TTS.  Wake uses same STT instance as utterance.
  §10   Mic ON only within a wake window; consent is the caller's responsibility.
  §12   Decision 2026-05-01: Porcupine replaced by Whisper-polling detector.

Public API
----------
WakeConfig            – frozen config dataclass.
WakeWordError         – raised on queue-closed or bad usage.
WakeDetector          – VAD-gated, Whisper-polled wake-word detector.
matches_wake          – pure utility: substring match of transcript vs. config.
QUIET_ON_PHRASES      – set of Korean phrases that activate quiet mode.
QUIET_OFF_PHRASES     – set of Korean phrases that deactivate quiet mode.
detect_quiet_intent   – returns 'on', 'off', or None from a transcript string.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class WakeWordError(Exception):
    """Raised when the wake detector encounters an unrecoverable condition."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class WakeConfig:
    """Configuration for the VAD+Whisper wake-word detector.

    Parameters
    ----------
    keyword:
        Primary Korean wake phrase, e.g. ``"자기야"``.  Any natural-language
        Korean phrase works — Whisper understands it natively.
    aliases:
        Additional phrases that are also accepted as wake words.
        Checked via the same substring matching as ``keyword``.
    case_sensitive:
        When ``False`` (default), matching is case-insensitive.  Korean has
        no case distinction, so this only matters for ASCII aliases.
    strip_whitespace:
        When ``True`` (default), leading/trailing whitespace and inter-word
        whitespace runs in the transcript are normalised before matching.
    """

    keyword: str = "자기야"
    aliases: tuple[str, ...] = ()
    case_sensitive: bool = False
    strip_whitespace: bool = True


# ---------------------------------------------------------------------------
# matches_wake
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str, *, strip_whitespace: bool, case_sensitive: bool) -> str:
    """Apply normalisation steps to *text* for matching."""
    # Unicode NFC normalisation (important for Korean composed forms)
    text = unicodedata.normalize("NFC", text)
    if strip_whitespace:
        text = _WHITESPACE_RE.sub(" ", text).strip()
    if not case_sensitive:
        text = text.lower()
    return text


def matches_wake(transcript: str, config: WakeConfig) -> bool:
    """Return True when *transcript* contains the configured wake phrase.

    Performs a substring search.  Both the transcript and the keyword/aliases
    are normalised the same way (whitespace collapse, NFC, optional lowercase)
    so minor STT spacing variations still match.

    Korean has no case, so ``case_sensitive`` is relevant only for ASCII
    aliases.  ``strip_whitespace=True`` collapses internal whitespace runs so
    "자기 야" matches "자기야" if both are collapsed to the same form.

    Parameters
    ----------
    transcript:
        Full text returned by STT.
    config:
        WakeConfig describing the keyword and matching options.

    Returns
    -------
    bool
        ``True`` when any keyword/alias is a substring of the normalised
        transcript.
    """
    opts = {"strip_whitespace": config.strip_whitespace, "case_sensitive": config.case_sensitive}
    norm_transcript = _normalise(transcript, **opts)
    phrases = (config.keyword,) + config.aliases
    for phrase in phrases:
        norm_phrase = _normalise(phrase, **opts)
        if norm_phrase and norm_phrase in norm_transcript:
            return True
    return False


# ---------------------------------------------------------------------------
# WakeDetector
# ---------------------------------------------------------------------------


class WakeDetector:
    """VAD-gated, Whisper-polled wake-word detector.

    Reads raw PCM frames from a fanout queue, uses
    ``speech_segments_from_queue`` (from interrupt.py) to extract speech
    utterances, runs STT on each utterance, and returns as soon as the
    transcript matches the configured wake phrase.

    No external API keys are required — detection runs fully on-device using
    the same ``STT`` and ``VAD`` instances used for normal utterance handling.

    Parameters
    ----------
    stt:
        STT instance (must expose ``async transcribe(pcm_bytes) -> str``).
    vad:
        VAD instance (must expose ``is_speech(frame)`` and ``reset()``).
    config:
        WakeConfig describing keyword and matching options.
    max_segment_ms:
        Hard upper bound for wake-utterance length in milliseconds.  Keep
        short (default 4 000 ms) because wake phrases are brief.
    max_silence_ms:
        Silence duration (ms) that terminates a wake utterance.  Shorter
        than normal utterances (default 400 ms) to react fast.
    """

    def __init__(
        self,
        *,
        stt: Any,
        vad: Any,
        config: WakeConfig = WakeConfig(),
        max_segment_ms: int = 4_000,
        max_silence_ms: int = 400,
    ) -> None:
        self._stt = stt
        self._vad = vad
        self._config = config
        self._max_segment_ms = max_segment_ms
        self._max_silence_ms = max_silence_ms
        log.info(
            "wake_detector_ready",
            keyword=config.keyword,
            aliases=list(config.aliases),
            backend="faster-whisper",
        )

    # -- public properties ---------------------------------------------------

    @property
    def keyword(self) -> str:
        """Primary wake keyword (for status/logging)."""
        return self._config.keyword

    # -- async wait ----------------------------------------------------------

    async def wait_for_wake(self, queue: asyncio.Queue[bytes | None]) -> str:
        """Block until a wake-matching utterance is heard.

        Iterates over VAD-extracted speech segments from *queue*, runs STT on
        each, and returns as soon as :func:`matches_wake` is satisfied.

        Parameters
        ----------
        queue:
            ``asyncio.Queue[bytes | None]`` fed by ``_FrameFanout.subscribe()``.
            A ``None`` sentinel signals end-of-stream.

        Returns
        -------
        str
            The full transcript of the matching utterance.  The caller may
            use any trailing text after the wake phrase as the first user
            input (e.g. ``"자기야 오늘 날씨 어때?"``).

        Raises
        ------
        WakeWordError
            When the queue is closed (None sentinel received) before any wake
            word is detected.
        asyncio.CancelledError
            If this coroutine is cancelled externally.
        """
        from apps.channels.voice.interrupt import speech_segments_from_queue

        log.debug("wake_detector.waiting", keyword=self._config.keyword)

        async for segment_bytes in speech_segments_from_queue(
            queue,
            self._vad,
            min_speech_ms=100,  # short — wake phrases can be very brief
            max_silence_ms=self._max_silence_ms,
            max_segment_ms=self._max_segment_ms,
        ):
            # Run STT in thread (already wrapped in asyncio.to_thread by STT)
            try:
                transcript = await self._stt.transcribe(segment_bytes)
            except Exception as exc:
                log.warning("wake_detector.stt_failed", error=str(exc))
                continue

            if not transcript.strip():
                log.debug("wake_detector.empty_transcript")
                continue

            log.debug("wake_detector.candidate")  # no transcript in log — PII

            if matches_wake(transcript, self._config):
                log.info("wake_word_detected", keyword=self._config.keyword)
                return transcript

            log.debug("wake_detector.no_match")

        # Queue was exhausted (None sentinel received) without a match
        raise WakeWordError("queue closed before wake")

    # -- cleanup -------------------------------------------------------------

    def close(self) -> None:
        """No-op: STT and VAD are owned by the caller.

        Retained for API compatibility with the previous Porcupine detector
        so channel.py's ``wake_detector.close()`` call remains valid.
        """
        log.debug("wake_detector_closed")


# ---------------------------------------------------------------------------
# Quiet-mode utilities
# ---------------------------------------------------------------------------

QUIET_ON_PHRASES: frozenset[str] = frozenset(
    {
        "조용히 해",
        "조용 모드",
        "조용히",
    }
)

QUIET_OFF_PHRASES: frozenset[str] = frozenset(
    {
        "이제 말해도 돼",
        "다시 말해줘",
        "조용 모드 해제",
    }
)


def detect_quiet_intent(transcript: str) -> str | None:
    """Return ``'on'``, ``'off'``, or ``None`` based on substring match.

    Matching is case-insensitive and ignores leading/trailing whitespace.
    The transcript is checked against ``QUIET_OFF_PHRASES`` first so that a
    phrase like "조용 모드 해제" is not accidentally matched by the shorter
    "조용 모드" on-phrase.

    Returns
    -------
    ``'on'``
        The transcript contains a quiet-on phrase.
    ``'off'``
        The transcript contains a quiet-off phrase.
    ``None``
        Neither set matched.
    """
    text = transcript.strip().lower()

    for phrase in QUIET_OFF_PHRASES:
        if phrase.lower() in text:
            return "off"

    for phrase in QUIET_ON_PHRASES:
        if phrase.lower() in text:
            return "on"

    return None
