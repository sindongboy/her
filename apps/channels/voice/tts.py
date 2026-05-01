"""TTS module: Gemini TTS primary + macOS `say -v Yuna` fallback.

Implements streaming partial TTS per CLAUDE.md §2.1 and §6.1 (Path B).

Public API
----------
TTSConfig          – frozen config dataclass.
TTSError           – base exception for TTS failures.
GeminiTTS          – streaming Gemini TTS (gemini-2.5-flash-preview-tts).
SayFallbackTTS     – macOS `say` subprocess TTS.
TTS                – façade: Gemini primary → Say fallback with circuit breaker.

Sentence-boundary chunker
-------------------------
chunk_at_sentence_boundary(text_stream) — utility async generator used by both engines.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from apps.channels.voice.audio import AudioOutputStream

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Sentence boundary helpers
# ---------------------------------------------------------------------------

SENTENCE_END: frozenset[str] = frozenset(".!?。？！\n…")


async def chunk_at_sentence_boundary(
    text_stream: AsyncIterator[str],
    *,
    max_chars: int = 80,
) -> AsyncIterator[str]:
    """Yield text chunks bounded by sentence punctuation or *max_chars*.

    Accumulates tokens from *text_stream* and emits a chunk when:
    - a sentence-ending character is seen, OR
    - the buffer reaches *max_chars* (force flush), OR
    - the upstream closes (final flush of whatever remains).
    """
    buf: list[str] = []
    buf_len = 0

    async for token in text_stream:
        for ch in token:
            buf.append(ch)
            buf_len += 1
            if ch in SENTENCE_END or buf_len >= max_chars:
                chunk = "".join(buf).strip()
                if chunk:
                    yield chunk
                buf = []
                buf_len = 0

    if buf:
        chunk = "".join(buf).strip()
        if chunk:
            yield chunk


# ---------------------------------------------------------------------------
# Config & Error
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class TTSConfig:
    """TTS engine configuration.

    model_id: Gemini TTS model (CLAUDE.md §3.2 today's decision).
    voice: Gemini prebuilt voice name. Default "Kore" (Korean-optimised).
    speaking_rate: speed multiplier (1.0 = normal).
    sample_rate: output PCM sample rate in Hz (24 kHz = Gemini TTS default).
    """

    model_id: str = "gemini-2.5-flash-preview-tts"
    voice: str = "Kore"
    speaking_rate: float = 1.0
    sample_rate: int = 24_000


class TTSError(Exception):
    """Raised when a TTS engine fails to produce audio."""


# ---------------------------------------------------------------------------
# Transient vs. permanent error classification
# ---------------------------------------------------------------------------

_QUOTA_KEYWORDS = (
    "quota",
    "resource exhausted",
    "resource_exhausted",  # Gemini's actual error string uses underscore
    "rate limit",
    "rate_limit",
    "permission denied",
    "permission_denied",
    "unauthorized",
    "invalid_argument",  # 400 — usually means request shape is wrong; retrying won't help
    "invalid argument",
    "429",  # HTTP code itself
    "403",
    "401",
    "400",
)


def _is_transient(exc: BaseException) -> bool:
    """Return True if *exc* looks like a transient (retryable) error."""
    msg = str(exc).lower()
    return not any(kw in msg for kw in _QUOTA_KEYWORDS)


# ---------------------------------------------------------------------------
# GeminiTTS
# ---------------------------------------------------------------------------


class GeminiTTS:
    """Streaming Gemini TTS.

    Buffers text at sentence boundaries, sends one Gemini generate_content
    request per chunk, and yields PCM int16 audio bytes (24 kHz mono).
    """

    def __init__(
        self,
        *,
        config: TTSConfig = TTSConfig(),
        api_key: str | None = None,
        client: object | None = None,
        max_chars: int = 80,
    ) -> None:
        self._config = config
        self._max_chars = max_chars

        if client is not None:
            self._client = client
        else:
            try:
                from google import genai  # lazy — avoids import errors in tests

                resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
                self._client = genai.Client(api_key=resolved_key)
            except ImportError as exc:
                raise TTSError("google-genai not installed") from exc

    # ── public ──────────────────────────────────────────────────────────────

    async def synth(self, text: str) -> bytes:
        """Single-shot synthesis: full *text* → full PCM bytes."""
        chunks: list[bytes] = []
        async for chunk in self._synth_text(text):
            chunks.append(chunk)
        return b"".join(chunks)

    async def warmup(self) -> None:
        """Issue a tiny synth call so the first real utterance doesn't pay
        cold-start latency (which causes the TTS façade to fall back to the
        macOS `say` voice for that first sentence — see CLAUDE.md §3.2).
        Failure is swallowed; warmup is best-effort.

        We use a real Korean word ("준비") rather than a single character —
        Gemini TTS rejects punctuation-only inputs and retries until timeout.
        """
        try:
            await self.synth("준비")
        except Exception as exc:  # pragma: no cover
            logger.debug("gemini_tts_warmup_failed", error=str(exc))

    async def synth_stream(
        self, text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """Consume *text_stream* tokens and emit PCM bytes per sentence chunk."""
        async for text_chunk in chunk_at_sentence_boundary(
            text_stream, max_chars=self._max_chars
        ):
            async for pcm in self._synth_text(text_chunk):
                yield pcm

    # ── private ─────────────────────────────────────────────────────────────

    async def _synth_text(self, text: str) -> AsyncIterator[bytes]:
        """Call Gemini TTS for a single text chunk; yield PCM bytes."""
        pcm = await self._call_with_retry(text)
        yield pcm

    async def _call_gemini(self, text: str) -> bytes:
        """Raw Gemini API call. Returns PCM bytes for *text*.

        The Gemini TTS preview models occasionally interpret bare text as a
        chat prompt (returns 400 INVALID_ARGUMENT: "Model tried to generate
        text, but it should only be used for TTS"). Wrapping the text with an
        explicit TTS instruction makes the intent unambiguous.
        """
        try:
            from google.genai import types  # lazy import
        except ImportError as exc:
            raise TTSError("google-genai not installed") from exc

        config = self._config
        gen_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=config.voice,
                    )
                )
            ),
        )

        wrapped = f"다음 문장을 따뜻하고 자연스러운 한국어 목소리로 읽어주세요: {text}"

        try:
            resp = await asyncio.to_thread(
                self._client.models.generate_content,  # type: ignore[union-attr]
                model=config.model_id,
                contents=wrapped,
                config=gen_config,
            )
        except Exception as exc:
            raise TTSError(f"Gemini TTS call failed: {exc}") from exc

        # Defensive unpacking — preview model occasionally returns 200 OK with
        # an empty candidate (no content / no parts). Classify as permanent so
        # the façade falls back instead of burning retries on a doomed input.
        try:
            cand = resp.candidates[0]
        except (AttributeError, IndexError, TypeError):
            raise TTSError(
                f"Gemini TTS empty response (no candidates). finish_reason=unknown. invalid_argument."
            )
        finish = getattr(cand, "finish_reason", None)
        if cand.content is None or not getattr(cand.content, "parts", None):
            raise TTSError(
                f"Gemini TTS empty response (no content). finish_reason={finish}. invalid_argument."
            )
        try:
            pcm: bytes = cand.content.parts[0].inline_data.data
        except (AttributeError, IndexError, TypeError) as exc:
            raise TTSError(
                f"Gemini TTS malformed response: {exc}. invalid_argument."
            ) from exc
        if not pcm:
            raise TTSError(
                f"Gemini TTS empty audio. finish_reason={finish}. invalid_argument."
            )
        logger.debug("gemini_tts_synth_ok", chars=len(text))
        return pcm

    async def _call_with_retry(self, text: str) -> bytes:
        """Retry _call_gemini up to 3 times with exponential backoff for transient errors."""
        last_exc: BaseException | None = None
        for attempt in range(3):
            try:
                return await self._call_gemini(text)
            except TTSError as exc:
                if not _is_transient(exc):
                    logger.warning("gemini_tts_permanent_error", attempt=attempt)
                    raise
                last_exc = exc
                if attempt < 2:
                    wait = 1.0 * (2**attempt)  # 1s, 2s
                    logger.warning(
                        "gemini_tts_transient_retry",
                        attempt=attempt,
                        wait=wait,
                        error=str(exc),
                    )
                    await asyncio.sleep(wait)

        raise TTSError(f"Gemini TTS failed after 3 attempts") from last_exc


# ---------------------------------------------------------------------------
# SayFallbackTTS
# ---------------------------------------------------------------------------


class SayFallbackTTS:
    """macOS `say -v Yuna` subprocess TTS fallback.

    Outputs raw PCM int16 LE @ *sample_rate* Hz via AIFF + afconvert pipeline.
    Raises RuntimeError on non-macOS platforms.
    """

    def __init__(self, *, voice: str = "Yuna", sample_rate: int = 24_000) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("SayFallbackTTS is macOS-only (requires `say` command)")
        self._voice = voice
        self._sample_rate = sample_rate
        self._seq = 0

    # ── public ──────────────────────────────────────────────────────────────

    async def synth(self, text: str) -> bytes:
        """Synthesise *text* via `say`; return raw int16 PCM bytes."""
        return await self._say_to_pcm(text)

    async def synth_stream(
        self, text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """Sentence-chunked synthesis: yields PCM per chunk."""
        async for chunk in chunk_at_sentence_boundary(text_stream):
            pcm = await self._say_to_pcm(chunk)
            yield pcm

    async def speak(
        self,
        text_stream: AsyncIterator[str],
        output: "AudioOutputStream",
    ) -> None:
        """Stream *text_stream* through `say` → *output*. Used when this is
        the standalone TTS (i.e. ``tts_provider="say"``), not just a fallback."""
        async for pcm in self.synth_stream(text_stream):
            await output.write(pcm)
        await output.flush()

    async def speak_text(self, text: str, output: "AudioOutputStream") -> None:
        """Convenience for single-shot utterances."""
        pcm = await self._say_to_pcm(text)
        await output.write(pcm)
        await output.flush()

    async def warmup(self) -> None:
        """No-op — `say` is local subprocess; cold-start is just process spawn."""
        return None

    # ── private ─────────────────────────────────────────────────────────────

    async def _say_to_pcm(self, text: str) -> bytes:
        """Run `say` → AIFF tmp file → `afconvert` → raw int16 PCM bytes."""
        pid = os.getpid()
        self._seq += 1
        aiff_path = f"/tmp/her_say_{pid}_{self._seq}.aiff"
        raw_path = f"/tmp/her_say_{pid}_{self._seq}.raw"

        try:
            # Step 1: say → AIFF
            proc = await asyncio.create_subprocess_exec(
                "say",
                "-v", self._voice,
                "-o", aiff_path,
                "--",
                text,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                err = stderr.decode(errors="replace").strip()
                raise TTSError(f"`say` failed (rc={proc.returncode}): {err}")

            # Step 2: AIFF → raw int16 LE PCM via afconvert
            proc2 = await asyncio.create_subprocess_exec(
                "afconvert",
                "-f", "WAVE",
                "-d", f"LEI16@{self._sample_rate}",
                aiff_path,
                raw_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr2 = await proc2.communicate()
            if proc2.returncode != 0:
                err2 = stderr2.decode(errors="replace").strip()
                raise TTSError(f"`afconvert` failed (rc={proc2.returncode}): {err2}")

            # Read raw PCM, strip 44-byte WAV header
            pcm = await asyncio.to_thread(self._read_wav_data, raw_path)
            logger.debug("say_tts_synth_ok", chars=len(text))
            return pcm

        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"SayFallbackTTS error: {exc}") from exc
        finally:
            for path in (aiff_path, raw_path):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _read_wav_data(path: str) -> bytes:
        """Read a WAV file and return only the PCM data (skip 44-byte header)."""
        with open(path, "rb") as f:
            data = f.read()
        # Skip WAV header (44 bytes for standard PCM WAV)
        return data[44:]


# ---------------------------------------------------------------------------
# Circuit breaker state (internal)
# ---------------------------------------------------------------------------


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    # Initialised to -inf so the first failure starts a fresh window
    _failure_window_start: float = field(default_factory=lambda: -float("inf"))
    _FAILURE_THRESHOLD: int = 3
    _OPEN_DURATION: float = 300.0  # 5 min
    _WINDOW: float = 60.0          # failures counted within 60s

    def record_failure(self, now: float) -> None:
        """Record a failure; open circuit if threshold exceeded within window."""
        if now - self._failure_window_start > self._WINDOW:
            # Reset window: new window starts at *now*, counter resets to 0
            self.consecutive_failures = 0
            self._failure_window_start = now
        self.consecutive_failures += 1
        if self.consecutive_failures >= self._FAILURE_THRESHOLD:
            self.circuit_open_until = now + self._OPEN_DURATION
            logger.warning(
                "tts_circuit_open",
                open_until=self.circuit_open_until,
                failures=self.consecutive_failures,
            )

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def is_open(self, now: float) -> bool:
        return now < self.circuit_open_until


# Sentinel: distinguishes "caller passed None explicitly" vs "not provided"
_UNSET: object = object()


# ---------------------------------------------------------------------------
# TTS Façade
# ---------------------------------------------------------------------------


class TTS:
    """Façade: GeminiTTS primary, SayFallbackTTS fallback with circuit breaker.

    Circuit breaker: after 3 consecutive Gemini failures within 60s, the
    circuit opens and primary is skipped for 5 min before re-probing.

    Pass ``fallback=None`` explicitly to disable the fallback entirely.
    Omit ``fallback`` (or use the default) to auto-create SayFallbackTTS on macOS.
    """

    def __init__(
        self,
        *,
        primary: GeminiTTS | None = None,
        fallback: SayFallbackTTS | None | object = _UNSET,
        _clock: object | None = None,  # injectable for tests: callable() -> float
    ) -> None:
        self._primary = primary or GeminiTTS()

        if fallback is _UNSET:
            # Auto-create Say fallback on macOS
            self._fallback: SayFallbackTTS | None = None
            if sys.platform == "darwin":
                try:
                    self._fallback = SayFallbackTTS()
                except Exception:
                    pass
        else:
            # Caller passed an explicit value (including None = disabled)
            self._fallback = fallback  # type: ignore[assignment]
        self._circuit = _CircuitState()
        self._clock: object = _clock if _clock is not None else time.monotonic

    # ── public ──────────────────────────────────────────────────────────────

    async def speak(
        self,
        text_stream: AsyncIterator[str],
        output: "AudioOutputStream",
    ) -> None:
        """Stream *text_stream* through TTS → *output*.

        Delivers first audio as soon as Gemini emits the first sentence's PCM,
        achieving §2.1 streaming partial TTS SLA.
        """
        # Drain the text stream, collecting chunks and yielding audio
        async for pcm_chunk in self._synth_stream_with_fallback(text_stream):
            await output.write(pcm_chunk)
        await output.flush()

    async def speak_text(self, text: str, output: "AudioOutputStream") -> None:
        """Convenience for single-shot utterances."""
        async def _single() -> AsyncIterator[str]:
            yield text

        await self.speak(_single(), output)

    async def warmup(self) -> None:
        """Wake the primary TTS so the first real utterance doesn't fall back
        to macOS `say` due to cold-start latency. Best-effort."""
        await self._primary.warmup()

    # ── private ─────────────────────────────────────────────────────────────

    async def _synth_stream_with_fallback(
        self, text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """Drive sentence-chunked synthesis with per-chunk Gemini→fallback."""
        async for text_chunk in chunk_at_sentence_boundary(text_stream):
            pcm = await self._synth_one(text_chunk)
            yield pcm

    async def _synth_one(self, text: str) -> bytes:
        """Synthesise one text chunk, using primary unless circuit is open."""
        now: float = self._clock()  # type: ignore[call-arg]
        use_primary = not self._circuit.is_open(now)

        if use_primary:
            try:
                pcm = await self._primary.synth(text)
                self._circuit.record_success()
                return pcm
            except TTSError as exc:
                logger.warning("tts_primary_failed_falling_back", error=str(exc))
                self._circuit.record_failure(now)

        # Fallback path
        if self._fallback is not None:
            try:
                return await self._fallback.synth(text)
            except TTSError as exc:
                logger.error("tts_fallback_also_failed", error=str(exc))
                raise

        raise TTSError("TTS primary failed and no fallback available")
