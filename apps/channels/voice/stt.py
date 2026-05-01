# Runtime deps: faster-whisper, numpy. voice-ch-eng adds to pyproject.
"""Speech-to-Text using faster-whisper.

Public API
----------
STT         – async transcriber; lazy-loads the Whisper model on first use.

Configuration
-------------
HER_STT_MODEL   env var   Whisper model size (default: "medium").
                           Valid values: tiny | base | small | medium | large-v3

Per CLAUDE.md §3.2 (STT row) and §6.1 (Voice Channel, Path B pipeline).
"""

from __future__ import annotations

import asyncio
import os
import time

import numpy as np
import structlog

from apps.channels.voice.audio import AudioFormat

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL_SIZE = "medium"
_ENV_MODEL_KEY = "HER_STT_MODEL"


class STT:
    """Async speech-to-text transcriber backed by faster-whisper.

    The Whisper model is loaded lazily on the first :meth:`transcribe` or
    :meth:`warmup` call to keep import time cheap.

    Parameters
    ----------
    model_size:
        Whisper model size.  Falls back to ``HER_STT_MODEL`` env var, then
        ``"medium"``.
    language:
        BCP-47 language code passed to Whisper.  Default ``"ko"`` (Korean).
    compute_type:
        CTranslate2 quantisation type.  ``"int8"`` is the recommended default
        on Apple Silicon CPU.
    """

    def __init__(
        self,
        *,
        model_size: str | None = None,
        language: str = "ko",
        compute_type: str = "int8",
    ) -> None:
        self._model_size: str = (
            model_size or os.environ.get(_ENV_MODEL_KEY) or _DEFAULT_MODEL_SIZE
        )
        self._language = language
        self._compute_type = compute_type
        self._model: object | None = None  # loaded lazily

    # ── public ──────────────────────────────────────────────────────────

    async def transcribe(
        self,
        pcm_bytes: bytes,
        *,
        fmt: AudioFormat = AudioFormat(),
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribe a PCM int16 blob to text.

        Converts raw int16 bytes → float32 numpy array in ``[-1, 1]``, then
        runs Whisper inference in a thread pool to avoid blocking the event loop.

        Parameters
        ----------
        pcm_bytes:
            Raw PCM audio as int16 little-endian bytes.
        fmt:
            Describes the audio format (sample rate, channels, width).
        initial_prompt:
            Optional text hint injected into Whisper's decoder — use for
            family names / custom vocabulary (CLAUDE.md §6.1 STT후처리).

        Returns
        -------
        str
            Transcription text, stripped of leading/trailing whitespace.
            Returns ``""`` when Whisper produces no output — never raises on
            empty/silent input.
        """
        if not pcm_bytes:
            return ""

        await self._ensure_model()
        audio_array = _pcm_to_float32(pcm_bytes)
        return await asyncio.to_thread(
            self._run_transcribe, audio_array, initial_prompt
        )

    async def warmup(self) -> None:
        """Load model weights by running a tiny silent inference.

        Call this before the mic loop starts to amortise the first-load latency.
        """
        await self._ensure_model()
        silence = np.zeros(3200, dtype=np.float32)  # 200 ms at 16 kHz
        t0 = time.monotonic()
        await asyncio.to_thread(self._run_transcribe, silence, None)
        logger.info("stt_warmup_done", elapsed_s=round(time.monotonic() - t0, 2))

    # ── private ─────────────────────────────────────────────────────────

    async def _ensure_model(self) -> None:
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model)

    def _load_model(self) -> object:
        from faster_whisper import WhisperModel  # lazy import

        t0 = time.monotonic()
        model = WhisperModel(
            self._model_size,
            device="cpu",
            compute_type=self._compute_type,
        )
        elapsed = round(time.monotonic() - t0, 2)
        logger.info(
            "stt_model_loaded",
            model_size=self._model_size,
            compute_type=self._compute_type,
            elapsed_s=elapsed,
        )
        return model

    def _run_transcribe(
        self,
        audio: np.ndarray,
        initial_prompt: str | None,
    ) -> str:
        """Blocking Whisper call — must be run in a thread via asyncio.to_thread."""
        segments, _info = self._model.transcribe(  # type: ignore[union-attr]
            audio,
            language=self._language,
            initial_prompt=initial_prompt,
            beam_size=5,
            vad_filter=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw int16 PCM bytes to a float32 array in ``[-1.0, 1.0]``."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0
