# Runtime deps: silero-vad, numpy. voice-ch-eng adds to pyproject.
"""Voice Activity Detection wrapping silero-vad v5.

Public API
----------
VAD                 – frame-level speech/silence classifier.
speech_segments     – async generator; emits one PCM blob per utterance.
VADUnavailableError – raised when the silero model cannot be loaded.

Per CLAUDE.md §6.1 (Voice Channel, Path B pipeline) and §3.2 (STT row).
"""

from __future__ import annotations

import asyncio
import collections
import struct
from typing import AsyncIterator

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Silero-vad v5 requires exactly 512-sample frames at 16 kHz.
_SILERO_FRAME_SAMPLES: int = 512
_SILERO_SAMPLE_RATE: int = 16_000
_FRAME_BYTES: int = _SILERO_FRAME_SAMPLES * 2  # int16 = 2 bytes/sample


class VADUnavailableError(RuntimeError):
    """Raised when the silero-vad model cannot be loaded."""


class VAD:
    """Frame-level voice-activity detector wrapping silero-vad v5.

    Each call to :meth:`is_speech` consumes exactly one 32 ms / 512-sample
    PCM int16 frame (1024 bytes).  silero keeps its own recurrent state;
    :meth:`reset` clears it between utterances.
    """

    def __init__(
        self,
        *,
        sample_rate: int = _SILERO_SAMPLE_RATE,
        threshold: float = 0.5,
        min_silence_ms: int = 700,
        speech_pad_ms: int = 100,
    ) -> None:
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_silence_ms = min_silence_ms
        self._speech_pad_ms = speech_pad_ms
        self._model: object | None = None
        self._load_model()

    # ── public ──────────────────────────────────────────────────────────

    def is_speech(self, frame: bytes) -> bool:
        """Return True when *frame* (1024 bytes int16 PCM) contains speech.

        The frame is converted to float32 and passed to silero.
        """
        if len(frame) != _FRAME_BYTES:
            raise ValueError(
                f"Frame must be {_FRAME_BYTES} bytes ({_SILERO_FRAME_SAMPLES} samples × 2); "
                f"got {len(frame)}"
            )
        prob = self._model_call(frame)
        return prob >= self._threshold

    def reset(self) -> None:
        """Reset silero's recurrent state (call between utterances)."""
        if self._model is not None:
            try:
                self._model.reset_states()  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning("vad_reset_failed", error=str(exc))

    # ── private ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            from silero_vad import load_silero_vad  # type: ignore[import]

            self._model = load_silero_vad()
            logger.info("vad_model_loaded", sample_rate=self._sample_rate)
        except Exception as exc:
            raise VADUnavailableError("silero-vad 모델 로드 실패") from exc

    def _model_call(self, frame: bytes) -> float:
        """Convert raw PCM bytes → float32 tensor and run inference.

        Kept as a separate method so tests can monkeypatch it cleanly.
        """
        import torch  # lazy; bundled with silero-vad

        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(samples)
        with torch.no_grad():
            prob_tensor = self._model(tensor, self._sample_rate)  # type: ignore[misc]
        return float(prob_tensor.squeeze())


# ---------------------------------------------------------------------------
# speech_segments
# ---------------------------------------------------------------------------


async def speech_segments(
    stream: object,  # AudioInputStream protocol
    vad: VAD,
    *,
    min_speech_ms: int = 200,
    max_silence_ms: int = 700,
    max_segment_ms: int = 30_000,
) -> AsyncIterator[bytes]:
    """Consume *stream* frames and yield one PCM blob per detected utterance.

    Algorithm:
    1. Maintain a ring buffer of the last ``speech_pad_ms`` frames.
    2. When a speech frame arrives, begin a segment, prepending the ring.
    3. Accumulate frames; end segment on ``max_silence_ms`` consecutive silence.
    4. Drop segments shorter than ``min_speech_ms``.
    5. Force-yield at ``max_segment_ms`` even if speech continues.

    *stream* must satisfy the AudioInputStream protocol defined in
    ``apps.channels.voice.audio``.
    """
    fmt = stream.fmt  # type: ignore[attr-defined]
    frame_ms: int = int(_SILERO_FRAME_SAMPLES / fmt.sample_rate * 1000)  # = 32 ms
    bytes_per_ms: float = fmt.sample_rate * fmt.sample_width_bytes / 1000.0

    pad_frames: int = max(1, vad._speech_pad_ms // frame_ms)
    ring: collections.deque[bytes] = collections.deque(maxlen=pad_frames)

    in_speech: bool = False
    segment: list[bytes] = []
    silence_ms: int = 0
    speech_ms: int = 0

    async for frame in stream.frames(frame_ms=frame_ms):  # type: ignore[attr-defined]
        is_speech_frame = vad.is_speech(frame)

        if not in_speech:
            # Pre-speech ring buffer
            ring.append(frame)
            if is_speech_frame:
                in_speech = True
                silence_ms = 0
                speech_ms = frame_ms
                segment = list(ring)  # include pre-roll padding
                logger.debug("vad_segment_start")
        else:
            segment.append(frame)

            if is_speech_frame:
                silence_ms = 0
                speech_ms += frame_ms
            else:
                silence_ms += frame_ms

            segment_ms = speech_ms + silence_ms
            force_end = segment_ms >= max_segment_ms
            natural_end = silence_ms >= max_silence_ms

            if force_end or natural_end:
                if speech_ms >= min_speech_ms:
                    yield b"".join(segment)
                    logger.debug(
                        "vad_segment_end",
                        speech_ms=speech_ms,
                        silence_ms=silence_ms,
                        forced=force_end,
                    )
                else:
                    logger.debug("vad_segment_dropped_too_short", speech_ms=speech_ms)

                # Reset state
                vad.reset()
                ring.clear()
                in_speech = False
                segment = []
                silence_ms = 0
                speech_ms = 0

                if force_end:
                    # Keep listening; don't return
                    continue

    # Stream exhausted — flush any open segment
    if in_speech and speech_ms >= min_speech_ms:
        yield b"".join(segment)
        logger.debug("vad_segment_flush_on_stream_end", speech_ms=speech_ms)
