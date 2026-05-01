"""Barge-in interruption detection for Phase 2 voice channel.

CLAUDE.md references:
  §6.1  Interruption: user speaks while assistant speaks → stop TTS + agent
        within 300ms, captured speech becomes next turn input.
  §10   Mic is ON only during TTS playback — no always-on recording.

Public API
----------
InterruptionDetector  – detects speech onset while assistant is speaking.
_FrameFanout          – fan-out a single mic stream to multiple consumers.
speech_segments_from_queue – variant of speech_segments that reads from an
                             asyncio.Queue instead of an AudioInputStream.
                             Choice (b) from spec: cleaner than a wrapper class
                             because it avoids faking the full AudioInputStream
                             Protocol just to change the read source.
"""

from __future__ import annotations

import asyncio
import collections
from collections.abc import AsyncIterator
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Silero v5 frame size at 16 kHz (matches vad.py constants).
_SILERO_FRAME_SAMPLES: int = 512
_SILERO_SAMPLE_RATE: int = 16_000
_FRAME_BYTES: int = _SILERO_FRAME_SAMPLES * 2  # int16 = 2 bytes/sample
_FRAME_MS: int = 32  # 512 samples / 16 000 Hz * 1000 = 32 ms


# ---------------------------------------------------------------------------
# _FrameFanout
# ---------------------------------------------------------------------------


class _FrameFanout:
    """One mic producer → multiple async consumers via asyncio.Queue per subscriber.

    Usage:
        fanout = _FrameFanout(mic)
        q1 = fanout.subscribe()
        q2 = fanout.subscribe()
        asyncio.create_task(fanout.run())   # start background pump
        # consumers read from q1 and q2 independently

    Each subscriber queue is bounded (maxsize=200).  On overflow the oldest
    frame is dropped and a warning is logged to avoid unbounded lag.
    """

    _QUEUE_MAXSIZE: int = 200

    def __init__(self, mic: Any, *, frame_ms: int = _FRAME_MS) -> None:
        self._mic = mic
        self._frame_ms = frame_ms
        self._queues: list[asyncio.Queue[bytes | None]] = []

    def subscribe(self) -> asyncio.Queue[bytes | None]:
        """Return a new subscriber queue.  Must be called before run()."""
        q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._queues.append(q)
        return q

    async def run(self) -> None:
        """Pump frames from mic into all subscriber queues.

        Runs until the mic stream ends or this task is cancelled.
        Sends None sentinel to each queue on exit so consumers can detect EOF.
        """
        try:
            async for frame in self._mic.frames(frame_ms=self._frame_ms):
                for q in self._queues:
                    if q.full():
                        try:
                            q.get_nowait()  # drop oldest
                        except asyncio.QueueEmpty:
                            pass
                        log.warning("fanout_queue_overflow_dropped_oldest")
                    await q.put(frame)
        except asyncio.CancelledError:
            pass
        finally:
            # Signal all consumers that the stream has ended
            for q in self._queues:
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    pass


# ---------------------------------------------------------------------------
# speech_segments_from_queue
# ---------------------------------------------------------------------------


async def speech_segments_from_queue(
    queue: asyncio.Queue[bytes | None],
    vad: Any,
    *,
    min_speech_ms: int = 200,
    max_silence_ms: int = 700,
    max_segment_ms: int = 30_000,
) -> AsyncIterator[bytes]:
    """Variant of vad.speech_segments that reads from an asyncio.Queue.

    Choice (b): sibling helper that mirrors vad.speech_segments exactly but
    takes a Queue[bytes | None] instead of an AudioInputStream.  This avoids
    wrapping the queue in a fake AudioInputStream Protocol object and keeps
    the tests clean.

    Yields one PCM blob per detected utterance; stops when None sentinel
    is received from the queue.
    """
    pad_ms: int = getattr(vad, "_speech_pad_ms", 100)
    frame_ms: int = _FRAME_MS
    pad_frames: int = max(1, pad_ms // frame_ms)
    ring: collections.deque[bytes] = collections.deque(maxlen=pad_frames)

    in_speech: bool = False
    segment: list[bytes] = []
    silence_ms: int = 0
    speech_ms: int = 0

    while True:
        frame = await queue.get()
        if frame is None:
            break  # EOF sentinel

        is_speech_frame: bool = vad.is_speech(frame)

        if not in_speech:
            ring.append(frame)
            if is_speech_frame:
                in_speech = True
                silence_ms = 0
                speech_ms = frame_ms
                segment = list(ring)
                log.debug("vad_queue_segment_start")
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
                    log.debug(
                        "vad_queue_segment_end",
                        speech_ms=speech_ms,
                        silence_ms=silence_ms,
                        forced=force_end,
                    )
                else:
                    log.debug("vad_queue_segment_dropped_too_short", speech_ms=speech_ms)

                vad.reset()
                ring.clear()
                in_speech = False
                segment = []
                silence_ms = 0
                speech_ms = 0

                if force_end:
                    continue

    # Queue exhausted — flush any open segment
    if in_speech and speech_ms >= min_speech_ms:
        yield b"".join(segment)
        log.debug("vad_queue_segment_flush_on_eof", speech_ms=speech_ms)


# ---------------------------------------------------------------------------
# InterruptionDetector
# ---------------------------------------------------------------------------


class InterruptionDetector:
    """Detects user speech onset while the assistant is speaking.

    Reads mic frames from a subscriber queue (populated by _FrameFanout),
    runs VAD on each frame, and signals as soon as confirmed speech is
    detected.  The pre-speech ring buffer ensures the captured bytes include
    audio that started *before* VAD confirmation (no front-clipping).

    Parameters
    ----------
    queue:
        asyncio.Queue[bytes | None] fed by _FrameFanout.subscribe().
    vad:
        VAD instance from vad.py — must expose is_speech(frame) and reset().
    onset_ms:
        Confirm speech after this much continuous voice (default 200 ms).
    frame_ms:
        Silero v5 frame size at 16 kHz = 32 ms.
    prebuffer_ms:
        Keep this many ms of frames before VAD confirmation so the captured
        speech includes audio from 100-300 ms before the detection point.
    """

    def __init__(
        self,
        queue: asyncio.Queue[bytes | None],
        vad: Any,
        *,
        onset_ms: int = 200,
        frame_ms: int = _FRAME_MS,
        prebuffer_ms: int = 500,
    ) -> None:
        self._queue = queue
        self._vad = vad
        self._onset_ms = onset_ms
        self._frame_ms = frame_ms
        self._prebuffer_ms = prebuffer_ms

    async def wait_for_interrupt(self) -> bytes:
        """Block until speech onset confirmed; return captured PCM bytes.

        Returns prebuffer + onset frames so the next STT turn loses no audio.
        Raises asyncio.CancelledError if this coroutine is cancelled.
        """
        prebuffer_frames = max(1, self._prebuffer_ms // self._frame_ms)
        onset_frames_needed = max(1, self._onset_ms // self._frame_ms)

        # ring holds the last `prebuffer_frames` frames for pre-roll
        ring: collections.deque[bytes] = collections.deque(maxlen=prebuffer_frames)
        onset_buf: list[bytes] = []
        consecutive_speech: int = 0

        while True:
            frame = await self._queue.get()
            if frame is None:
                # Stream ended without detecting speech — return empty
                log.debug("interrupt_detector_stream_ended_no_speech")
                return b""

            is_speech = self._vad.is_speech(frame)

            if consecutive_speech == 0:
                # Not yet in speech onset: keep in ring buffer
                ring.append(frame)

            if is_speech:
                consecutive_speech += 1
                onset_buf.append(frame)
                if consecutive_speech >= onset_frames_needed:
                    # Confirmed speech onset
                    captured = b"".join(list(ring)[: prebuffer_frames - len(onset_buf)] if len(ring) > len(onset_buf) else ring) + b"".join(onset_buf)
                    log.info(
                        "interrupt_detected",
                        onset_ms=consecutive_speech * self._frame_ms,
                        captured_bytes=len(captured),
                    )
                    return captured
            else:
                # Speech broken — reset
                if consecutive_speech > 0:
                    # Put onset frames back into ring as pre-roll
                    for f in onset_buf:
                        ring.append(f)
                consecutive_speech = 0
                onset_buf = []
