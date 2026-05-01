# Runtime deps: sounddevice, numpy. System: portaudio (`brew install portaudio`).
# voice-ch-eng: add `sounddevice` and `numpy` to pyproject.toml dependencies.
"""Mic/speaker abstractions backed by sounddevice.

Public API
----------
AudioFormat         – PCM format descriptor (sample_rate, channels, sample_width_bytes).
AudioInputStream    – Protocol: async iterator over PCM int16 mic frames.
AudioOutputStream   – Protocol: async sink for PCM int16 speaker frames.
SoundDeviceMicrophone / SoundDeviceSpeaker – concrete sounddevice implementations.
FakeMicrophone / FakeSpeaker              – test doubles (no hardware required).
open_microphone / open_speaker            – factory functions.

Per CLAUDE.md §6.1 (Voice Channel, Path B pipeline).
"""

from __future__ import annotations

import asyncio
import collections
import threading
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# AudioFormat
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class AudioFormat:
    """Describes a raw PCM stream."""

    sample_rate: int = 16_000        # Hz — whisper-native default
    channels: int = 1
    sample_width_bytes: int = 2      # int16 PCM

    def frame_bytes(self, frame_ms: int) -> int:
        """Return the byte count for one frame of *frame_ms* milliseconds."""
        samples = int(self.sample_rate * frame_ms / 1000)
        return samples * self.channels * self.sample_width_bytes


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class AudioInputStream(Protocol):
    """Async iterator over PCM int16 frames captured from a source (e.g. mic)."""

    fmt: AudioFormat

    def frames(self, *, frame_ms: int = 20) -> AsyncIterator[bytes]:
        """Yield raw PCM frames of the requested duration."""
        ...

    def drain(self) -> int:
        """Discard any frames currently buffered. Returns the number dropped.

        Used after the assistant's TTS finishes playing to avoid feeding the
        speaker's own output back through STT (echo / self-loopback).
        """
        ...

    async def close(self) -> None:
        """Stop capture and release resources."""
        ...


@runtime_checkable
class AudioOutputStream(Protocol):
    """Async sink for PCM int16 frames (e.g. speaker)."""

    fmt: AudioFormat

    async def write(self, pcm_bytes: bytes) -> None:
        """Append audio to the playback buffer. May suspend if buffer is full."""
        ...

    async def stop(self) -> None:
        """Abort playback immediately, drop pending buffer (Phase 2 interruption)."""
        ...

    async def flush(self) -> None:
        """Wait until all buffered audio has been played."""
        ...

    async def close(self) -> None:
        """Stop and release resources."""
        ...


# ---------------------------------------------------------------------------
# Project-specific exception
# ---------------------------------------------------------------------------

class AudioDeviceError(RuntimeError):
    """Raised when a sounddevice / PortAudio device cannot be opened.

    The message is Korean so it can be surfaced to the user directly.
    """


# ---------------------------------------------------------------------------
# SoundDeviceMicrophone
# ---------------------------------------------------------------------------

class SoundDeviceMicrophone:
    """Capture PCM int16 audio from a sounddevice RawInputStream.

    sounddevice is imported lazily so tests on machines without portaudio
    can import this module and use the Fake* test doubles.
    """

    def __init__(
        self,
        *,
        fmt: AudioFormat = AudioFormat(),
        device: int | str | None = None,
    ) -> None:
        self.fmt = fmt
        self._device = device
        self._stream: object | None = None
        # ~32 seconds of headroom at 20ms frames — fits a typical agent +
        # TTS round-trip in linear mode without dropping frames mid-turn.
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=1600)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    # ── public ──────────────────────────────────────────────────────────

    async def frames(self, *, frame_ms: int = 20) -> AsyncIterator[bytes]:
        """Yield raw PCM int16 frames of *frame_ms* milliseconds each."""
        await self._ensure_open(frame_ms)
        try:
            while not self._closed:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            pass  # keep stream open; caller calls close() when done

    def drain(self) -> int:
        """Discard all currently buffered frames. Returns count dropped."""
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        if dropped:
            logger.debug("mic_drained", frames=dropped)
        return dropped

    async def close(self) -> None:
        """Stop capture, drain the queue, and release the stream."""
        self._closed = True
        await self._queue.put(None)       # unblock any waiting consumer
        if self._stream is not None:
            try:
                self._stream.stop()       # type: ignore[union-attr]
                self._stream.close()      # type: ignore[union-attr]
            except Exception:
                pass
            self._stream = None

    # ── private ─────────────────────────────────────────────────────────

    async def _ensure_open(self, frame_ms: int) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd  # lazy import
        except ImportError as exc:
            raise AudioDeviceError(
                "마이크/스피커를 찾을 수 없어요. "
                "macOS 시스템 설정에서 권한 확인하세요. "
                f"(sounddevice not installed: {exc})"
            ) from exc

        self._loop = asyncio.get_running_loop()
        blocksize = int(self.fmt.sample_rate * frame_ms / 1000)

        def _callback(indata: bytes, frames: int, time: object, status: object) -> None:
            if status:
                logger.warning("mic_callback_status", status=str(status))
            data = bytes(indata)
            loop = self._loop
            q = self._queue
            if loop is None:
                return
            if q.full():
                # Drop oldest to keep real-time. In linear mode this is the
                # expected steady state while STT/agent/TTS are running — the
                # mic keeps capturing but the consumer is busy. Logged at
                # DEBUG so it doesn't spam the operator console.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                logger.debug("mic_queue_full_dropped_oldest")
            loop.call_soon_threadsafe(q.put_nowait, data)

        try:
            stream = sd.RawInputStream(
                samplerate=self.fmt.sample_rate,
                blocksize=blocksize,
                dtype="int16",
                channels=self.fmt.channels,
                device=self._device,
                callback=_callback,
            )
            stream.start()
            self._stream = stream
        except sd.PortAudioError as exc:
            raise AudioDeviceError(
                "마이크/스피커를 찾을 수 없어요. "
                "macOS 시스템 설정에서 권한 확인하세요."
            ) from exc

        # Log device info
        try:
            dev_info = sd.query_devices(self._device, "input")
            logger.info(
                "mic_opened",
                device_index=dev_info.get("index"),
                device_name=dev_info.get("name"),
                sample_rate=self.fmt.sample_rate,
                frame_ms=frame_ms,
            )
        except Exception:
            logger.info("mic_opened", device=self._device, sample_rate=self.fmt.sample_rate)


# ---------------------------------------------------------------------------
# SoundDeviceSpeaker
# ---------------------------------------------------------------------------

class SoundDeviceSpeaker:
    """Play PCM int16 audio through a sounddevice RawOutputStream.

    sounddevice is imported lazily — see SoundDeviceMicrophone for rationale.
    """

    # 24 kHz default: matches Gemini TTS PCM output (CLAUDE.md §3.2)
    def __init__(
        self,
        *,
        fmt: AudioFormat = AudioFormat(sample_rate=24_000),
        device: int | str | None = None,
    ) -> None:
        self.fmt = fmt
        self._device = device
        self._stream: object | None = None
        self._pending: collections.deque[bytes] = collections.deque()
        self._lock = threading.Lock()
        self._stop_flag = False
        self._done_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    # ── public ──────────────────────────────────────────────────────────

    async def write(self, pcm_bytes: bytes) -> None:
        """Append *pcm_bytes* to the playback buffer."""
        await self._ensure_open()
        with self._lock:
            self._stop_flag = False
            self._pending.append(pcm_bytes)
            if self._done_event is not None:
                self._done_event.clear()

    async def stop(self) -> None:
        """Abort playback, drop all pending audio."""
        with self._lock:
            self._stop_flag = True
            self._pending.clear()
        # Signal flush waiters immediately
        if self._done_event is not None:
            self._done_event.set()

    async def flush(self) -> None:
        """Await until the pending buffer has been fully played."""
        if self._done_event is None:
            return
        with self._lock:
            if not self._pending and not self._stop_flag:
                return
            if self._stop_flag:
                return
        await self._done_event.wait()

    async def close(self) -> None:
        """Stop playback and release resources."""
        self._closed = True
        await self.stop()
        if self._stream is not None:
            try:
                self._stream.stop()   # type: ignore[union-attr]
                self._stream.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._stream = None

    # ── private ─────────────────────────────────────────────────────────

    async def _ensure_open(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd  # lazy import
            import numpy as np        # noqa: F401  (validates numpy is available)
        except ImportError as exc:
            raise AudioDeviceError(
                "마이크/스피커를 찾을 수 없어요. "
                "macOS 시스템 설정에서 권한 확인하세요. "
                f"(sounddevice/numpy not installed: {exc})"
            ) from exc

        self._loop = asyncio.get_running_loop()
        self._done_event = asyncio.Event()
        self._done_event.set()  # initially "done" (nothing pending)

        def _callback(outdata: bytearray, frames: int, time: object, status: object) -> None:
            if status:
                logger.warning("speaker_callback_status", status=str(status))
            needed = len(outdata)
            written = 0
            with self._lock:
                if self._stop_flag:
                    outdata[:] = b"\x00" * needed
                    return
                while written < needed and self._pending:
                    chunk = self._pending[0]
                    available = len(chunk)
                    take = min(available, needed - written)
                    outdata[written : written + take] = chunk[:take]
                    written += take
                    if take == available:
                        self._pending.popleft()
                    else:
                        self._pending[0] = chunk[take:]
                # Silence-pad if buffer ran dry
                if written < needed:
                    outdata[written:] = b"\x00" * (needed - written)
                # Signal flush when buffer empty
                if not self._pending:
                    loop = self._loop
                    ev = self._done_event
                    if loop is not None and ev is not None:
                        loop.call_soon_threadsafe(ev.set)

        try:
            stream = sd.RawOutputStream(
                samplerate=self.fmt.sample_rate,
                dtype="int16",
                channels=self.fmt.channels,
                device=self._device,
                callback=_callback,
            )
            stream.start()
            self._stream = stream
        except sd.PortAudioError as exc:
            raise AudioDeviceError(
                "마이크/스피커를 찾을 수 없어요. "
                "macOS 시스템 설정에서 권한 확인하세요."
            ) from exc

        try:
            dev_info = sd.query_devices(self._device, "output")
            logger.info(
                "speaker_opened",
                device_index=dev_info.get("index"),
                device_name=dev_info.get("name"),
                sample_rate=self.fmt.sample_rate,
            )
        except Exception:
            logger.info("speaker_opened", device=self._device, sample_rate=self.fmt.sample_rate)


# ---------------------------------------------------------------------------
# FakeMicrophone — test double
# ---------------------------------------------------------------------------

class FakeMicrophone:
    """Replays a fixed list of PCM frames in sequence, then stops.

    No hardware required. For use in unit tests only.
    """

    def __init__(
        self,
        frames_data: list[bytes],
        fmt: AudioFormat = AudioFormat(),
    ) -> None:
        self.fmt = fmt
        self._frames = list(frames_data)
        self._closed = False

    async def frames(self, *, frame_ms: int = 20) -> AsyncIterator[bytes]:  # type: ignore[override]
        for frame in self._frames:
            if self._closed:
                return
            yield frame

    def drain(self) -> int:
        """No-op for test double — pre-seeded frames are consumed via iteration."""
        return 0

    async def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# FakeSpeaker — test double
# ---------------------------------------------------------------------------

class FakeSpeaker:
    """Records every write() call into self.written for assertion in tests.

    No hardware required. For use in unit tests only.
    """

    def __init__(self, fmt: AudioFormat = AudioFormat(sample_rate=24_000)) -> None:
        self.fmt = fmt
        self._written: bytearray = bytearray()
        self._stopped = False
        self._closed = False

    @property
    def written(self) -> bytes:
        """Return all bytes written so far as an immutable bytes object."""
        return bytes(self._written)

    async def write(self, pcm_bytes: bytes) -> None:
        if not self._stopped:
            self._written.extend(pcm_bytes)

    async def stop(self) -> None:
        """Clear pending state; flush() will return immediately afterwards."""
        self._stopped = True
        self._written = bytearray()

    async def flush(self) -> None:
        """No-op for fake — everything is synchronously 'played'."""
        return

    async def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def open_microphone(
    *,
    fmt: AudioFormat = AudioFormat(),
    device: int | str | None = None,
) -> AudioInputStream:
    """Return a SoundDeviceMicrophone configured with *fmt* and *device*."""
    return SoundDeviceMicrophone(fmt=fmt, device=device)  # type: ignore[return-value]


def open_speaker(
    *,
    fmt: AudioFormat = AudioFormat(sample_rate=24_000),
    device: int | str | None = None,
) -> AudioOutputStream:
    """Return a SoundDeviceSpeaker configured with *fmt* and *device*."""
    return SoundDeviceSpeaker(fmt=fmt, device=device)  # type: ignore[return-value]
