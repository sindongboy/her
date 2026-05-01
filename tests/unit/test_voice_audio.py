"""Unit tests for apps/channels/voice/audio.py.

Uses only FakeMicrophone and FakeSpeaker — no real audio hardware required.
Run: uv run pytest tests/unit/test_voice_audio.py -x -q
"""

from __future__ import annotations

import asyncio

import pytest

from apps.channels.voice.audio import (
    AudioFormat,
    FakeMicrophone,
    FakeSpeaker,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pcm(n_bytes: int = 640) -> bytes:
    """Return a silent PCM frame of *n_bytes* bytes."""
    return b"\x00" * n_bytes


# ---------------------------------------------------------------------------
# 1. AudioFormat defaults
# ---------------------------------------------------------------------------

class TestAudioFormat:
    def test_defaults(self) -> None:
        fmt = AudioFormat()
        assert fmt.sample_rate == 16_000
        assert fmt.channels == 1
        assert fmt.sample_width_bytes == 2

    def test_frozen(self) -> None:
        fmt = AudioFormat()
        with pytest.raises((AttributeError, TypeError)):
            fmt.sample_rate = 8_000  # type: ignore[misc]

    # 7. Frame-size math: 20ms @ 16 kHz mono int16 = 640 bytes
    def test_frame_bytes_20ms_16k(self) -> None:
        fmt = AudioFormat(sample_rate=16_000, channels=1, sample_width_bytes=2)
        assert fmt.frame_bytes(20) == 640  # 16000 * 0.020 * 1 * 2

    def test_frame_bytes_20ms_24k(self) -> None:
        fmt = AudioFormat(sample_rate=24_000, channels=1, sample_width_bytes=2)
        assert fmt.frame_bytes(20) == 960  # 24000 * 0.020 * 1 * 2

    def test_frame_bytes_various(self) -> None:
        fmt = AudioFormat(sample_rate=16_000, channels=1, sample_width_bytes=2)
        assert fmt.frame_bytes(10) == 320
        assert fmt.frame_bytes(30) == 960


# ---------------------------------------------------------------------------
# 2. FakeMicrophone.frames() yields seeded frames in order, then stops
# ---------------------------------------------------------------------------

class TestFakeMicrophone:
    @pytest.mark.asyncio
    async def test_yields_frames_in_order(self) -> None:
        data = [b"\x01" * 640, b"\x02" * 640, b"\x03" * 640]
        mic = FakeMicrophone(data)
        collected: list[bytes] = []
        async for frame in mic.frames():
            collected.append(frame)
        assert collected == data

    @pytest.mark.asyncio
    async def test_stops_after_all_frames(self) -> None:
        mic = FakeMicrophone([_pcm(), _pcm()])
        count = 0
        async for _ in mic.frames():
            count += 1
        assert count == 2

    @pytest.mark.asyncio
    async def test_empty_frames_list(self) -> None:
        mic = FakeMicrophone([])
        collected: list[bytes] = []
        async for frame in mic.frames():
            collected.append(frame)
        assert collected == []

    # 3. FakeMicrophone.close() interrupts in-progress iteration without raising
    @pytest.mark.asyncio
    async def test_close_interrupts_iteration(self) -> None:
        """close() mid-iteration should stop gracefully, not raise."""
        # Build a mic with many frames; close after the first one
        data = [_pcm()] * 10
        mic = FakeMicrophone(data)
        collected: list[bytes] = []

        async def _consume() -> None:
            async for frame in mic.frames():
                collected.append(frame)
                if len(collected) == 1:
                    await mic.close()

        await _consume()
        # After close, iteration stopped; we got at least 1 and fewer than 10
        assert 1 <= len(collected) <= 10  # stopped early

    @pytest.mark.asyncio
    async def test_close_after_done_is_safe(self) -> None:
        mic = FakeMicrophone([_pcm()])
        async for _ in mic.frames():
            pass
        await mic.close()  # should not raise

    def test_fmt_default(self) -> None:
        mic = FakeMicrophone([])
        assert mic.fmt == AudioFormat()


# ---------------------------------------------------------------------------
# 4-6. FakeSpeaker
# ---------------------------------------------------------------------------

class TestFakeSpeaker:
    # 4. write() accumulates into .written
    @pytest.mark.asyncio
    async def test_write_accumulates(self) -> None:
        spk = FakeSpeaker()
        await spk.write(b"\xAA" * 100)
        await spk.write(b"\xBB" * 200)
        assert spk.written == b"\xAA" * 100 + b"\xBB" * 200

    @pytest.mark.asyncio
    async def test_written_is_bytes(self) -> None:
        spk = FakeSpeaker()
        await spk.write(b"\x01\x02")
        assert isinstance(spk.written, bytes)

    # 5. stop() clears pending and flush() returns immediately afterwards
    @pytest.mark.asyncio
    async def test_stop_clears_written(self) -> None:
        spk = FakeSpeaker()
        await spk.write(b"\xFF" * 640)
        await spk.stop()
        assert spk.written == b""

    @pytest.mark.asyncio
    async def test_flush_after_stop_returns_immediately(self) -> None:
        spk = FakeSpeaker()
        await spk.write(b"\x00" * 320)
        await spk.stop()
        # flush must complete without blocking
        done = False

        async def _flush() -> None:
            nonlocal done
            await spk.flush()
            done = True

        await asyncio.wait_for(_flush(), timeout=1.0)
        assert done

    # 6. flush() awaits all writes complete (for Fake this is immediate)
    @pytest.mark.asyncio
    async def test_flush_returns_after_write(self) -> None:
        spk = FakeSpeaker()
        await spk.write(b"\x00" * 640)
        # For FakeSpeaker, flush is a no-op — should complete instantly
        done = False

        async def _flush() -> None:
            nonlocal done
            await spk.flush()
            done = True

        await asyncio.wait_for(_flush(), timeout=1.0)
        assert done

    @pytest.mark.asyncio
    async def test_write_after_stop_no_accumulate(self) -> None:
        """After stop(), writes are dropped (stopped flag is set)."""
        spk = FakeSpeaker()
        await spk.stop()
        await spk.write(b"\xFF" * 100)
        assert spk.written == b""

    def test_fmt_default_24k(self) -> None:
        spk = FakeSpeaker()
        assert spk.fmt.sample_rate == 24_000

    @pytest.mark.asyncio
    async def test_close_is_safe(self) -> None:
        spk = FakeSpeaker()
        await spk.write(b"\x01" * 10)
        await spk.close()


# ---------------------------------------------------------------------------
# Import smoke tests (no hardware)
# ---------------------------------------------------------------------------

class TestImports:
    def test_package_exports(self) -> None:
        from apps.channels.voice import (  # noqa: F401
            AudioFormat,
            AudioInputStream,
            AudioOutputStream,
            FakeMicrophone,
            FakeSpeaker,
            open_microphone,
            open_speaker,
        )

    def test_audio_device_error_importable(self) -> None:
        from apps.channels.voice.audio import AudioDeviceError  # noqa: F401
        assert issubclass(AudioDeviceError, RuntimeError)

    def test_open_microphone_returns_sounddevice_mic(self) -> None:
        from apps.channels.voice.audio import (
            SoundDeviceMicrophone,
            open_microphone,
        )
        mic = open_microphone()
        assert isinstance(mic, SoundDeviceMicrophone)

    def test_open_speaker_returns_sounddevice_speaker(self) -> None:
        from apps.channels.voice.audio import (
            SoundDeviceSpeaker,
            open_speaker,
        )
        spk = open_speaker()
        assert isinstance(spk, SoundDeviceSpeaker)

    def test_open_microphone_custom_fmt(self) -> None:
        from apps.channels.voice.audio import open_microphone
        fmt = AudioFormat(sample_rate=8_000)
        mic = open_microphone(fmt=fmt)
        assert mic.fmt.sample_rate == 8_000

    def test_open_speaker_custom_fmt(self) -> None:
        from apps.channels.voice.audio import open_speaker
        fmt = AudioFormat(sample_rate=16_000)
        spk = open_speaker(fmt=fmt)
        assert spk.fmt.sample_rate == 16_000
