"""Unit tests for apps.channels.voice.vad.

silero-vad is monkeypatched via VAD._model_call so no real neural network
or torch inference runs during the test suite.
"""

from __future__ import annotations

import asyncio
import struct
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from apps.channels.voice.audio import AudioFormat, FakeMicrophone
from apps.channels.voice.vad import (
    VAD,
    _FRAME_BYTES,
    _SILERO_FRAME_SAMPLES,
    _SILERO_SAMPLE_RATE,
    speech_segments,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FMT = AudioFormat(sample_rate=_SILERO_SAMPLE_RATE, channels=1, sample_width_bytes=2)
_FRAME_MS = int(_SILERO_FRAME_SAMPLES / _SILERO_SAMPLE_RATE * 1000)  # 32 ms


def _make_vad(*, speech_prob: float = 0.9, threshold: float = 0.5) -> VAD:
    """Return a VAD whose _model_call always returns *speech_prob*."""
    with patch("apps.channels.voice.vad.VAD._load_model"):
        vad = VAD.__new__(VAD)
        vad._sample_rate = _SILERO_SAMPLE_RATE
        vad._threshold = threshold
        vad._min_silence_ms = 700
        vad._speech_pad_ms = 100
        vad._model = MagicMock()
        vad._model.reset_states = MagicMock()
    return vad


def _speech_frame() -> bytes:
    """Return a frame that is marked as speech in tests."""
    return b"\x01" * _FRAME_BYTES  # non-zero marker


def _silence_frame() -> bytes:
    """Return a silent (zeroed) PCM frame."""
    return b"\x00" * _FRAME_BYTES


async def _collect(gen: AsyncIterator[bytes]) -> list[bytes]:
    result: list[bytes] = []
    async for chunk in gen:
        result.append(chunk)
    return result


# ---------------------------------------------------------------------------
# is_speech
# ---------------------------------------------------------------------------


class TestIsSpeech:
    def test_returns_true_above_threshold(self) -> None:
        vad = _make_vad(threshold=0.5)
        # Patch _model_call to return 0.9
        vad._model_call = lambda frame: 0.9  # type: ignore[method-assign]
        assert vad.is_speech(_speech_frame()) is True

    def test_returns_false_below_threshold(self) -> None:
        vad = _make_vad(threshold=0.5)
        vad._model_call = lambda frame: 0.1  # type: ignore[method-assign]
        assert vad.is_speech(_silence_frame()) is False

    def test_returns_true_at_exact_threshold(self) -> None:
        vad = _make_vad(threshold=0.5)
        vad._model_call = lambda frame: 0.5  # type: ignore[method-assign]
        assert vad.is_speech(_speech_frame()) is True

    def test_raises_on_wrong_frame_size(self) -> None:
        vad = _make_vad()
        vad._model_call = lambda frame: 0.9  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="1024 bytes"):
            vad.is_speech(b"\x00" * 10)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_calls_model_reset_states(self) -> None:
        vad = _make_vad()
        vad._model_call = lambda frame: 0.0  # type: ignore[method-assign]
        vad.reset()
        vad._model.reset_states.assert_called_once()

    def test_reset_clears_vad_state_so_next_segment_unaffected(self) -> None:
        """After reset, is_speech still works correctly."""
        vad = _make_vad(threshold=0.5)
        vad._model_call = lambda frame: 0.9  # type: ignore[method-assign]
        assert vad.is_speech(_speech_frame()) is True
        vad.reset()
        assert vad.is_speech(_speech_frame()) is True


# ---------------------------------------------------------------------------
# speech_segments — helper: build FakeMicrophone with annotated frames
# ---------------------------------------------------------------------------


def _make_stream(speech_pattern: list[bool]) -> FakeMicrophone:
    """Build a FakeMicrophone whose frames follow *speech_pattern*.

    True  → speech frame (0x01 marker byte at start)
    False → silence frame (all zeros)
    """
    frames = [_speech_frame() if s else _silence_frame() for s in speech_pattern]
    return FakeMicrophone(frames_data=frames, fmt=_FMT)


def _vad_from_pattern(speech_pattern: list[bool], threshold: float = 0.5) -> VAD:
    """Return a VAD where _model_call reads the first byte as speech marker."""
    vad = _make_vad(threshold=threshold)

    def _fake_model_call(frame: bytes) -> float:
        return 0.9 if frame[0] != 0 else 0.0

    vad._model_call = _fake_model_call  # type: ignore[method-assign]
    return vad


# ---------------------------------------------------------------------------
# speech_segments — correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speech_segments_emits_one_segment() -> None:
    """5 speech frames bracketed by silence emit exactly one segment."""
    # Layout: 2 silence | 5 speech | 25 silence (>700 ms max_silence_ms)
    #         = 2 × 32 ms silence + 5 × 32 ms speech + 25 × 32 ms silence
    # max_silence_ms = 700 → needs 700/32 ≈ 22 silence frames to end segment
    pattern = [False] * 2 + [True] * 5 + [False] * 25
    stream = _make_stream(pattern)
    vad = _vad_from_pattern(pattern)

    segments = await _collect(
        speech_segments(stream, vad, min_speech_ms=50, max_silence_ms=700)
    )

    assert len(segments) == 1
    # Emitted blob must contain the 5 speech frames (plus pre-roll padding)
    assert len(segments[0]) >= 5 * _FRAME_BYTES


@pytest.mark.asyncio
async def test_short_segments_are_dropped() -> None:
    """Segment shorter than min_speech_ms is silently discarded."""
    # 1 speech frame = 32 ms < min_speech_ms = 200 ms → dropped
    pattern = [False] * 2 + [True] * 1 + [False] * 25
    stream = _make_stream(pattern)
    vad = _vad_from_pattern(pattern)

    segments = await _collect(
        speech_segments(stream, vad, min_speech_ms=200, max_silence_ms=700)
    )

    assert segments == []


@pytest.mark.asyncio
async def test_segments_longer_than_max_are_force_yielded() -> None:
    """A segment exceeding max_segment_ms is yielded even if speech continues."""
    # max_segment_ms = 320 ms = 10 frames × 32 ms
    # We feed 20 consecutive speech frames; expect at least 1 forced yield.
    pattern = [True] * 20
    stream = _make_stream(pattern)
    vad = _vad_from_pattern(pattern)

    segments = await _collect(
        speech_segments(
            stream,
            vad,
            min_speech_ms=50,
            max_silence_ms=700,
            max_segment_ms=320,  # 10 frames
        )
    )

    assert len(segments) >= 1


@pytest.mark.asyncio
async def test_pre_roll_padding_included() -> None:
    """The pre-speech ring buffer (speech_pad_ms) is prepended to the segment."""
    # speech_pad_ms = 100 ms → 3 pad frames (3 × 32 ms = 96 ms ≈ 100 ms)
    # Feed: 3 silence + 5 speech + 25 silence
    # The 3 silence frames go into the ring; when speech starts the ring is
    # prepended, so the segment starts with ~3 silence frames.
    pattern = [False] * 3 + [True] * 5 + [False] * 25
    stream = _make_stream(pattern)

    vad = _make_vad(threshold=0.5)
    vad._speech_pad_ms = 100  # 100 ms pad → ≤3 frames of 32 ms

    def _fake_model_call(frame: bytes) -> float:
        return 0.9 if frame[0] != 0 else 0.0

    vad._model_call = _fake_model_call  # type: ignore[method-assign]

    segments = await _collect(
        speech_segments(stream, vad, min_speech_ms=50, max_silence_ms=700)
    )

    assert len(segments) == 1
    # Segment should be larger than bare 5 speech frames due to pre-roll
    assert len(segments[0]) > 5 * _FRAME_BYTES


@pytest.mark.asyncio
async def test_multiple_segments() -> None:
    """Two bursts of speech yield two separate segments."""
    # 5 speech | 25 silence | 5 speech | 25 silence
    pattern = [True] * 5 + [False] * 25 + [True] * 5 + [False] * 25
    stream = _make_stream(pattern)
    vad = _vad_from_pattern(pattern)

    segments = await _collect(
        speech_segments(stream, vad, min_speech_ms=50, max_silence_ms=700)
    )

    assert len(segments) == 2


@pytest.mark.asyncio
async def test_empty_stream_yields_nothing() -> None:
    """An empty stream produces no segments."""
    stream = FakeMicrophone(frames_data=[], fmt=_FMT)
    vad = _vad_from_pattern([])

    segments = await _collect(speech_segments(stream, vad, min_speech_ms=50))
    assert segments == []
