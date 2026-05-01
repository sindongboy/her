"""Unit tests for apps.channels.voice.stt.

faster-whisper is monkeypatched so no model download or inference happens.
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from apps.channels.voice.audio import AudioFormat
from apps.channels.voice.stt import STT, _pcm_to_float32


# ---------------------------------------------------------------------------
# Fake WhisperModel
# ---------------------------------------------------------------------------


@dataclass
class _FakeSegment:
    text: str


class _FakeWhisperModel:
    """Returns a single segment with text '안녕하세요' for any input."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass  # accept any constructor args

    def transcribe(
        self,
        audio: Any,
        **kwargs: Any,
    ) -> tuple[Iterable[_FakeSegment], object]:
        return [_FakeSegment(text="안녕하세요")], MagicMock()


class _FakeSilentWhisperModel:
    """Returns no segments (simulates silent audio)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Iterable[_FakeSegment], object]:
        return [], MagicMock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pcm(n_samples: int = 1600) -> bytes:
    """Return *n_samples* silent int16 samples as raw bytes."""
    return (b"\x00\x00") * n_samples


def _make_stt_patched(**kwargs: Any) -> STT:
    """Return an STT instance with faster_whisper.WhisperModel patched out."""
    return STT(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_returns_transcription_text(self) -> None:
        """transcribe() returns '안녕하세요' when the fake model produces that."""
        with patch("apps.channels.voice.stt.STT._load_model", return_value=_FakeWhisperModel()):
            stt = STT()
            result = await stt.transcribe(_make_pcm())
        assert result == "안녕하세요"

    @pytest.mark.asyncio
    async def test_empty_bytes_returns_empty_string(self) -> None:
        """transcribe() on empty bytes returns '' without raising."""
        with patch("apps.channels.voice.stt.STT._load_model", return_value=_FakeWhisperModel()):
            stt = STT()
            result = await stt.transcribe(b"")
        # Empty bytes short-circuits before model load
        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_bytes_does_not_load_model(self) -> None:
        """Model is NOT loaded when input is empty (short-circuit)."""
        load_called = False

        def _fake_load() -> _FakeWhisperModel:
            nonlocal load_called
            load_called = True
            return _FakeWhisperModel()

        with patch("apps.channels.voice.stt.STT._load_model", side_effect=_fake_load):
            stt = STT()
            await stt.transcribe(b"")

        assert not load_called

    @pytest.mark.asyncio
    async def test_initial_prompt_plumbed_through(self) -> None:
        """initial_prompt kwarg is forwarded to the underlying model.transcribe."""
        captured: dict[str, Any] = {}

        class _CapturingModel:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list[_FakeSegment], object]:
                captured.update(kwargs)
                return [_FakeSegment(text="테스트")], MagicMock()

        with patch("apps.channels.voice.stt.STT._load_model", return_value=_CapturingModel()):
            stt = STT()
            await stt.transcribe(_make_pcm(), initial_prompt="김민준 박서연")

        assert captured.get("initial_prompt") == "김민준 박서연"


class TestLazyLoad:
    @pytest.mark.asyncio
    async def test_model_not_loaded_at_init(self) -> None:
        """_model is None right after __init__ — no load until first call."""
        stt = STT()
        assert stt._model is None

    @pytest.mark.asyncio
    async def test_model_loaded_on_first_transcribe(self) -> None:
        """_model is set after the first transcribe() call."""
        with patch("apps.channels.voice.stt.STT._load_model", return_value=_FakeWhisperModel()):
            stt = STT()
            assert stt._model is None
            await stt.transcribe(_make_pcm())
            assert stt._model is not None

    @pytest.mark.asyncio
    async def test_warmup_triggers_load(self) -> None:
        """warmup() causes the model to be loaded."""
        with patch("apps.channels.voice.stt.STT._load_model", return_value=_FakeSilentWhisperModel()):
            stt = STT()
            assert stt._model is None
            await stt.warmup()
            assert stt._model is not None

    @pytest.mark.asyncio
    async def test_model_loaded_only_once(self) -> None:
        """Multiple transcribe() calls share the same model instance."""
        load_count = 0

        def _fake_load() -> _FakeWhisperModel:
            nonlocal load_count
            load_count += 1
            return _FakeWhisperModel()

        with patch("apps.channels.voice.stt.STT._load_model", side_effect=_fake_load):
            stt = STT()
            await stt.transcribe(_make_pcm())
            await stt.transcribe(_make_pcm())
            await stt.transcribe(_make_pcm())

        assert load_count == 1


class TestModelSizeEnvOverride:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HER_STT_MODEL env var sets the model size."""
        monkeypatch.setenv("HER_STT_MODEL", "tiny")
        stt = STT()
        assert stt._model_size == "tiny"

    def test_explicit_arg_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit model_size arg wins over env var."""
        monkeypatch.setenv("HER_STT_MODEL", "tiny")
        stt = STT(model_size="large-v3")
        assert stt._model_size == "large-v3"

    def test_default_is_medium(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default model size is 'medium' when env var is unset."""
        monkeypatch.delenv("HER_STT_MODEL", raising=False)
        stt = STT()
        assert stt._model_size == "medium"


class TestPcmNormalization:
    def test_max_positive_int16(self) -> None:
        """int16 value 0x7FFF (32767) normalises to ≈ +1.0."""
        raw = struct.pack("<h", 32767)  # one sample = 0x7FFF
        arr = _pcm_to_float32(raw)
        assert arr.shape == (1,)
        assert abs(arr[0] - (32767 / 32768.0)) < 1e-5

    def test_sentinel_two_samples(self) -> None:
        """Two copies of 0xFF7F (int16 little-endian = 32511) → ~0.992."""
        raw = bytes([0xFF, 0x7F]) * 2  # bytes([0xff, 0x7f]) = int16 LE 0x7FFF
        arr = _pcm_to_float32(raw)
        # 0x7FFF = 32767, so value ≈ +0.9999...
        assert arr.shape == (2,)
        assert all(abs(v - 32767 / 32768.0) < 1e-4 for v in arr)

    def test_silence_is_zero(self) -> None:
        """All-zero PCM → all-zero float array."""
        raw = b"\x00\x00" * 10
        arr = _pcm_to_float32(raw)
        assert np.all(arr == 0.0)

    def test_dtype_is_float32(self) -> None:
        raw = b"\x00\x00" * 4
        arr = _pcm_to_float32(raw)
        assert arr.dtype == np.float32

    def test_range_within_minus_one_to_one(self) -> None:
        """All int16 values must map into [-1, 1]."""
        import random

        vals = [random.randint(-32768, 32767) for _ in range(200)]
        raw = struct.pack(f"<{len(vals)}h", *vals)
        arr = _pcm_to_float32(raw)
        assert np.all(arr >= -1.0)
        assert np.all(arr <= 1.0)
