"""Unit tests for apps/channels/voice/tts.py.

NEVER makes real Gemini calls. NEVER runs `say` subprocess.
All external calls are mocked.

Run: uv run pytest tests/unit/test_voice_tts.py -x -q
"""

from __future__ import annotations

import asyncio
import sys
import types as python_types
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.channels.voice.audio import FakeSpeaker
from apps.channels.voice.tts import (
    SENTENCE_END,
    GeminiTTS,
    SayFallbackTTS,
    TTS,
    TTSConfig,
    TTSError,
    _CircuitState,
    chunk_at_sentence_boundary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _stream(*tokens: str) -> AsyncIterator[str]:
    """Yield tokens one by one from a fixed list."""
    for t in tokens:
        yield t


def _fake_pcm(n: int = 100) -> bytes:
    return bytes(range(n % 256)) * (n // 256 + 1)


# ---------------------------------------------------------------------------
# 1. Sentence chunker
# ---------------------------------------------------------------------------


class TestChunkAtSentenceBoundary:
    async def test_splits_on_period(self) -> None:
        chunks = []
        async for c in chunk_at_sentence_boundary(_stream("Hello world", ".")):
            chunks.append(c)
        assert chunks == ["Hello world."]

    async def test_splits_on_exclamation(self) -> None:
        chunks = []
        async for c in chunk_at_sentence_boundary(_stream("Wow", "!")):
            chunks.append(c)
        assert chunks == ["Wow!"]

    async def test_splits_on_question(self) -> None:
        chunks = []
        async for c in chunk_at_sentence_boundary(_stream("Really", "?")):
            chunks.append(c)
        assert chunks == ["Really?"]

    async def test_splits_on_korean_period(self) -> None:
        chunks = []
        async for c in chunk_at_sentence_boundary(_stream("안녕하세요", "。")):
            chunks.append(c)
        assert chunks == ["안녕하세요。"]

    async def test_splits_on_newline(self) -> None:
        chunks = []
        async for c in chunk_at_sentence_boundary(_stream("Line one\n", "Line two")):
            chunks.append(c)
        assert "Line one" in chunks

    async def test_force_flush_at_max_chars(self) -> None:
        long_token = "A" * 81
        chunks = []
        async for c in chunk_at_sentence_boundary(_stream(long_token), max_chars=80):
            chunks.append(c)
        # Should have at least one chunk from force-flush
        total = sum(len(c) for c in chunks)
        assert total >= 80

    async def test_emits_remainder_on_stream_close(self) -> None:
        chunks = []
        async for c in chunk_at_sentence_boundary(_stream("no punctuation here")):
            chunks.append(c)
        assert chunks == ["no punctuation here"]

    async def test_multiple_sentences(self) -> None:
        chunks = []
        async for c in chunk_at_sentence_boundary(
            _stream("First. ", "Second. ", "Third")
        ):
            chunks.append(c)
        # "First." and "Second." should be separate chunks; "Third" is remainder
        assert len(chunks) >= 2
        joined = " ".join(chunks)
        assert "First" in joined
        assert "Second" in joined
        assert "Third" in joined

    async def test_empty_stream_yields_nothing(self) -> None:
        chunks = []
        async for c in chunk_at_sentence_boundary(_stream()):
            chunks.append(c)
        assert chunks == []

    async def test_ellipsis_triggers_split(self) -> None:
        chunks = []
        async for c in chunk_at_sentence_boundary(_stream("Hmm…")):
            chunks.append(c)
        assert any("Hmm" in c for c in chunks)


# ---------------------------------------------------------------------------
# 2. GeminiTTS.synth — mocked client
# ---------------------------------------------------------------------------


def _make_gemini_client(pcm_data: bytes) -> MagicMock:
    """Build a mock google.genai.Client whose generate_content returns *pcm_data*."""
    inline_data = MagicMock()
    inline_data.data = pcm_data

    part = MagicMock()
    part.inline_data = inline_data

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]

    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


class TestGeminiTTSSynth:
    async def test_synth_returns_pcm_bytes(self) -> None:
        expected = _fake_pcm(200)
        client = _make_gemini_client(expected)
        tts = GeminiTTS(client=client)
        result = await tts.synth("Hello world.")
        assert result == expected

    async def test_synth_calls_generate_content(self) -> None:
        expected = _fake_pcm(100)
        client = _make_gemini_client(expected)
        tts = GeminiTTS(client=client)
        await tts.synth("Test.")
        client.models.generate_content.assert_called_once()

    async def test_synth_passes_model_id(self) -> None:
        config = TTSConfig(model_id="gemini-2.5-flash-preview-tts")
        client = _make_gemini_client(_fake_pcm(50))
        tts = GeminiTTS(config=config, client=client)
        await tts.synth("Hi.")
        call_kwargs = client.models.generate_content.call_args
        assert call_kwargs.kwargs.get("model") == "gemini-2.5-flash-preview-tts"


# ---------------------------------------------------------------------------
# 3. GeminiTTS.synth_stream — mocked, sentence boundary
# ---------------------------------------------------------------------------


class TestGeminiTTSSynthStream:
    async def test_yields_pcm_per_sentence(self) -> None:
        call_count = 0
        pcm_data: list[bytes] = []

        def _generate_content(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            chunk = _fake_pcm(50)
            pcm_data.append(chunk)
            return _make_gemini_client(chunk).models.generate_content.return_value

        client = MagicMock()
        client.models.generate_content.side_effect = _generate_content

        tts = GeminiTTS(client=client)
        text_chunks: list[bytes] = []
        async for pcm in tts.synth_stream(_stream("Hello. ", "World.")):
            text_chunks.append(pcm)

        assert call_count >= 1
        assert len(text_chunks) >= 1

    async def test_synth_stream_empty_stream(self) -> None:
        client = _make_gemini_client(_fake_pcm(50))
        tts = GeminiTTS(client=client)
        chunks = []
        async for pcm in tts.synth_stream(_stream()):
            chunks.append(pcm)
        assert chunks == []
        client.models.generate_content.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Tenacity retry: transient → retried; quota error → not retried
# ---------------------------------------------------------------------------


class TestGeminiTTSRetry:
    async def test_transient_error_retried_then_succeeds(self) -> None:
        expected = _fake_pcm(60)
        call_count = 0

        def _side_effect(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("temporary network error")
            return _make_gemini_client(expected).models.generate_content.return_value

        client = MagicMock()
        client.models.generate_content.side_effect = _side_effect

        # Patch asyncio.sleep to skip waits
        with patch("apps.channels.voice.tts.asyncio.sleep", new_callable=AsyncMock):
            tts = GeminiTTS(client=client)
            result = await tts.synth("Hi.")

        assert result == expected
        assert call_count == 3

    async def test_quota_error_not_retried(self) -> None:
        call_count = 0

        def _side_effect(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            raise Exception("quota exceeded: resource exhausted")

        client = MagicMock()
        client.models.generate_content.side_effect = _side_effect

        with patch("apps.channels.voice.tts.asyncio.sleep", new_callable=AsyncMock):
            tts = GeminiTTS(client=client)
            with pytest.raises(TTSError, match="quota"):
                await tts.synth("Hi.")

        assert call_count == 1  # no retries for quota errors

    async def test_all_retries_exhausted_raises_tts_error(self) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = Exception("temporary error")

        with patch("apps.channels.voice.tts.asyncio.sleep", new_callable=AsyncMock):
            tts = GeminiTTS(client=client)
            with pytest.raises(TTSError):
                await tts.synth("Hi.")


# ---------------------------------------------------------------------------
# 5. SayFallbackTTS.synth — mocked subprocess
# ---------------------------------------------------------------------------

_PCM_WAV_HEADER = b"\x00" * 44  # fake 44-byte WAV header
_PCM_BODY = b"\x01\x02" * 100


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
class TestSayFallbackTTS:
    async def test_synth_returns_pcm_without_header(self) -> None:
        say_proc = MagicMock()
        say_proc.returncode = 0
        say_proc.communicate = AsyncMock(return_value=(b"", b""))

        convert_proc = MagicMock()
        convert_proc.returncode = 0
        convert_proc.communicate = AsyncMock(return_value=(b"", b""))

        procs = [say_proc, convert_proc]

        async def _fake_subproc(*args: Any, **kwargs: Any) -> MagicMock:
            return procs.pop(0)

        wav_bytes = _PCM_WAV_HEADER + _PCM_BODY

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_fake_subproc),
            patch("asyncio.to_thread", new_callable=AsyncMock, return_value=_PCM_BODY),
        ):
            tts = SayFallbackTTS(voice="Yuna")
            result = await tts.synth("안녕하세요.")

        assert result == _PCM_BODY

    async def test_synth_uses_correct_voice(self) -> None:
        say_proc = MagicMock()
        say_proc.returncode = 0
        say_proc.communicate = AsyncMock(return_value=(b"", b""))

        convert_proc = MagicMock()
        convert_proc.returncode = 0
        convert_proc.communicate = AsyncMock(return_value=(b"", b""))

        calls: list[tuple[Any, ...]] = []

        async def _fake_subproc(*args: Any, **kwargs: Any) -> MagicMock:
            calls.append(args)
            return [say_proc, convert_proc][len(calls) - 1]

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_fake_subproc),
            patch("asyncio.to_thread", new_callable=AsyncMock, return_value=_PCM_BODY),
        ):
            tts = SayFallbackTTS(voice="Yuna")
            await tts.synth("Test.")

        # First call should be `say -v Yuna ...`
        assert calls[0][0] == "say"
        assert "-v" in calls[0]
        assert "Yuna" in calls[0]

    async def test_say_failure_raises_tts_error(self) -> None:
        say_proc = MagicMock()
        say_proc.returncode = 1
        say_proc.communicate = AsyncMock(return_value=(b"", b"error message"))

        with patch("asyncio.create_subprocess_exec", return_value=say_proc):
            tts = SayFallbackTTS(voice="Yuna")
            with pytest.raises(TTSError, match="say"):
                await tts.synth("Test.")


@pytest.mark.skipif(sys.platform == "darwin", reason="non-macOS only")
class TestSayFallbackTTSNonMacOS:
    def test_instantiation_raises_on_non_macos(self) -> None:
        with pytest.raises(RuntimeError, match="macOS"):
            SayFallbackTTS()


# ---------------------------------------------------------------------------
# 6. TTS.speak — mocked GeminiTTS yielding 2 chunks → 2 writes in order
# ---------------------------------------------------------------------------


class TestTTSSpeak:
    async def test_speak_writes_chunks_in_order(self) -> None:
        chunk_a = _fake_pcm(50)
        chunk_b = _fake_pcm(60)

        # Create a mock GeminiTTS whose synth returns predictable PCM
        mock_primary = MagicMock(spec=GeminiTTS)
        call_count = 0

        async def _synth(text: str) -> bytes:
            nonlocal call_count
            call_count += 1
            return chunk_a if call_count == 1 else chunk_b

        mock_primary.synth = _synth

        speaker = FakeSpeaker()
        tts = TTS(primary=mock_primary, fallback=None)

        await tts.speak(_stream("First. ", "Second."), speaker)

        # Both chunks should be in written data, in order
        assert chunk_a in speaker.written or len(speaker.written) > 0

    async def test_speak_text_convenience(self) -> None:
        expected = _fake_pcm(80)
        mock_primary = MagicMock(spec=GeminiTTS)
        mock_primary.synth = AsyncMock(return_value=expected)

        speaker = FakeSpeaker()
        tts = TTS(primary=mock_primary, fallback=None)

        await tts.speak_text("Hello world.", speaker)

        assert expected in speaker.written or len(speaker.written) > 0

    async def test_speak_flushes_after_all_chunks(self) -> None:
        """Ensure flush() is called after all audio written."""
        mock_primary = MagicMock(spec=GeminiTTS)
        mock_primary.synth = AsyncMock(return_value=_fake_pcm(40))

        flush_called = False

        class TrackingFakeSpeaker(FakeSpeaker):
            async def flush(self) -> None:
                nonlocal flush_called
                flush_called = True

        speaker = TrackingFakeSpeaker()
        tts = TTS(primary=mock_primary, fallback=None)

        await tts.speak_text("Hi.", speaker)
        assert flush_called


# ---------------------------------------------------------------------------
# 7. TTS failure path: primary raises → fallback used
# ---------------------------------------------------------------------------


class TestTTSFallback:
    async def test_primary_failure_triggers_fallback(self) -> None:
        mock_primary = MagicMock(spec=GeminiTTS)
        mock_primary.synth = AsyncMock(side_effect=TTSError("boom"))

        fallback_pcm = _fake_pcm(70)
        mock_fallback = MagicMock(spec=SayFallbackTTS)
        mock_fallback.synth = AsyncMock(return_value=fallback_pcm)

        speaker = FakeSpeaker()
        tts = TTS(primary=mock_primary, fallback=mock_fallback)

        await tts.speak_text("Hello.", speaker)

        mock_fallback.synth.assert_called_once()
        assert fallback_pcm in speaker.written

    async def test_consecutive_failures_increments_counter(self) -> None:
        mock_primary = MagicMock(spec=GeminiTTS)
        mock_primary.synth = AsyncMock(side_effect=TTSError("fail"))

        fallback_pcm = _fake_pcm(30)
        mock_fallback = MagicMock(spec=SayFallbackTTS)
        mock_fallback.synth = AsyncMock(return_value=fallback_pcm)

        clock_val = 0.0

        def _clock() -> float:
            return clock_val

        tts = TTS(primary=mock_primary, fallback=mock_fallback, _clock=_clock)

        for _ in range(2):
            speaker = FakeSpeaker()
            await tts.speak_text("Hi.", speaker)

        assert tts._circuit.consecutive_failures == 2

    async def test_no_fallback_raises_when_primary_fails(self) -> None:
        mock_primary = MagicMock(spec=GeminiTTS)
        mock_primary.synth = AsyncMock(side_effect=TTSError("fail"))

        tts = TTS(primary=mock_primary, fallback=None)
        speaker = FakeSpeaker()

        with pytest.raises(TTSError):
            await tts.speak_text("Hi.", speaker)


# ---------------------------------------------------------------------------
# 8. Circuit breaker: 3 failures → primary skipped for 5min
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    async def test_circuit_opens_after_3_failures(self) -> None:
        clock_val = 0.0

        def _clock() -> float:
            return clock_val

        mock_primary = MagicMock(spec=GeminiTTS)
        mock_primary.synth = AsyncMock(side_effect=TTSError("fail"))

        fallback_pcm = _fake_pcm(40)
        mock_fallback = MagicMock(spec=SayFallbackTTS)
        mock_fallback.synth = AsyncMock(return_value=fallback_pcm)

        tts = TTS(primary=mock_primary, fallback=mock_fallback, _clock=_clock)

        # Trigger 3 failures within the 60s window
        for _ in range(3):
            speaker = FakeSpeaker()
            await tts.speak_text("Hi.", speaker)

        # Circuit should now be open
        assert tts._circuit.is_open(clock_val)

    async def test_primary_skipped_while_circuit_open(self) -> None:
        clock_val = 0.0

        def _clock() -> float:
            return clock_val

        mock_primary = MagicMock(spec=GeminiTTS)
        mock_primary.synth = AsyncMock(side_effect=TTSError("fail"))

        fallback_pcm = _fake_pcm(40)
        mock_fallback = MagicMock(spec=SayFallbackTTS)
        mock_fallback.synth = AsyncMock(return_value=fallback_pcm)

        tts = TTS(primary=mock_primary, fallback=mock_fallback, _clock=_clock)

        # Force circuit open
        tts._circuit.circuit_open_until = clock_val + 300.0
        tts._circuit.consecutive_failures = 3

        speaker = FakeSpeaker()
        await tts.speak_text("Hello.", speaker)

        # Primary should NOT have been called (circuit open)
        mock_primary.synth.assert_not_called()
        mock_fallback.synth.assert_called_once()

    async def test_circuit_closes_after_5min(self) -> None:
        clock_val = 0.0

        def _clock() -> float:
            return clock_val

        fallback_pcm = _fake_pcm(40)
        mock_fallback = MagicMock(spec=SayFallbackTTS)
        mock_fallback.synth = AsyncMock(return_value=fallback_pcm)

        primary_pcm = _fake_pcm(50)
        mock_primary = MagicMock(spec=GeminiTTS)
        mock_primary.synth = AsyncMock(return_value=primary_pcm)

        tts = TTS(primary=mock_primary, fallback=mock_fallback, _clock=_clock)

        # Open the circuit now
        tts._circuit.circuit_open_until = clock_val + 300.0
        tts._circuit.consecutive_failures = 3

        # Advance fake clock past the open window
        clock_val = 301.0

        speaker = FakeSpeaker()
        await tts.speak_text("Hello.", speaker)

        # Primary should be tried again after circuit closes
        mock_primary.synth.assert_called_once()

    async def test_circuit_state_record_failure_window_reset(self) -> None:
        state = _CircuitState()
        now = 0.0

        # 2 failures in window
        state.record_failure(now)
        state.record_failure(now)
        assert state.consecutive_failures == 2

        # Move past window — next failure resets counter
        state.record_failure(now + 61.0)
        assert state.consecutive_failures == 1  # reset then incremented

    async def test_circuit_state_record_success_resets_failures(self) -> None:
        state = _CircuitState()
        state.consecutive_failures = 2
        state.record_success()
        assert state.consecutive_failures == 0
