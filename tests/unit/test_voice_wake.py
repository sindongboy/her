"""Unit tests for apps/channels/voice/wake.py and wake-word integration
into VoiceChannel.

No pvporcupine or real STT/VAD inference required — all tests use fakes.

Run:
    cd /Users/sindongboy/workspace/her
    uv run pytest tests/unit/test_voice_wake.py -x -q
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from apps.channels.voice.audio import AudioFormat, FakeSpeaker
from apps.channels.voice.channel import VoiceChannel


# ---------------------------------------------------------------------------
# Fake STT — returns canned transcripts in sequence
# ---------------------------------------------------------------------------


class FakeSTT:
    """Returns transcripts from a pre-loaded list, one per call."""

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)
        self._index = 0
        self.warmup_called = False
        self.transcribe_calls: int = 0

    async def warmup(self) -> None:
        self.warmup_called = True

    async def transcribe(self, pcm_bytes: bytes, **kwargs: Any) -> str:
        self.transcribe_calls += 1
        if self._index < len(self._transcripts):
            result = self._transcripts[self._index]
            self._index += 1
            return result
        return ""


# ---------------------------------------------------------------------------
# Fake VAD — uses a marker byte to classify frames
# ---------------------------------------------------------------------------


class FakeVAD:
    """Classifies a frame as speech if it is NOT all-zero bytes.

    Tests embed speech by using any non-zero byte value (e.g. 0x7f).
    Silence frames are all-zero.
    """

    _speech_pad_ms = 100

    def is_speech(self, frame: bytes) -> bool:
        return frame != b"\x00" * len(frame)

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Other fakes shared with channel tests
# ---------------------------------------------------------------------------


class FakeTTS:
    def __init__(self) -> None:
        self.speak_calls: list[list[str]] = []
        self.speak_text_calls: list[str] = []

    async def speak(self, text_stream: AsyncIterator[str], output: Any) -> None:
        chunks: list[str] = []
        async for chunk in text_stream:
            chunks.append(chunk)
        self.speak_calls.append(chunks)
        await output.write(b"\xAA" * 32)

    async def speak_text(self, text: str, output: Any) -> None:
        self.speak_text_calls.append(text)
        await output.write(b"\xBB" * 32)


class FakeAgent:
    def __init__(self, responses: list[list[str]]) -> None:
        self._responses = list(responses)
        self._index = 0
        self._ep = 1
        self.last_episode_id: int | None = None
        self.stream_calls: list[tuple[str, int | None]] = []

    async def stream_respond(
        self,
        message: str,
        *,
        episode_id: int | None = None,
        channel: str = "text",
    ) -> AsyncIterator[str]:
        self.stream_calls.append((message, episode_id))
        ep = episode_id if episode_id is not None else self._ep
        self._ep = ep + 1
        self.last_episode_id = ep
        chunks = self._responses[self._index] if self._index < len(self._responses) else ["ok"]
        self._index += 1
        return _async_iter(chunks)


async def _async_iter(items: list[Any]) -> AsyncIterator[Any]:  # type: ignore[misc]
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

_FRAME_MS = 32
_FRAME_SAMPLES = 512
_FRAME_BYTES = _FRAME_SAMPLES * 2

_SILENCE_FRAME = b"\x00" * _FRAME_BYTES
_SPEECH_FRAME = b"\x7f" * _FRAME_BYTES  # non-zero → speech for FakeVAD


def _speech(n: int) -> list[bytes]:
    return [_SPEECH_FRAME] * n


def _silence(n: int) -> list[bytes]:
    return [_SILENCE_FRAME] * n


class ListMic:
    """Mic that yields a fixed list of frames then EOF."""

    fmt = AudioFormat()

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames

    async def frames(self, *, frame_ms: int = _FRAME_MS) -> AsyncIterator[bytes]:
        for f in self._frames:
            yield f

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 1. matches_wake — pure utility
# ---------------------------------------------------------------------------


class TestMatchesWake:
    def test_basic_substring_match(self) -> None:
        from apps.channels.voice.wake import WakeConfig, matches_wake

        cfg = WakeConfig(keyword="자기야")
        assert matches_wake("자기야", cfg) is True
        assert matches_wake("야 자기야 오늘 어때?", cfg) is True

    def test_no_match_returns_false(self) -> None:
        from apps.channels.voice.wake import WakeConfig, matches_wake

        cfg = WakeConfig(keyword="자기야")
        assert matches_wake("안녕하세요", cfg) is False
        assert matches_wake("", cfg) is False

    def test_aliases_counted(self) -> None:
        from apps.channels.voice.wake import WakeConfig, matches_wake

        cfg = WakeConfig(keyword="자기야", aliases=("비서야", "헤이"))
        assert matches_wake("비서야 일어나", cfg) is True
        assert matches_wake("헤이 거기", cfg) is True
        assert matches_wake("어디야", cfg) is False

    def test_whitespace_normalised(self) -> None:
        from apps.channels.voice.wake import WakeConfig, matches_wake

        cfg = WakeConfig(keyword="자기야", strip_whitespace=True)
        # Extra surrounding whitespace in transcript is stripped
        assert matches_wake("  자기야  ", cfg) is True

    def test_case_insensitive_ascii(self) -> None:
        from apps.channels.voice.wake import WakeConfig, matches_wake

        cfg = WakeConfig(keyword="hey", case_sensitive=False)
        assert matches_wake("HEY there", cfg) is True
        assert matches_wake("Hey", cfg) is True

    def test_case_sensitive_ascii(self) -> None:
        from apps.channels.voice.wake import WakeConfig, matches_wake

        cfg = WakeConfig(keyword="Hey", case_sensitive=True)
        assert matches_wake("HEY there", cfg) is False
        assert matches_wake("Hey there", cfg) is True

    def test_korean_substring_no_normalisation_needed(self) -> None:
        """Korean words don't need special normalisation beyond NFC."""
        from apps.channels.voice.wake import WakeConfig, matches_wake

        cfg = WakeConfig(keyword="비서야")
        assert matches_wake("아 비서야 잠깐만", cfg) is True
        assert matches_wake("비서야!", cfg) is True

    def test_empty_keyword_never_matches(self) -> None:
        from apps.channels.voice.wake import WakeConfig, matches_wake

        cfg = WakeConfig(keyword="")
        # Empty keyword after normalisation → no match (guard: `if norm_phrase`)
        assert matches_wake("아무말", cfg) is False


# ---------------------------------------------------------------------------
# 2. detect_quiet_intent — preserve existing behaviour
# ---------------------------------------------------------------------------


class TestDetectQuietIntent:
    def test_on_phrases_return_on(self) -> None:
        from apps.channels.voice.wake import detect_quiet_intent

        assert detect_quiet_intent("조용히 해") == "on"
        assert detect_quiet_intent("조용 모드") == "on"
        assert detect_quiet_intent("조용히") == "on"

    def test_off_phrases_return_off(self) -> None:
        from apps.channels.voice.wake import detect_quiet_intent

        assert detect_quiet_intent("이제 말해도 돼") == "off"
        assert detect_quiet_intent("다시 말해줘") == "off"
        assert detect_quiet_intent("조용 모드 해제") == "off"

    def test_unrelated_text_returns_none(self) -> None:
        from apps.channels.voice.wake import detect_quiet_intent

        assert detect_quiet_intent("안녕하세요") is None
        assert detect_quiet_intent("오늘 날씨 어때?") is None
        assert detect_quiet_intent("") is None
        assert detect_quiet_intent("   ") is None

    def test_off_takes_priority_over_on_substring(self) -> None:
        """'조용 모드 해제' contains '조용 모드' but should return 'off'."""
        from apps.channels.voice.wake import detect_quiet_intent

        assert detect_quiet_intent("조용 모드 해제") == "off"

    def test_whitespace_trimmed(self) -> None:
        from apps.channels.voice.wake import detect_quiet_intent

        assert detect_quiet_intent("  조용히 해  ") == "on"
        assert detect_quiet_intent("  이제 말해도 돼  ") == "off"

    def test_phrase_embedded_in_longer_sentence(self) -> None:
        from apps.channels.voice.wake import detect_quiet_intent

        assert detect_quiet_intent("지금부터 조용히 해줘") == "on"
        assert detect_quiet_intent("알겠어, 이제 말해도 돼") == "off"


# ---------------------------------------------------------------------------
# 3. WakeDetector.wait_for_wake — core behaviour
# ---------------------------------------------------------------------------


class TestWakeDetectorWaitForWake:
    @pytest.mark.asyncio
    async def test_returns_transcript_on_match(self) -> None:
        """wait_for_wake returns the matching transcript."""
        from apps.channels.voice.wake import WakeConfig, WakeDetector

        stt = FakeSTT(["자기야 안녕"])
        vad = FakeVAD()
        detector = WakeDetector(stt=stt, vad=vad, config=WakeConfig(keyword="자기야"))

        q: asyncio.Queue[bytes | None] = asyncio.Queue()
        # Speech segment: 7 speech frames + silence to end segment
        for _ in range(7):
            await q.put(_SPEECH_FRAME)
        for _ in range(20):
            await q.put(_SILENCE_FRAME)
        await q.put(None)

        result = await asyncio.wait_for(detector.wait_for_wake(q), timeout=5.0)
        assert result == "자기야 안녕"

    @pytest.mark.asyncio
    async def test_ignores_non_matching_utterances(self) -> None:
        """Non-matching utterances are skipped; matching one triggers return."""
        from apps.channels.voice.wake import WakeConfig, WakeDetector

        # First two transcripts don't match; third does
        stt = FakeSTT(["안녕", "오늘 날씨", "자기야 일어나"])
        vad = FakeVAD()
        detector = WakeDetector(stt=stt, vad=vad, config=WakeConfig(keyword="자기야"))

        q: asyncio.Queue[bytes | None] = asyncio.Queue()

        # Three speech segments separated by silence
        for _ in range(3):
            for _ in range(5):
                await q.put(_SPEECH_FRAME)
            for _ in range(15):
                await q.put(_SILENCE_FRAME)
        await q.put(None)

        result = await asyncio.wait_for(detector.wait_for_wake(q), timeout=5.0)
        assert result == "자기야 일어나"
        assert stt.transcribe_calls == 3

    @pytest.mark.asyncio
    async def test_queue_closed_raises_wake_word_error(self) -> None:
        """None sentinel before any match → WakeWordError raised."""
        from apps.channels.voice.wake import WakeConfig, WakeDetector, WakeWordError

        stt = FakeSTT([])
        vad = FakeVAD()
        detector = WakeDetector(stt=stt, vad=vad, config=WakeConfig(keyword="자기야"))

        q: asyncio.Queue[bytes | None] = asyncio.Queue()
        await q.put(None)  # immediate EOF

        with pytest.raises(WakeWordError, match="queue closed before wake"):
            await asyncio.wait_for(detector.wait_for_wake(q), timeout=2.0)

    @pytest.mark.asyncio
    async def test_queue_closed_after_non_matching_utterances(self) -> None:
        """EOF after non-matching utterances → WakeWordError (not silent return)."""
        from apps.channels.voice.wake import WakeConfig, WakeDetector, WakeWordError

        stt = FakeSTT(["안녕", "날씨 어때"])
        vad = FakeVAD()
        detector = WakeDetector(stt=stt, vad=vad, config=WakeConfig(keyword="자기야"))

        q: asyncio.Queue[bytes | None] = asyncio.Queue()
        for _ in range(2):
            for _ in range(5):
                await q.put(_SPEECH_FRAME)
            for _ in range(15):
                await q.put(_SILENCE_FRAME)
        await q.put(None)

        with pytest.raises(WakeWordError):
            await asyncio.wait_for(detector.wait_for_wake(q), timeout=5.0)

    @pytest.mark.asyncio
    async def test_returned_transcript_includes_full_text(self) -> None:
        """Transcript includes text after the wake phrase (caller strips if needed)."""
        from apps.channels.voice.wake import WakeConfig, WakeDetector

        full_transcript = "자기야 오늘 날씨 어때?"
        stt = FakeSTT([full_transcript])
        vad = FakeVAD()
        detector = WakeDetector(stt=stt, vad=vad, config=WakeConfig(keyword="자기야"))

        q: asyncio.Queue[bytes | None] = asyncio.Queue()
        for _ in range(7):
            await q.put(_SPEECH_FRAME)
        for _ in range(20):
            await q.put(_SILENCE_FRAME)
        await q.put(None)

        result = await asyncio.wait_for(detector.wait_for_wake(q), timeout=5.0)
        assert result == full_transcript

    @pytest.mark.asyncio
    async def test_aliases_trigger_wake(self) -> None:
        """An alias phrase also counts as a wake word."""
        from apps.channels.voice.wake import WakeConfig, WakeDetector

        stt = FakeSTT(["비서야 잠깐"])
        vad = FakeVAD()
        cfg = WakeConfig(keyword="자기야", aliases=("비서야",))
        detector = WakeDetector(stt=stt, vad=vad, config=cfg)

        q: asyncio.Queue[bytes | None] = asyncio.Queue()
        for _ in range(7):
            await q.put(_SPEECH_FRAME)
        for _ in range(20):
            await q.put(_SILENCE_FRAME)
        await q.put(None)

        result = await asyncio.wait_for(detector.wait_for_wake(q), timeout=5.0)
        assert "비서야" in result

    def test_close_is_noop(self) -> None:
        """close() does not raise and is safe to call."""
        from apps.channels.voice.wake import WakeDetector

        stt = FakeSTT([])
        vad = FakeVAD()
        detector = WakeDetector(stt=stt, vad=vad)
        detector.close()  # must not raise

    def test_keyword_property(self) -> None:
        from apps.channels.voice.wake import WakeConfig, WakeDetector

        cfg = WakeConfig(keyword="비서야")
        detector = WakeDetector(stt=FakeSTT([]), vad=FakeVAD(), config=cfg)
        assert detector.keyword == "비서야"


# ---------------------------------------------------------------------------
# 4. WakeDetector — STT error is skipped, not fatal
# ---------------------------------------------------------------------------


class TestWakeDetectorSTTError:
    @pytest.mark.asyncio
    async def test_stt_error_skips_segment(self) -> None:
        """An STT exception on one segment is caught; next segment is tried."""
        from apps.channels.voice.wake import WakeConfig, WakeDetector

        class BrokenThenOkSTT:
            def __init__(self) -> None:
                self._calls = 0

            async def warmup(self) -> None:
                pass

            async def transcribe(self, pcm_bytes: bytes, **kw: Any) -> str:
                self._calls += 1
                if self._calls == 1:
                    raise RuntimeError("STT failed")
                return "자기야"

        stt = BrokenThenOkSTT()
        vad = FakeVAD()
        detector = WakeDetector(stt=stt, vad=vad, config=WakeConfig(keyword="자기야"))

        q: asyncio.Queue[bytes | None] = asyncio.Queue()
        # Two segments
        for _ in range(2):
            for _ in range(7):
                await q.put(_SPEECH_FRAME)
            for _ in range(20):
                await q.put(_SILENCE_FRAME)
        await q.put(None)

        result = await asyncio.wait_for(detector.wait_for_wake(q), timeout=5.0)
        assert result == "자기야"


# ---------------------------------------------------------------------------
# Helpers: fake WakeDetector factories for VoiceChannel tests
# ---------------------------------------------------------------------------


class _ImmediateWakeDetector:
    """Fires immediately on the first (or n-th) wait_for_wake call."""

    def __init__(self, fires_on_call: int = 0) -> None:
        self._target = fires_on_call
        self._call_count = 0
        self.closed = False

    async def wait_for_wake(self, queue: asyncio.Queue[bytes | None]) -> str:
        if self._call_count >= self._target:
            self._call_count += 1
            return "자기야"
        self._call_count += 1
        # Drain queue before returning
        try:
            while True:
                item = queue.get_nowait()
                if item is None:
                    break
        except asyncio.QueueEmpty:
            pass
        return "자기야"

    def close(self) -> None:
        self.closed = True


class _NeverWakeDetector:
    """Never wakes — drains queue and returns only on EOF sentinel."""

    def __init__(self) -> None:
        self.closed = False

    async def wait_for_wake(self, queue: asyncio.Queue[bytes | None]) -> str:
        while True:
            item = await queue.get()
            if item is None:
                raise __import__("apps.channels.voice.wake", fromlist=["WakeWordError"]).WakeWordError(
                    "queue closed before wake"
                )
        # unreachable — silences mypy
        return ""  # pragma: no cover

    def close(self) -> None:
        self.closed = True


def _default_wake_factory(wake_detector: Any) -> Any:
    """Return a factory compatible with the new (stt, vad, *, keyword) signature."""
    return lambda stt, vad, *, keyword: wake_detector


# ---------------------------------------------------------------------------
# Helper: build a wake-enabled VoiceChannel
# ---------------------------------------------------------------------------


def _make_wake_channel(
    *,
    mic_frames: list[bytes],
    transcripts: list[str],
    agent_responses: list[list[str]],
    wake_detector: Any,
    status_log: list[str] | None = None,
    is_quiet: bool = False,
    enable_interruption: bool = False,
    active_window_s: float = 0.5,
    tts: FakeTTS | None = None,
) -> tuple[VoiceChannel, FakeAgent, FakeTTS, FakeSpeaker]:
    mic = ListMic(mic_frames)
    speaker = FakeSpeaker()
    stt = FakeSTT(transcripts)
    tts_inst = tts or FakeTTS()
    agent = FakeAgent(agent_responses)
    vad = FakeVAD()
    status: list[str] = [] if status_log is None else status_log

    channel = VoiceChannel(
        agent,
        object(),
        mic=mic,
        speaker=speaker,
        vad=vad,
        stt=stt,
        tts=tts_inst,
        status_fn=lambda msg: status.append(msg),
        enable_wake_word=True,
        wake_detector_fn=_default_wake_factory(wake_detector),
        wake_active_window_s=active_window_s,
        is_quiet=is_quiet,
        enable_interruption=enable_interruption,
    )
    return channel, agent, tts_inst, speaker


# ---------------------------------------------------------------------------
# 5. VoiceChannel: stays silent until wake fires, then handles utterance
# ---------------------------------------------------------------------------


class TestWakeWordModeBasic:
    @pytest.mark.asyncio
    async def test_channel_stays_silent_then_responds(self) -> None:
        """Channel runs with wake mode; wakes and processes one utterance."""
        frames = _speech(7) + _silence(22)
        wake_detector = _ImmediateWakeDetector(fires_on_call=0)

        channel, agent, tts, speaker = _make_wake_channel(
            mic_frames=frames,
            transcripts=["안녕"],
            agent_responses=[["안녕하세요!"]],
            wake_detector=wake_detector,
            active_window_s=0.5,
        )

        await asyncio.wait_for(channel.run(), timeout=5.0)

        assert len(agent.stream_calls) == 1
        assert agent.stream_calls[0][0] == "안녕"
        assert len(tts.speak_calls) == 1

    @pytest.mark.asyncio
    async def test_wake_status_emitted(self) -> None:
        """[깨어남] status is emitted after wake word fires."""
        frames = _speech(7) + _silence(22)
        status_log: list[str] = []
        wake_detector = _ImmediateWakeDetector(fires_on_call=0)

        channel, _, _, _ = _make_wake_channel(
            mic_frames=frames,
            transcripts=["안녕"],
            agent_responses=[["응답"]],
            wake_detector=wake_detector,
            status_log=status_log,
            active_window_s=0.5,
        )

        await asyncio.wait_for(channel.run(), timeout=5.0)
        assert any("깨어남" in m for m in status_log), f"Status: {status_log}"

    @pytest.mark.asyncio
    async def test_sleeping_status_emitted_before_wake(self) -> None:
        """[잠듦] status is emitted while waiting for wake word."""
        frames = _speech(7) + _silence(22)
        status_log: list[str] = []
        wake_detector = _ImmediateWakeDetector(fires_on_call=0)

        channel, _, _, _ = _make_wake_channel(
            mic_frames=frames,
            transcripts=["안녕"],
            agent_responses=[["응답"]],
            wake_detector=wake_detector,
            status_log=status_log,
            active_window_s=0.5,
        )

        await asyncio.wait_for(channel.run(), timeout=5.0)
        assert any("잠듦" in m for m in status_log), f"Status: {status_log}"


# ---------------------------------------------------------------------------
# 6. Quiet-mode tests
# ---------------------------------------------------------------------------


class TestQuietModeOn:
    @pytest.mark.asyncio
    async def test_agent_called_tts_not_called_when_quiet(self) -> None:
        """With is_quiet=True, agent.stream_respond is called but tts.speak is not."""
        frames = _speech(7) + _silence(22)
        wake_detector = _ImmediateWakeDetector(fires_on_call=0)
        tts = FakeTTS()

        channel, agent, tts_inst, speaker = _make_wake_channel(
            mic_frames=frames,
            transcripts=["오늘 날씨 어때"],
            agent_responses=[["맑아요."]],
            wake_detector=wake_detector,
            is_quiet=True,
            tts=tts,
            active_window_s=0.5,
        )

        await asyncio.wait_for(channel.run(), timeout=5.0)

        assert len(agent.stream_calls) == 1
        assert len(tts_inst.speak_calls) == 0
        assert speaker.written == b""

    @pytest.mark.asyncio
    async def test_set_quiet_toggles_state(self) -> None:
        """set_quiet() changes is_quiet attribute."""
        wake_detector = _NeverWakeDetector()

        channel, _, _, _ = _make_wake_channel(
            mic_frames=[],
            transcripts=[],
            agent_responses=[],
            wake_detector=wake_detector,
        )

        assert channel.is_quiet is False
        channel.set_quiet(True)
        assert channel.is_quiet is True
        channel.set_quiet(False)
        assert channel.is_quiet is False


class TestQuietModeOff:
    @pytest.mark.asyncio
    async def test_off_phrase_triggers_ack_tts(self) -> None:
        """When user says quiet-off phrase, speak_text is called for ack."""
        frames = _speech(7) + _silence(22)
        wake_detector = _ImmediateWakeDetector(fires_on_call=0)
        tts = FakeTTS()

        channel, agent, tts_inst, speaker = _make_wake_channel(
            mic_frames=frames,
            transcripts=["이제 말해도 돼"],
            agent_responses=[["never called"]],
            wake_detector=wake_detector,
            is_quiet=True,
            tts=tts,
            active_window_s=0.5,
        )

        await asyncio.wait_for(channel.run(), timeout=5.0)

        assert len(agent.stream_calls) == 0
        assert len(tts_inst.speak_text_calls) >= 1
        assert channel.is_quiet is False

    @pytest.mark.asyncio
    async def test_on_phrase_suppresses_tts_and_stays_quiet(self) -> None:
        """When user says quiet-on phrase, no TTS, is_quiet becomes True."""
        frames = _speech(7) + _silence(22)
        wake_detector = _ImmediateWakeDetector(fires_on_call=0)
        tts = FakeTTS()

        channel, agent, tts_inst, speaker = _make_wake_channel(
            mic_frames=frames,
            transcripts=["조용히 해"],
            agent_responses=[["never called"]],
            wake_detector=wake_detector,
            is_quiet=False,
            tts=tts,
            active_window_s=0.5,
        )

        await asyncio.wait_for(channel.run(), timeout=5.0)

        assert len(agent.stream_calls) == 0
        assert len(tts_inst.speak_calls) == 0
        assert channel.is_quiet is True


# ---------------------------------------------------------------------------
# 7. Active window timeout
# ---------------------------------------------------------------------------


class TestActiveWindowTimeout:
    @pytest.mark.asyncio
    async def test_window_timeout_returns_to_sleeping(self) -> None:
        """After wake_active_window_s of no utterance, channel returns to sleeping."""
        frames = _silence(50)
        status_log: list[str] = []
        wake_detector = _ImmediateWakeDetector(fires_on_call=0)

        channel, agent, tts, _ = _make_wake_channel(
            mic_frames=frames,
            transcripts=[],
            agent_responses=[],
            wake_detector=wake_detector,
            status_log=status_log,
            active_window_s=0.1,
        )

        await asyncio.wait_for(channel.run(), timeout=5.0)

        assert len(agent.stream_calls) == 0
        assert any("잠" in m for m in status_log), f"Status: {status_log}"


# ---------------------------------------------------------------------------
# 8. Fanout distributes to multiple subscribers
# ---------------------------------------------------------------------------


class TestFanoutWithWakeAndInterrupt:
    @pytest.mark.asyncio
    async def test_fanout_distributes_to_multiple_subscribers(self) -> None:
        """_FrameFanout correctly serves wake_queue and interrupt_queue in parallel."""
        from apps.channels.voice.interrupt import _FrameFanout

        frames = [bytes([i % 256]) * _FRAME_BYTES for i in range(10)]

        class InfiniteListMic:
            fmt = AudioFormat()

            def __init__(self, frames_data: list[bytes]) -> None:
                self._frames = frames_data

            async def frames(self, *, frame_ms: int = _FRAME_MS) -> AsyncIterator[bytes]:
                for f in self._frames:
                    yield f

            async def close(self) -> None:
                pass

        mic = InfiniteListMic(frames)
        fanout = _FrameFanout(mic)
        q_wake = fanout.subscribe()
        q_interrupt = fanout.subscribe()

        await asyncio.create_task(fanout.run())

        received_wake: list[bytes] = []
        received_int: list[bytes] = []

        while not q_wake.empty():
            f = q_wake.get_nowait()
            if f is None:
                break
            received_wake.append(f)

        while not q_interrupt.empty():
            f = q_interrupt.get_nowait()
            if f is None:
                break
            received_int.append(f)

        assert received_wake == received_int == frames


# ---------------------------------------------------------------------------
# 9. Phase 1/2 regression guard
# ---------------------------------------------------------------------------


class TestPhase12Regression:
    @pytest.mark.asyncio
    async def test_linear_mode_unaffected_by_wake_params(self) -> None:
        """Constructing VoiceChannel without enable_wake_word still runs linear mode."""
        from apps.channels.voice.audio import FakeMicrophone

        stt = FakeSTT(["안녕"])
        tts = FakeTTS()
        agent = FakeAgent([["응답"]])
        vad = FakeVAD()
        mic = FakeMicrophone([_SPEECH_FRAME])
        speaker = FakeSpeaker()

        async def fake_segs(stream: Any, v: Any, **kw: Any) -> AsyncIterator[bytes]:
            yield _SPEECH_FRAME

        channel = VoiceChannel(
            agent,
            object(),
            mic=mic,
            speaker=speaker,
            vad=vad,
            stt=stt,
            tts=tts,
            speech_segments_fn=fake_segs,  # type: ignore[arg-type]
            enable_wake_word=False,
            enable_interruption=False,
        )

        await asyncio.wait_for(channel.run(), timeout=3.0)
        assert len(agent.stream_calls) == 1

    @pytest.mark.asyncio
    async def test_quiet_mode_off_by_default_in_linear(self) -> None:
        """VoiceChannel.is_quiet is False by default; TTS is called normally."""
        from apps.channels.voice.audio import FakeMicrophone

        stt = FakeSTT(["테스트"])
        tts = FakeTTS()
        agent = FakeAgent([["응답"]])
        vad = FakeVAD()
        mic = FakeMicrophone([_SPEECH_FRAME])
        speaker = FakeSpeaker()

        async def fake_segs(stream: Any, v: Any, **kw: Any) -> AsyncIterator[bytes]:
            yield _SPEECH_FRAME

        channel = VoiceChannel(
            agent,
            object(),
            mic=mic,
            speaker=speaker,
            vad=vad,
            stt=stt,
            tts=tts,
            speech_segments_fn=fake_segs,  # type: ignore[arg-type]
        )

        assert channel.is_quiet is False
        await asyncio.wait_for(channel.run(), timeout=3.0)
        assert len(tts.speak_calls) == 1


# ---------------------------------------------------------------------------
# 10. Module importable without any special deps
# ---------------------------------------------------------------------------


class TestModuleImport:
    def test_wake_module_importable(self) -> None:
        """Importing wake.py must not fail — no pvporcupine or real STT needed."""
        import importlib

        import apps.channels.voice.wake as wake_mod

        importlib.reload(wake_mod)
        assert hasattr(wake_mod, "WakeConfig")
        assert hasattr(wake_mod, "WakeDetector")
        assert hasattr(wake_mod, "matches_wake")
        assert hasattr(wake_mod, "detect_quiet_intent")
        assert hasattr(wake_mod, "WakeWordError")

    def test_wake_detector_instantiates_without_api_key(self) -> None:
        """WakeDetector no longer needs any API key or native library."""
        from apps.channels.voice.wake import WakeDetector

        stt = FakeSTT([])
        vad = FakeVAD()
        # Must not raise — no key needed
        detector = WakeDetector(stt=stt, vad=vad)
        assert detector.keyword == "자기야"
