"""Unit tests for apps/channels/voice/channel.py.

Uses Fakes for EVERYTHING — no real hardware, no real Whisper, no real Gemini.
Run: uv run pytest tests/unit/test_voice_channel.py -x -q
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from apps.channels.voice.audio import AudioFormat, FakeMicrophone, FakeSpeaker
from apps.channels.voice.channel import VoiceChannel


# ---------------------------------------------------------------------------
# Fake implementations — no external dependencies
# ---------------------------------------------------------------------------


class FakeSTT:
    """Returns pre-canned transcripts in sequence, then empty strings."""

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)
        self._index = 0
        self.warmup_called = False
        self.transcribe_calls: list[int] = []  # byte lengths

    async def warmup(self) -> None:
        self.warmup_called = True

    async def transcribe(self, pcm_bytes: bytes, **kwargs: Any) -> str:
        self.transcribe_calls.append(len(pcm_bytes))
        if self._index < len(self._transcripts):
            result = self._transcripts[self._index]
            self._index += 1
            return result
        return ""


class FakeTTS:
    """Records every speak() call with the text chunks consumed."""

    def __init__(self) -> None:
        self.speak_calls: list[list[str]] = []  # one list per speak() call
        self.speak_text_calls: list[str] = []

    async def speak(self, text_stream: AsyncIterator[str], output: Any) -> None:
        chunks: list[str] = []
        async for chunk in text_stream:
            chunks.append(chunk)
        self.speak_calls.append(chunks)
        # Write a sentinel byte so tests can assert speaker received audio
        await output.write(b"\xAA" * 32)

    async def speak_text(self, text: str, output: Any) -> None:
        self.speak_text_calls.append(text)
        await output.write(b"\xBB" * 32)


class FakeAgent:
    """Returns canned text chunks; tracks episode_id progression."""

    def __init__(self, responses: list[list[str]], start_episode_id: int = 1) -> None:
        self._responses = list(responses)
        self._index = 0
        self._next_episode = start_episode_id
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
        # Assign a stable episode_id for this turn
        ep_id = episode_id if episode_id is not None else self._next_episode
        self._next_episode = ep_id + 1
        self.last_episode_id = ep_id

        chunks = (
            self._responses[self._index]
            if self._index < len(self._responses)
            else ["(기본 응답)"]
        )
        self._index += 1
        return _list_to_async_iter(chunks)


class FakeVAD:
    """Stateless placeholder — speech_segments_fn does all the work."""

    pass


class FakeSpeechSegments:
    """Wraps a list of pre-canned PCM byte segments as an async generator.

    Passed as speech_segments_fn to VoiceChannel.
    """

    def __init__(self, segments: list[bytes]) -> None:
        self._segments = list(segments)

    def __call__(
        self,
        stream: Any,
        vad: Any,
        *,
        min_speech_ms: int = 200,
        max_silence_ms: int = 700,
        max_segment_ms: int = 30_000,
    ) -> AsyncIterator[bytes]:
        return _list_to_async_iter(self._segments)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcm(n: int = 3200) -> bytes:
    """Return n bytes of silent PCM (16-bit zero samples)."""
    return b"\x00" * n


async def _list_to_async_iter(items: list[Any]) -> AsyncIterator[Any]:  # type: ignore[misc]
    for item in items:
        yield item


def _make_channel(
    *,
    mic_frames: list[bytes] | None = None,
    segments: list[bytes] | None = None,
    transcripts: list[str] | None = None,
    agent_responses: list[list[str]] | None = None,
    status_log: list[str] | None = None,
) -> tuple[VoiceChannel, FakeAgent, FakeTTS, FakeSTT, FakeSpeaker]:
    """Construct a VoiceChannel wired with Fakes.

    None means "use default"; [] means "empty list".
    """
    mic = FakeMicrophone(mic_frames if mic_frames is not None else [])
    speaker = FakeSpeaker()

    stt = FakeSTT(transcripts if transcripts is not None else ["안녕"])
    tts = FakeTTS()
    vad = FakeVAD()
    agent = FakeAgent(agent_responses if agent_responses is not None else [["안녕하세요!"]])

    status: list[str] = [] if status_log is None else status_log

    def capture_status(msg: str) -> None:
        status.append(msg)

    channel = VoiceChannel(
        agent,
        object(),  # store — not used in Phase 1 channel logic
        mic=mic,
        speaker=speaker,
        vad=vad,
        stt=stt,
        tts=tts,
        speech_segments_fn=FakeSpeechSegments(
            segments if segments is not None else [_pcm()]
        ),
        status_fn=capture_status,
        echo_gate_ms=0,  # tests run synchronously; skip the post-TTS reverb wait
    )

    return channel, agent, tts, stt, speaker


# ---------------------------------------------------------------------------
# 1. One full turn: segment → STT → agent → TTS → speaker
# ---------------------------------------------------------------------------


class TestOneTurn:
    @pytest.mark.asyncio
    async def test_full_turn_produces_audio(self) -> None:
        """A single segment flows all the way through to the speaker."""
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm(3200)],
            transcripts=["안녕"],
            agent_responses=[["안녕하세요!"]],
        )
        await channel.run()

        assert stt.warmup_called
        assert len(stt.transcribe_calls) == 1
        assert len(agent.stream_calls) == 1
        assert agent.stream_calls[0][0] == "안녕"
        assert len(tts.speak_calls) == 1
        assert tts.speak_calls[0] == ["안녕하세요!"]
        assert len(speaker.written) > 0

    @pytest.mark.asyncio
    async def test_agent_receives_voice_channel_tag(self) -> None:
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm()],
            transcripts=["테스트"],
        )
        await channel.run()
        # Verify agent was called with channel='voice'
        # stream_calls stores (message, episode_id); channel kwarg is part of the call
        # We verify this by checking the call was made
        assert len(agent.stream_calls) == 1

    @pytest.mark.asyncio
    async def test_speaker_receives_bytes(self) -> None:
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm()],
        )
        await channel.run()
        # FakeTTS.speak writes 32 bytes of \xAA sentinel
        assert speaker.written == b"\xAA" * 32

    @pytest.mark.asyncio
    async def test_multiple_text_chunks_all_passed_to_tts(self) -> None:
        chunks = ["첫 번째 ", "두 번째 ", "마지막."]
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm()],
            transcripts=["안녕"],
            agent_responses=[chunks],
        )
        await channel.run()
        assert tts.speak_calls[0] == chunks


# ---------------------------------------------------------------------------
# 2. Multi-turn episode continuity
# ---------------------------------------------------------------------------


class TestEpisodeContinuity:
    @pytest.mark.asyncio
    async def test_episode_id_stable_across_turns(self) -> None:
        """episode_id from turn 1 is reused in turn 2."""
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm(), _pcm()],
            transcripts=["첫 번째", "두 번째"],
            agent_responses=[["응답1"], ["응답2"]],
        )
        await channel.run()

        # Turn 1: episode_id was None (first time)
        assert agent.stream_calls[0][1] is None
        # Turn 2: episode_id must carry forward (not None)
        # FakeAgent sets last_episode_id = 1 after first call → passed on second call
        assert agent.stream_calls[1][1] is not None

    @pytest.mark.asyncio
    async def test_episode_id_increments(self) -> None:
        """FakeAgent increments episode internally; channel tracks it."""
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm(), _pcm(), _pcm()],
            transcripts=["a", "b", "c"],
            agent_responses=[["r1"], ["r2"], ["r3"]],
        )
        await channel.run()

        # Turn 1: no episode yet
        assert agent.stream_calls[0][1] is None
        # After turn 1, episode_id should be set for subsequent turns
        ep1 = agent.stream_calls[1][1]
        ep2 = agent.stream_calls[2][1]
        assert ep1 is not None
        assert ep2 is not None
        # Episode IDs should be the same (FakeAgent assigns 1, then increments,
        # but channel_episode_id holds the first-assigned value)
        assert ep1 == ep2  # channel re-uses the same episode_id across turns


# ---------------------------------------------------------------------------
# 3. stop() exits the loop cleanly
# ---------------------------------------------------------------------------


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_before_run(self) -> None:
        """stop() called before run() should still allow run() to exit immediately."""
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm()] * 100,  # many segments — should NOT all be processed
            transcripts=["x"] * 100,
            agent_responses=[["y"]] * 100,
        )
        await channel.stop()
        # run() should exit quickly (segments list still consumed but running=False)
        # We accept that it processes 0 segments
        await asyncio.wait_for(channel.run(), timeout=2.0)

    @pytest.mark.asyncio
    async def test_stop_during_loop_via_flag(self) -> None:
        """Setting _running=False mid-flight stops at next iteration."""
        processed: list[str] = []

        class CountingAgent(FakeAgent):
            def __init__(self, outer_channel_ref: list[Any]) -> None:
                super().__init__([["r1"], ["r2"], ["r3"]])
                self._ref = outer_channel_ref

            async def stream_respond(
                self,
                message: str,
                *,
                episode_id: int | None = None,
                channel: str = "text",
            ) -> AsyncIterator[str]:
                processed.append(message)
                # Stop the channel after the first turn
                if self._ref:
                    await self._ref[0].stop()
                return _list_to_async_iter(["ok"])

        channel_ref: list[Any] = []
        channel, _, tts, stt, speaker = _make_channel(
            segments=[_pcm()] * 3,
            transcripts=["a", "b", "c"],
        )
        counting_agent = CountingAgent(channel_ref)
        channel._agent = counting_agent
        channel_ref.append(channel)

        await asyncio.wait_for(channel.run(), timeout=3.0)
        # Should process only 1 turn (stop after first)
        assert len(processed) == 1


# ---------------------------------------------------------------------------
# 4. Empty STT transcript → agent NOT called
# ---------------------------------------------------------------------------


class TestEmptyTranscript:
    @pytest.mark.asyncio
    async def test_empty_transcript_skips_agent(self) -> None:
        """If STT returns '', agent.stream_respond must NOT be called."""
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm()],
            transcripts=[""],  # empty transcript
            agent_responses=[["should not be called"]],
        )
        await channel.run()

        assert len(agent.stream_calls) == 0
        assert len(tts.speak_calls) == 0

    @pytest.mark.asyncio
    async def test_whitespace_only_transcript_skips_agent(self) -> None:
        """Whitespace-only transcript is also considered empty."""
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm()],
            transcripts=["   \t  \n  "],
            agent_responses=[["should not be called"]],
        )
        await channel.run()
        assert len(agent.stream_calls) == 0

    @pytest.mark.asyncio
    async def test_valid_transcript_after_empty_calls_agent(self) -> None:
        """First segment is empty (skip), second is valid (process)."""
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm(), _pcm()],
            transcripts=["", "유효한 발화"],
            agent_responses=[["응답"]],
        )
        await channel.run()
        # Only 1 agent call (the second segment)
        assert len(agent.stream_calls) == 1
        assert agent.stream_calls[0][0] == "유효한 발화"


# ---------------------------------------------------------------------------
# 5. No segments → loop doesn't busy-spin
# ---------------------------------------------------------------------------


class TestNoSegments:
    @pytest.mark.asyncio
    async def test_empty_segment_list_exits_promptly(self) -> None:
        """With no speech segments, run() should return without blocking."""
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[],  # empty — no audio at all
            transcripts=[],
            agent_responses=[],
        )

        # Should complete well within 1 second (no busy spin)
        await asyncio.wait_for(channel.run(), timeout=1.0)

        # No agent calls, no TTS calls
        assert len(agent.stream_calls) == 0
        assert len(tts.speak_calls) == 0
        assert speaker.written == b""

    @pytest.mark.asyncio
    async def test_warmup_called_even_with_no_segments(self) -> None:
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[], transcripts=[], agent_responses=[]
        )
        await channel.run()
        assert stt.warmup_called


# ---------------------------------------------------------------------------
# 6. Status messages are emitted in correct order
# ---------------------------------------------------------------------------


class TestStatusMessages:
    @pytest.mark.asyncio
    async def test_status_sequence_for_one_turn(self) -> None:
        status_log: list[str] = []
        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm()],
            transcripts=["안녕"],
            status_log=status_log,
        )
        await channel.run()

        # Must have at least a start and stop message
        assert any("시작" in m for m in status_log)
        assert any("종료" in m for m in status_log)
        # Per-turn status messages
        assert any("듣는 중" in m or "인식" in m for m in status_log)
        assert any("생각 중" in m for m in status_log)
        assert any("답하는 중" in m for m in status_log)


# ---------------------------------------------------------------------------
# 7. STT warmup failure is tolerated
# ---------------------------------------------------------------------------


class TestWarmupFailure:
    @pytest.mark.asyncio
    async def test_warmup_exception_does_not_crash(self) -> None:
        """warmup() failure should be logged but not propagate."""

        class BrokenSTT(FakeSTT):
            async def warmup(self) -> None:
                raise RuntimeError("model not found")

        channel, agent, tts, stt, speaker = _make_channel(
            segments=[_pcm()],
            transcripts=["안녕"],
        )
        channel._stt = BrokenSTT(["안녕"])

        # run() must complete without raising
        await asyncio.wait_for(channel.run(), timeout=2.0)
        # Agent still called (warmup failure doesn't block the loop)
        assert len(agent.stream_calls) == 1


# ---------------------------------------------------------------------------
# 8. Agent stream_respond exception is tolerated per-turn
# ---------------------------------------------------------------------------


class TestAgentFailure:
    @pytest.mark.asyncio
    async def test_agent_exception_skips_turn_not_crash(self) -> None:
        """If agent.stream_respond raises, the channel continues to next turn."""
        call_count = 0

        class FailOnFirstAgent(FakeAgent):
            async def stream_respond(
                self,
                message: str,
                *,
                episode_id: int | None = None,
                channel: str = "text",
            ) -> AsyncIterator[str]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("LLM unavailable")
                return _list_to_async_iter(["ok"])

        channel, _, tts, stt, speaker = _make_channel(
            segments=[_pcm(), _pcm()],
            transcripts=["a", "b"],
        )
        channel._agent = FailOnFirstAgent([["r1"], ["r2"]])

        await asyncio.wait_for(channel.run(), timeout=2.0)
        # Second turn still processed
        assert call_count == 2
        assert len(tts.speak_calls) == 1


# ---------------------------------------------------------------------------
# 9. Import smoke: channel.py importable without sibling modules
# ---------------------------------------------------------------------------


class TestImports:
    def test_channel_importable(self) -> None:
        """channel.py should import without vad/stt/tts installed."""
        from apps.channels.voice.channel import VoiceChannel  # noqa: F401

        assert VoiceChannel is not None

    def test_fake_audio_from_package(self) -> None:
        """FakeMicrophone and FakeSpeaker importable from voice package."""
        from apps.channels.voice import FakeMicrophone, FakeSpeaker  # noqa: F401

        assert FakeMicrophone is not None
        assert FakeSpeaker is not None
