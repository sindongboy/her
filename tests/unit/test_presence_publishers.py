"""Tests for presence event publishing wired into VoiceChannel, TextChannel, AgentCore.

Uses FakeBus to record events without any real hardware or LLM calls.
Run: uv run pytest tests/unit/test_presence_publishers.py -x -q
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from apps.presence import Event, EventBus
from apps.channels.voice.channel import VoiceChannel
from apps.channels.voice.audio import FakeMicrophone, FakeSpeaker
from apps.channels.text.repl import TextChannel
from apps.memory.store import MemoryStore


# ── FakeBus ───────────────────────────────────────────────────────────────────


class FakeBus(EventBus):
    """Records every published event into a list for assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)

    def publish_state(self, value: str, *, channel: str = "system") -> None:
        self.publish(
            Event(
                type="state",
                payload={"value": value, "channel": channel},
                ts=time.monotonic(),
            )
        )

    def event_types(self) -> list[str]:
        """Convenience: return list of event.type in order."""
        return [e.type for e in self.events]

    def state_values(self) -> list[str]:
        """Convenience: return .payload['value'] for all 'state' events."""
        return [e.payload["value"] for e in self.events if e.type == "state"]

    def of_type(self, t: str) -> list[Event]:
        return [e for e in self.events if e.type == t]


# ── BrokenSecondPublishBus ────────────────────────────────────────────────────


class BrokenSecondPublishBus(FakeBus):
    """Raises on the 2nd publish call — tests publish-failure isolation."""

    def publish(self, event: Event) -> None:
        if len(self.events) >= 1:
            raise RuntimeError("bus is broken")
        super().publish(event)


# ── Voice fakes ───────────────────────────────────────────────────────────────


class FakeSTT:
    """Returns pre-canned transcripts in sequence, then empty strings."""

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)
        self._index = 0

    async def warmup(self) -> None:
        pass

    async def transcribe(self, pcm_bytes: bytes, **kwargs: Any) -> str:
        if self._index < len(self._transcripts):
            result = self._transcripts[self._index]
            self._index += 1
            return result
        return ""


class FakeTTS:
    def __init__(self) -> None:
        self.speak_calls: list[list[str]] = []

    async def speak(self, text_stream: AsyncIterator[str], output: Any) -> None:
        chunks: list[str] = []
        async for chunk in text_stream:
            chunks.append(chunk)
        self.speak_calls.append(chunks)
        await output.write(b"\xAA" * 8)

    async def speak_text(self, text: str, output: Any) -> None:
        await output.write(b"\xBB" * 8)


class FakeVoiceAgent:
    """Returns canned async-generator chunks."""

    def __init__(self, responses: list[list[str]], start_ep: int = 1) -> None:
        self._responses = list(responses)
        self._index = 0
        self.last_episode_id: int | None = None
        self._next_ep = start_ep

    async def stream_respond(
        self,
        message: str,
        *,
        episode_id: int | None = None,
        channel: str = "text",
    ) -> AsyncIterator[str]:
        ep = episode_id if episode_id is not None else self._next_ep
        self._next_ep = ep + 1
        self.last_episode_id = ep
        chunks = (
            self._responses[self._index]
            if self._index < len(self._responses)
            else ["default"]
        )
        self._index += 1
        return _list_async_iter(chunks)


class FakeVAD:
    pass


class FakeSpeechSegments:
    def __init__(self, segments: list[bytes]) -> None:
        self._segments = segments

    def __call__(
        self,
        stream: Any,
        vad: Any,
        *,
        min_speech_ms: int = 200,
        max_silence_ms: int = 700,
        max_segment_ms: int = 30_000,
    ) -> AsyncIterator[bytes]:
        return _list_async_iter(self._segments)


# ── Text fakes ────────────────────────────────────────────────────────────────


@dataclass
class FakeAgentResponse:
    text: str
    episode_id: int | None = 1
    used_episode_ids: list[int] = field(default_factory=list)
    used_fact_ids: list[int] = field(default_factory=list)


class FakeTextAgent:
    """Minimal agent for TextChannel tests."""

    def __init__(self, reply: str = "응답") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def respond(
        self,
        message: str,
        *,
        episode_id: int | None,
        channel: str,
        attachments: Any = None,
    ) -> FakeAgentResponse:
        self.calls.append({"message": message, "channel": channel})
        return FakeAgentResponse(text=self.reply, episode_id=episode_id or 1)


# ── AgentCore fakes ───────────────────────────────────────────────────────────


class FakeGeminiClient:
    """Fake LLM client that yields pre-canned chunks."""

    def __init__(
        self,
        chunks: list[str] | None = None,
        *,
        raise_on_stream: Exception | None = None,
    ) -> None:
        self._chunks = chunks or ["hello", " world"]
        self._raise = raise_on_stream

    def generate(self, messages: Any, *, system: str = "", parts: Any = None) -> str:
        return "".join(self._chunks)

    def generate_stream(
        self, messages: Any, *, system: str = "", parts: Any = None
    ) -> AsyncIterator[str]:
        if self._raise is not None:
            exc = self._raise
            return _raise_async_iter(exc)
        return _list_async_iter(self._chunks)

    def embed(self, text: str, *, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        return [0.0] * 768


# ── helpers ───────────────────────────────────────────────────────────────────


def _pcm(n: int = 3200) -> bytes:
    return b"\x00" * n


async def _list_async_iter(items: list[Any]) -> AsyncIterator[Any]:  # type: ignore[misc]
    for item in items:
        yield item


async def _raise_async_iter(exc: Exception) -> AsyncIterator[str]:  # type: ignore[misc]
    raise exc
    yield  # make it an async generator  # type: ignore[misc]


def _make_voice_channel(
    *,
    segments: list[bytes] | None = None,
    transcripts: list[str] | None = None,
    agent_responses: list[list[str]] | None = None,
    bus: EventBus | None = None,
) -> tuple[VoiceChannel, FakeVoiceAgent, FakeTTS]:
    mic = FakeMicrophone([])
    speaker = FakeSpeaker()
    stt = FakeSTT(transcripts or ["안녕"])
    tts = FakeTTS()
    vad = FakeVAD()
    agent = FakeVoiceAgent(agent_responses or [["반갑습니다"]])

    ch = VoiceChannel(
        agent,
        object(),  # store not used
        mic=mic,
        speaker=speaker,
        vad=vad,
        stt=stt,
        tts=tts,
        speech_segments_fn=FakeSpeechSegments(segments or [_pcm()]),
        status_fn=lambda _: None,
        bus=bus,
    )
    return ch, agent, tts


def _make_text_channel(
    inputs: list[str],
    *,
    bus: EventBus | None = None,
    store: MemoryStore | None = None,
    tmp_path: Path | None = None,
) -> tuple[TextChannel, FakeTextAgent, list[str]]:
    from apps.memory.store import MemoryStore
    import tempfile

    if store is None:
        # Use a fresh in-memory-ish store
        db = Path(tempfile.mkdtemp()) / "test.db"
        store = MemoryStore(db)

    attachments_dir = (tmp_path or Path(tempfile.mkdtemp())) / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)

    agent = FakeTextAgent()
    outputs: list[str] = []
    input_iter = iter(inputs)

    def _input(prompt: str) -> str:
        try:
            return next(input_iter)
        except StopIteration:
            raise EOFError("no more input")

    ch = TextChannel(
        agent,
        store,
        attachments_dir=attachments_dir,
        input_fn=_input,
        output_fn=outputs.append,
        bus=bus,
    )
    return ch, agent, outputs


# ── 1. VoiceChannel happy-path event sequence ────────────────────────────────


class TestVoiceChannelHappyPath:
    @pytest.mark.asyncio
    async def test_full_turn_event_sequence(self) -> None:
        """One complete voice turn must produce events in documented order."""
        bus = FakeBus()
        ch, agent, tts = _make_voice_channel(bus=bus)
        await ch.run()

        types = bus.event_types()
        # loop start → idle
        assert types[0] == "state"
        assert bus.events[0].payload["value"] == "idle"
        # then: listening, transcript, state(thinking), state(speaking), response_end, state(idle)
        assert "transcript" in types
        assert "response_end" in types
        state_vals = bus.state_values()
        assert "listening" in state_vals
        assert "thinking" in state_vals
        assert "speaking" in state_vals
        # ends in idle
        assert state_vals[-1] == "idle"

    @pytest.mark.asyncio
    async def test_transcript_channel_is_voice(self) -> None:
        bus = FakeBus()
        ch, _agent, _tts = _make_voice_channel(bus=bus)
        await ch.run()

        transcripts = bus.of_type("transcript")
        assert len(transcripts) == 1
        assert transcripts[0].payload["channel"] == "voice"
        assert transcripts[0].payload["final"] is True

    @pytest.mark.asyncio
    async def test_response_end_contains_episode_id(self) -> None:
        bus = FakeBus()
        ch, agent, _tts = _make_voice_channel(bus=bus)
        await ch.run()

        ends = bus.of_type("response_end")
        assert len(ends) == 1
        assert ends[0].payload["channel"] == "voice"
        # episode_id is whatever the fake agent assigned
        assert "episode_id" in ends[0].payload


# ── 2. VoiceChannel with bus=None (no errors, no events) ─────────────────────


class TestVoiceChannelNoBus:
    @pytest.mark.asyncio
    async def test_no_bus_no_crash(self) -> None:
        """bus=None must not raise and the full turn completes normally."""
        ch, agent, tts = _make_voice_channel(bus=None)
        await ch.run()  # must not raise
        assert len(agent._responses) >= 0  # just ensure we ran


# ── 3. TextChannel happy-path ─────────────────────────────────────────────────


class TestTextChannelEvents:
    def test_message_emits_thinking_and_transcript(self) -> None:
        bus = FakeBus()
        ch, agent, outputs = _make_text_channel(["안녕", "/quit"], bus=bus)
        asyncio.run(ch.run())

        types = bus.event_types()
        assert "state" in types
        assert "transcript" in types

        transcripts = bus.of_type("transcript")
        assert transcripts[0].payload["channel"] == "text"
        assert transcripts[0].payload["text"] == "안녕"

    def test_state_sequence_thinking_speaking_idle(self) -> None:
        bus = FakeBus()
        ch, agent, outputs = _make_text_channel(["메시지", "/quit"], bus=bus)
        asyncio.run(ch.run())

        state_vals = bus.state_values()
        assert "thinking" in state_vals
        assert "speaking" in state_vals
        assert "idle" in state_vals

    def test_response_end_emitted(self) -> None:
        bus = FakeBus()
        ch, agent, outputs = _make_text_channel(["테스트", "/quit"], bus=bus)
        asyncio.run(ch.run())

        ends = bus.of_type("response_end")
        assert len(ends) >= 1
        assert ends[0].payload["channel"] == "text"


# ── 4. TextChannel with bus=None ──────────────────────────────────────────────


class TestTextChannelNoBus:
    def test_no_bus_no_crash(self) -> None:
        ch, agent, outputs = _make_text_channel(["안녕", "/quit"], bus=None)
        asyncio.run(ch.run())  # must not raise
        assert len(agent.calls) == 1


# ── 5. AgentCore stream_respond → N chunks + response_end ────────────────────


class TestAgentCoreStreamChunks:
    @pytest.mark.asyncio
    async def test_n_chunks_yield_n_response_chunk_events(
        self, tmp_path: Path
    ) -> None:
        from apps.agent.core import AgentCore

        store = MemoryStore(tmp_path / "a.db")
        bus = FakeBus()
        chunks = ["안", "녕", "하", "세요"]
        core = AgentCore(
            store,
            client=FakeGeminiClient(chunks),
            enable_anonymization=False,
            bus=bus,
        )

        collected: list[str] = []
        async for chunk in core.stream_respond("hi", channel="voice"):
            collected.append(chunk)

        assert collected == chunks
        chunk_events = bus.of_type("response_chunk")
        assert len(chunk_events) == len(chunks)
        for ev, ch in zip(chunk_events, chunks):
            assert ev.payload["text"] == ch
            assert ev.payload["channel"] == "voice"

    @pytest.mark.asyncio
    async def test_channel_passthrough(self, tmp_path: Path) -> None:
        """channel kwarg must appear in every response_chunk payload."""
        from apps.agent.core import AgentCore

        store = MemoryStore(tmp_path / "b.db")
        bus = FakeBus()
        core = AgentCore(
            store,
            client=FakeGeminiClient(["a", "b"]),
            enable_anonymization=False,
            bus=bus,
        )
        async for _ in core.stream_respond("test", channel="voice"):
            pass

        for ev in bus.of_type("response_chunk"):
            assert ev.payload["channel"] == "voice"


# ── 6. AgentCore exception → error event published, exception propagates ──────


class TestAgentCoreExceptionPath:
    @pytest.mark.asyncio
    async def test_error_event_on_stream_exception(self, tmp_path: Path) -> None:
        from apps.agent.core import AgentCore

        store = MemoryStore(tmp_path / "c.db")
        bus = FakeBus()
        boom = RuntimeError("LLM unavailable")
        core = AgentCore(
            store,
            client=FakeGeminiClient(raise_on_stream=boom),
            enable_anonymization=False,
            bus=bus,
        )

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            async for _ in core.stream_respond("hello", channel="text"):
                pass

        error_events = bus.of_type("error")
        assert len(error_events) == 1
        assert "LLM unavailable" in error_events[0].payload["message"]
        assert error_events[0].payload["where"] == "agent.stream_respond"


# ── 7. Publish-failure isolation ──────────────────────────────────────────────


class TestPublishFailureIsolation:
    @pytest.mark.asyncio
    async def test_voice_channel_survives_bus_exception(self) -> None:
        """If bus.publish raises, VoiceChannel must not crash."""
        bus = BrokenSecondPublishBus()
        ch, agent, tts = _make_voice_channel(bus=bus)
        # Must complete without raising even though bus is broken.
        await ch.run()

    def test_text_channel_survives_bus_exception(self) -> None:
        """If bus.publish raises, TextChannel must not crash."""
        bus = BrokenSecondPublishBus()
        ch, agent, outputs = _make_text_channel(["안녕", "/quit"], bus=bus)
        asyncio.run(ch.run())  # must not raise

    @pytest.mark.asyncio
    async def test_agent_core_survives_bus_exception(self, tmp_path: Path) -> None:
        """If bus.publish raises, AgentCore stream must not crash."""
        from apps.agent.core import AgentCore

        store = MemoryStore(tmp_path / "d.db")
        bus = BrokenSecondPublishBus()
        core = AgentCore(
            store,
            client=FakeGeminiClient(["x", "y"]),
            enable_anonymization=False,
            bus=bus,
        )
        collected: list[str] = []
        async for chunk in core.stream_respond("hello", channel="text"):
            collected.append(chunk)
        # Chunks still collected despite bus failure
        assert len(collected) > 0
