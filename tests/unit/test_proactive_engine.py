"""Tests for ProactiveEngine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from apps.proactive.engine import ProactiveConfig, ProactiveEngine
from apps.proactive.triggers import TriggerProposal


# ── Test doubles ───────────────────────────────────────────────────────────────


@dataclass
class _FakeResponse:
    text: str
    episode_id: int = 0
    used_episode_ids: list = None  # type: ignore
    used_fact_ids: list = None  # type: ignore

    def __post_init__(self) -> None:
        if self.used_episode_ids is None:
            self.used_episode_ids = []
        if self.used_fact_ids is None:
            self.used_fact_ids = []


class _FakeAgent:
    """Agent that returns a fixed utterance without calling LLM."""

    def __init__(self, response_text: str = "테스트 발화입니다.") -> None:
        self.response_text = response_text
        self.call_count = 0

    async def respond(
        self,
        message: str,
        *,
        episode_id: object = None,
        channel: str = "voice",
        attachments: object = None,
    ) -> _FakeResponse:
        self.call_count += 1
        return _FakeResponse(text=self.response_text)


class _FakeChannel:
    """Channel that captures say() calls without audio output."""

    def __init__(self) -> None:
        self.said: list[str] = []

    async def say(self, text: str) -> None:
        self.said.append(text)


class _AlwaysProposalTrigger:
    """Trigger that always returns a proposal."""

    def __init__(self, dedup_key: str = "test:always", priority: int = 50) -> None:
        self._dedup_key = dedup_key
        self._priority = priority

    async def evaluate(
        self, store: Any, *, now: datetime
    ) -> list[TriggerProposal]:
        return [
            TriggerProposal(
                source="test",
                priority=self._priority,
                context="테스트 컨텍스트입니다.",
                dedup_key=self._dedup_key,
            )
        ]


class _NeverProposalTrigger:
    """Trigger that never returns proposals."""

    async def evaluate(
        self, store: Any, *, now: datetime
    ) -> list[TriggerProposal]:
        return []


class _FailingTrigger:
    """Trigger that raises on evaluate."""

    async def evaluate(
        self, store: Any, *, now: datetime
    ) -> list[TriggerProposal]:
        raise RuntimeError("trigger exploded")


def _make_store(tmp_path: Path) -> Any:
    from apps.memory.store import MemoryStore

    return MemoryStore(tmp_path / "db.sqlite")


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── Basic fire logic ───────────────────────────────────────────────────────────


class TestBasicFire:
    def test_fires_on_first_tick(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            text_channel=channel,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=0),
            triggers=[_AlwaysProposalTrigger()],
        )
        result = asyncio.run(engine.trigger_once())
        assert result is True
        assert len(channel.said) == 1
        store.close()

    def test_increments_daily_count(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            text_channel=channel,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=0),
            triggers=[_AlwaysProposalTrigger(dedup_key="unique:1")],
        )
        now = _now()
        asyncio.run(engine.trigger_once(now=now))
        assert engine._count_today_fires(now) == 1
        store.close()

    def test_no_channel_no_crash(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=0),
            triggers=[_AlwaysProposalTrigger()],
        )
        # Should not raise even with no channels
        result = asyncio.run(engine.trigger_once())
        assert isinstance(result, bool)
        store.close()


# ── Daily limit ────────────────────────────────────────────────────────────────


class TestDailyLimit:
    def test_stops_at_daily_limit(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()
        now = _now()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            text_channel=channel,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=0),
            triggers=[],  # we'll use unique triggers per call
        )

        # Fire 3 times with distinct dedup keys
        for i in range(3):
            engine._triggers = [_AlwaysProposalTrigger(dedup_key=f"daily:{i}")]
            fired = asyncio.run(engine.trigger_once(now=now))
            assert fired is True

        # 4th should be blocked
        engine._triggers = [_AlwaysProposalTrigger(dedup_key="daily:4")]
        fired = asyncio.run(engine.trigger_once(now=now))
        assert fired is False
        assert len(channel.said) == 3
        store.close()

    def test_different_day_resets_limit(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            text_channel=channel,  # type: ignore
            config=ProactiveConfig(daily_limit=1, cooldown_minutes=0),
            triggers=[],
        )

        today = _now()
        engine._triggers = [_AlwaysProposalTrigger(dedup_key="day1:event")]
        asyncio.run(engine.trigger_once(now=today))

        # Simulate next day
        tomorrow = today + timedelta(days=1)
        engine._triggers = [_AlwaysProposalTrigger(dedup_key="day2:event")]
        fired = asyncio.run(engine.trigger_once(now=tomorrow))
        assert fired is True
        store.close()


# ── Cooldown ───────────────────────────────────────────────────────────────────


class TestCooldown:
    def test_cooldown_blocks_second_tick(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()
        now = _now()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            text_channel=channel,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=60),
            triggers=[],
        )

        engine._triggers = [_AlwaysProposalTrigger(dedup_key="cool:1")]
        asyncio.run(engine.trigger_once(now=now))

        # 5 minutes later — still in cooldown
        five_min_later = now + timedelta(minutes=5)
        engine._triggers = [_AlwaysProposalTrigger(dedup_key="cool:2")]
        fired = asyncio.run(engine.trigger_once(now=five_min_later))
        assert fired is False
        store.close()

    def test_after_cooldown_fires_again(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()
        now = _now()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            text_channel=channel,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=60),
            triggers=[],
        )

        engine._triggers = [_AlwaysProposalTrigger(dedup_key="cool:A")]
        asyncio.run(engine.trigger_once(now=now))

        # 65 minutes later — cooldown expired
        post_cooldown = now + timedelta(minutes=65)
        engine._triggers = [_AlwaysProposalTrigger(dedup_key="cool:B")]
        fired = asyncio.run(engine.trigger_once(now=post_cooldown))
        assert fired is True
        assert len(channel.said) == 2
        store.close()


# ── Quiet mode ─────────────────────────────────────────────────────────────────


class TestQuietMode:
    def test_quiet_mode_blocks_all_fires(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            text_channel=channel,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=0, quiet_mode=True),
            triggers=[_AlwaysProposalTrigger()],
        )
        result = asyncio.run(engine.trigger_once())
        assert result is False
        assert len(channel.said) == 0
        store.close()

    def test_disable_quiet_mode_allows_fire(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            text_channel=channel,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=0, quiet_mode=True),
            triggers=[_AlwaysProposalTrigger()],
        )
        engine.set_quiet(False)
        result = asyncio.run(engine.trigger_once())
        assert result is True
        store.close()


# ── Deduplication ──────────────────────────────────────────────────────────────


class TestDedup:
    def test_same_key_does_not_fire_twice_within_6h(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()
        now = _now()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            text_channel=channel,  # type: ignore
            config=ProactiveConfig(daily_limit=10, cooldown_minutes=0),
            triggers=[_AlwaysProposalTrigger(dedup_key="dedup:same")],
        )

        asyncio.run(engine.trigger_once(now=now))
        # 30 min later — same key, still within 6h
        later = now + timedelta(minutes=30)
        result = asyncio.run(engine.trigger_once(now=later))
        assert result is False
        assert len(channel.said) == 1
        store.close()

    def test_same_key_fires_after_6h(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()
        now = _now()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            text_channel=channel,  # type: ignore
            config=ProactiveConfig(daily_limit=10, cooldown_minutes=0),
            triggers=[_AlwaysProposalTrigger(dedup_key="dedup:after6h")],
        )

        asyncio.run(engine.trigger_once(now=now))
        # 7 hours later — dedup window expired
        after_window = now + timedelta(hours=7)
        result = asyncio.run(engine.trigger_once(now=after_window))
        assert result is True
        assert len(channel.said) == 2
        store.close()


# ── Channel preference ─────────────────────────────────────────────────────────


class TestChannelPreference:
    def test_prefers_voice_when_both_available(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        voice = _FakeChannel()
        text = _FakeChannel()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            voice_channel=voice,  # type: ignore
            text_channel=text,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=0),
            triggers=[_AlwaysProposalTrigger()],
        )
        asyncio.run(engine.trigger_once())
        assert len(voice.said) == 1
        assert len(text.said) == 0
        store.close()

    def test_falls_back_to_text_when_no_voice(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        text = _FakeChannel()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            voice_channel=None,
            text_channel=text,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=0),
            triggers=[_AlwaysProposalTrigger()],
        )
        asyncio.run(engine.trigger_once())
        assert len(text.said) == 1
        store.close()


# ── Bus events ─────────────────────────────────────────────────────────────────


class TestBusEvents:
    def test_emits_state_events_on_fire(self, tmp_path: Path) -> None:
        from apps.presence import EventBus

        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()
        bus = EventBus()

        # Collect events synchronously via a subscriber queue attached before firing.
        async def _run() -> list:
            # Subscribe first (before firing) so events are enqueued.
            q: asyncio.Queue = asyncio.Queue()
            received_local: list = []

            async def _collect() -> None:
                async for evt in bus.subscribe():
                    received_local.append(evt)

            collect_task = asyncio.create_task(_collect())
            # Yield control so the subscriber coroutine can start and register.
            await asyncio.sleep(0)

            engine = ProactiveEngine(
                store,
                agent,  # type: ignore
                text_channel=channel,  # type: ignore
                bus=bus,
                config=ProactiveConfig(daily_limit=3, cooldown_minutes=0),
                triggers=[_AlwaysProposalTrigger()],
            )
            await engine.trigger_once()
            # Give the collect task a chance to process queued events.
            await asyncio.sleep(0)
            await bus.close()
            try:
                await asyncio.wait_for(collect_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            return received_local

        received = asyncio.run(_run())

        state_types = [e.type for e in received]
        assert "state" in state_types
        store.close()


# ── Failing trigger ────────────────────────────────────────────────────────────


class TestFailingTrigger:
    def test_failing_trigger_does_not_crash_engine(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()
        channel = _FakeChannel()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            text_channel=channel,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=0),
            triggers=[_FailingTrigger(), _AlwaysProposalTrigger()],
        )
        result = asyncio.run(engine.trigger_once())
        # AlwaysProposalTrigger should still fire despite FailingTrigger
        assert result is True
        store.close()


# ── run() loop ────────────────────────────────────────────────────────────────


class TestRunLoop:
    def test_run_stops_on_stop_event(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()

        engine = ProactiveEngine(
            store,
            agent,  # type: ignore
            config=ProactiveConfig(daily_limit=3, cooldown_minutes=0, tick_interval_s=0.05),
            triggers=[_NeverProposalTrigger()],
        )

        async def _run() -> None:
            stop = asyncio.Event()
            # Stop after a brief delay
            asyncio.get_event_loop().call_later(0.1, stop.set)
            await engine.run(stop_event=stop)

        asyncio.run(_run())  # should return cleanly
        store.close()

    def test_stop_method_sets_stopped_flag(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        agent = _FakeAgent()

        engine = ProactiveEngine(store, agent)  # type: ignore
        asyncio.run(engine.stop())
        assert engine._stopped is True
        store.close()
