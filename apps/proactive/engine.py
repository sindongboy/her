"""ProactiveEngine — drives proactive utterances from triggers.

CLAUDE.md §2.4: fire at most N times/day, respect quiet mode, use last-active
channel, emit state events to orb bus.

Loop (every tick_interval_s, default 60s):
  1. If quiet_mode → skip.
  2. Evaluate all triggers concurrently.
  3. Sort proposals by priority desc.
  4. Dedupe (same dedup_key in last 6h → drop).
  5. Daily limit check.
  6. Cooldown check.
  7. Pick top proposal, generate utterance via agent, deliver via channel.
  8. Persist fire-record.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from apps.presence import Event, EventBus
from apps.proactive.triggers import Trigger, TriggerProposal, TimeBasedTrigger, SilenceTrigger, RecurringPatternTrigger

if TYPE_CHECKING:
    from apps.memory.store import MemoryStore
    from apps.agent.core import AgentCore
    from apps.channels.voice.channel import VoiceChannel
    from apps.channels.text.repl import TextChannel

log = structlog.get_logger(__name__)

_DEDUP_WINDOW_HOURS = 6.0


# ── safe publish helper ───────────────────────────────────────────────────────


def _publish(bus: EventBus | None, event: Event) -> None:
    if bus is None:
        return
    try:
        bus.publish(event)
    except Exception as exc:  # pragma: no cover
        log.warning("publish_failed", error=str(exc))


# ── ProactiveConfig ───────────────────────────────────────────────────────────


@dataclass
class ProactiveConfig:
    daily_limit: int = 3
    silence_threshold_hours: float = 4.0
    cooldown_minutes: int = 60
    quiet_mode: bool = False
    tick_interval_s: float = 60.0


# ── FireRecord (in-memory persistence) ───────────────────────────────────────


@dataclass
class _FireRecord:
    dedup_key: str
    fired_at: datetime  # UTC
    proposal: TriggerProposal


# ── ProactiveEngine ───────────────────────────────────────────────────────────


class ProactiveEngine:
    """Evaluates triggers and delivers proactive utterances.

    Parameters
    ----------
    store:
        Shared MemoryStore.
    agent:
        AgentCore (or compatible duck-type with .respond()).
    voice_channel:
        Optional VoiceChannel that has a .say(text) method.
    text_channel:
        Optional TextChannel that has a .say(text) method.
    bus:
        Optional EventBus for publishing proactive state events.
    config:
        ProactiveConfig controlling limits and timing.
    triggers:
        List of Trigger instances. Defaults to [TimeBasedTrigger(),
        SilenceTrigger(config.silence_threshold_hours),
        RecurringPatternTrigger()].
    _clock:
        Injected callable returning current datetime (UTC). Tests use this
        to control time without sleeping.
    """

    def __init__(
        self,
        store: "MemoryStore",
        agent: "AgentCore",
        *,
        voice_channel: "VoiceChannel | None" = None,
        text_channel: "TextChannel | None" = None,
        bus: EventBus | None = None,
        config: ProactiveConfig | None = None,
        triggers: list[Trigger] | None = None,
        _clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._agent = agent
        self._voice_channel = voice_channel
        self._text_channel = text_channel
        self._bus = bus
        self._config = config or ProactiveConfig()
        self._clock = _clock or (lambda: datetime.now(tz=timezone.utc))

        if triggers is None:
            self._triggers: list[Trigger] = [
                TimeBasedTrigger(),
                SilenceTrigger(threshold_hours=self._config.silence_threshold_hours),
                RecurringPatternTrigger(),
            ]
        else:
            self._triggers = list(triggers)

        # Fire history: used for dedup + daily limit + cooldown
        self._fire_records: list[_FireRecord] = []
        self._stopped = False

    # ── public API ────────────────────────────────────────────────────────────

    async def run(self, *, stop_event: asyncio.Event) -> None:
        """Main loop: tick every tick_interval_s until stop_event is set."""
        log.info("proactive_engine.started", config=self._config)
        while not stop_event.is_set() and not self._stopped:
            try:
                await self.trigger_once()
            except Exception as exc:
                log.error("proactive_engine.tick_error", error=str(exc))

            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=self._config.tick_interval_s
                )
            except asyncio.TimeoutError:
                pass  # normal — keep looping

        log.info("proactive_engine.stopped")

    async def trigger_once(self, *, now: datetime | None = None) -> bool:
        """Single evaluation tick.

        Returns True if a proactive utterance was delivered, False otherwise.
        """
        now = now or self._clock()

        # 1. Quiet mode check
        if self._config.quiet_mode:
            log.debug("proactive_engine.skip_quiet")
            return False

        # 2. Evaluate all triggers concurrently
        tasks = [t.evaluate(self._store, now=now) for t in self._triggers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        proposals: list[TriggerProposal] = []
        for result in results:
            if isinstance(result, Exception):
                log.warning("proactive_engine.trigger_error", error=str(result))
                continue
            proposals.extend(result)

        if not proposals:
            return False

        # 3. Sort by priority descending
        proposals.sort(key=lambda p: p.priority, reverse=True)

        # 4. Dedupe
        proposals = self._filter_dedup(proposals, now)

        if not proposals:
            return False

        # 5. Daily limit check
        today_count = self._count_today_fires(now)
        if today_count >= self._config.daily_limit:
            log.debug(
                "proactive_engine.daily_limit_reached",
                count=today_count,
                limit=self._config.daily_limit,
            )
            return False

        # 6. Cooldown check
        if self._in_cooldown(now):
            log.debug("proactive_engine.cooldown_active")
            return False

        # 7. Pick top proposal, generate utterance, deliver
        top = proposals[0]
        delivered = await self._deliver(top, now)
        return delivered

    def set_quiet(self, on: bool) -> None:
        """Toggle quiet mode."""
        self._config.quiet_mode = on
        log.info("proactive_engine.quiet_mode_set", is_quiet=on)

    async def stop(self) -> None:
        """Signal the engine to stop on next tick."""
        self._stopped = True
        log.info("proactive_engine.stop_requested")

    # ── private helpers ───────────────────────────────────────────────────────

    def _filter_dedup(
        self, proposals: list[TriggerProposal], now: datetime
    ) -> list[TriggerProposal]:
        """Drop proposals whose dedup_key fired within DEDUP_WINDOW_HOURS."""
        window_secs = _DEDUP_WINDOW_HOURS * 3600
        recent_keys: set[str] = set()
        for rec in self._fire_records:
            elapsed = (now - rec.fired_at).total_seconds()
            if elapsed < window_secs:
                recent_keys.add(rec.dedup_key)

        return [p for p in proposals if p.dedup_key not in recent_keys]

    def _count_today_fires(self, now: datetime) -> int:
        """Count fires that happened today (UTC date)."""
        today = now.date()
        return sum(
            1 for rec in self._fire_records if rec.fired_at.date() == today
        )

    def _in_cooldown(self, now: datetime) -> bool:
        """Return True if a fire happened within cooldown_minutes."""
        if not self._fire_records:
            return False
        last_fire = max(self._fire_records, key=lambda r: r.fired_at)
        elapsed_mins = (now - last_fire.fired_at).total_seconds() / 60.0
        return elapsed_mins < self._config.cooldown_minutes

    async def _deliver(self, proposal: TriggerProposal, now: datetime) -> bool:
        """Generate utterance via agent and deliver to the preferred channel."""
        # Emit proactive state to bus
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "thinking", "channel": "system"},
                ts=time.monotonic(),
            ),
        )

        # Generate utterance via agent (no-llm path: use context as-is)
        utterance = await self._generate_utterance(proposal)
        if not utterance:
            log.warning("proactive_engine.empty_utterance", source=proposal.source)
            return False

        # Pick delivery channel
        channel_name = self._pick_channel()
        log.info(
            "proactive_engine.delivering",
            source=proposal.source,
            channel=channel_name,
        )

        # Emit speaking state before delivery
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "speaking", "channel": channel_name},
                ts=time.monotonic(),
            ),
        )

        await self._say(utterance, channel_name)

        # Emit idle state after delivery
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "idle", "channel": channel_name},
                ts=time.monotonic(),
            ),
        )

        # Record fire
        self._fire_records.append(
            _FireRecord(dedup_key=proposal.dedup_key, fired_at=now, proposal=proposal)
        )

        log.info("proactive_engine.fired", source=proposal.source, channel=channel_name)
        return True

    async def _generate_utterance(self, proposal: TriggerProposal) -> str:
        """Call agent to turn proposal context into a natural Korean utterance."""
        try:
            response = await self._agent.respond(
                proposal.context,
                channel="voice",  # voice-style: no markdown, natural speech
                episode_id=None,
            )
            text: str = response.text if hasattr(response, "text") else str(response)
            return text.strip()
        except Exception as exc:
            log.error("proactive_engine.agent_error", error=str(exc))
            # Fall back to raw context as utterance (stripped to first sentence)
            sentences = proposal.context.split(".")
            return sentences[0].strip() + "." if sentences else proposal.context

    def _pick_channel(self) -> str:
        """Return 'voice' or 'text' — prefer voice if available, else text."""
        if self._voice_channel is not None:
            return "voice"
        if self._text_channel is not None:
            return "text"
        return "voice"  # no channel available — log warning

    async def _say(self, text: str, channel_name: str) -> None:
        """Deliver text via the named channel's .say() method."""
        channel = (
            self._voice_channel if channel_name == "voice" else self._text_channel
        )
        if channel is None:
            log.warning(
                "proactive_engine.no_channel",
                requested=channel_name,
                text_available=self._text_channel is not None,
                voice_available=self._voice_channel is not None,
            )
            return
        say_fn = getattr(channel, "say", None)
        if not callable(say_fn):
            log.warning("proactive_engine.channel_no_say", channel=channel_name)
            return
        try:
            result = say_fn(text)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            log.error("proactive_engine.say_error", channel=channel_name, error=str(exc))
