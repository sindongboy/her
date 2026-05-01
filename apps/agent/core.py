"""Agent Core — the main orchestrator.

Wires together: Memory recall → anonymization → Gemini LLM → de-anonymize
→ persist → return AgentResponse.

CLAUDE.md references:
  §2.1  Voice-friendly output when channel='voice' (no markdown)
  §2.3  Anonymization only at LLM boundary
  §3.1  Gemini exclusive
  §4    Channel-agnostic envelope; Agent Core knows no channel details
  §5.4  3-strategy recall
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from apps.agent.anonymize import Anonymizer
from apps.agent.gemini import GeminiClient
from apps.agent.multimodal import parts_from_attachment_refs
from apps.agent.recall import RecallContext, recall
from apps.agent.world_context import get_world_state_block, init_world_state_cache
from apps.memory.store import MemoryStore
from apps.presence import Event, EventBus

log = structlog.get_logger(__name__)

# Max memory_recall events emitted per call (facts + episodes + attachments).
_MAX_RECALL_EVENTS_EACH = 3


# ── safe publish helper ───────────────────────────────────────────────────────


def _publish(bus: EventBus | None, event: Event) -> None:
    """Publish *event* to *bus* without ever raising.

    None bus → silent no-op.  Any exception from the bus is swallowed and
    logged so presence failures never interrupt agent logic.
    """
    if bus is None:
        return
    try:
        bus.publish(event)
    except Exception as exc:  # pragma: no cover
        log.warning("publish_failed", error=str(exc))

# Max summary length stored in episodes.summary per turn.
_SUMMARY_MAX_CHARS = 200

_SYSTEM_BASE = """당신은 따뜻하고 가까운 가족 같은 AI 비서입니다.
사용자와 그 가족을 기억하고, 일정과 이벤트를 챙기며, 먼저 제안합니다.
한국어로 자연스럽게 대화하세요. 사용자가 영어로 말하면 영어로 답하세요."""

_SYSTEM_VOICE_ADDON = """
지금 출력 채널은 음성(TTS)입니다. 반드시 다음 규칙을 따르세요:
- 마크다운(#, **, -, 번호 목록 등) 사용 금지
- 짧고 자연스러운 구어체 문장으로만 응답
- 리스트가 필요하면 "첫째, 둘째, 셋째"처럼 말로 표현"""

_SYSTEM_TEXT_ADDON = """
지금 출력 채널은 텍스트입니다. 마크다운을 적절히 활용해 가독성을 높이세요."""


@dataclass(slots=True, frozen=True)
class AttachmentRef:
    """Stable handle for an attachment already on disk.

    The channel layer computes sha256 + stores to DB; the agent receives
    these refs and converts them to Gemini Parts for the LLM call.
    """

    path: Path
    mime: str | None = None
    sha256: str | None = None  # for caller bookkeeping only; agent ignores it
    description: str | None = None  # Korean OK; anonymizer applied before LLM


@dataclass(slots=True, frozen=True)
class AgentResponse:
    text: str
    episode_id: int
    used_episode_ids: list[int]
    used_fact_ids: list[int]
    used_attachment_ids: list[int] = field(default_factory=list)


class AgentCore:
    """Orchestrates recall → anonymize → LLM → de-anonymize → persist.

    Parameters
    ----------
    store:
        The MemoryStore instance (shared with other modules).
    api_key:
        Gemini API key. Falls back to GEMINI_API_KEY env var.
    model_id:
        Gemini model to use.
    client:
        Inject a test double implementing the same interface as GeminiClient.
    enable_anonymization:
        Set False to disable name redaction (e.g., trusted local deployments
        or unit tests that inspect raw content).
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        api_key: str | None = None,
        model_id: str = "gemini-3.1-pro-preview",
        client: object | None = None,
        enable_anonymization: bool = True,
        bus: EventBus | None = None,
        settings: object | None = None,
    ) -> None:
        self.store = store
        self.enable_anonymization = enable_anonymization
        self._bus: EventBus | None = bus

        if client is not None:
            self._gemini: GeminiClient = client  # type: ignore[assignment]
        else:
            self._gemini = GeminiClient(
                model_id=model_id,
                api_key=api_key,
            )

        # World-state cache (weather + location + time block).
        # If no settings provided, load from disk (or defaults).
        if settings is None:
            try:
                from apps.settings import load_settings as _load
                settings = _load()
            except Exception:
                settings = None
        self._settings = settings
        self._enable_search_grounding: bool = bool(
            getattr(settings, "enable_search_grounding", True)
        )
        self._world_state_cache = init_world_state_cache(settings)

    # ── public API ───────────────────────────────────────────────────────

    async def respond(
        self,
        message: str,
        *,
        episode_id: int | None = None,
        channel: str = "text",
        attachments: list[AttachmentRef] | None = None,
    ) -> AgentResponse:
        """Full response cycle: recall → LLM → persist → return."""
        ep_id = await self._ensure_episode(episode_id, channel)
        log.info(
            "agent.respond.start",
            episode_id=ep_id,
            channel=channel,
            attachment_count=len(attachments) if attachments else 0,
        )

        ctx = await recall(self.store, message, self._gemini, episode_id=ep_id)

        system_prompt = _build_system_prompt(channel)
        messages, alias_map = self._build_messages(message, ctx, channel)

        # Build multimodal parts from attachments (binary / text files).
        extra_parts = parts_from_attachment_refs(attachments) if attachments else []

        # generate() is network-bound; run in thread so event loop stays responsive.
        # _persist_turn writes to SQLite via the same conn — must NOT run in a
        # worker thread (SQLite single-thread constraint). Run it inline instead.
        text = await asyncio.to_thread(
            self._generate_sync,
            messages,
            system_prompt,
            extra_parts,
        )

        if self.enable_anonymization and alias_map:
            anonymizer = Anonymizer(self.store.list_people())
            text = anonymizer.restore(text, alias_map)

        self._persist_turn(ep_id, message, text)

        used_episode_ids = [e[0] for e in ctx.episodes]
        used_fact_ids = [f[0] for f in ctx.facts]
        used_attachment_ids = list(ctx.attachment_ids) if ctx.attachment_ids else []

        log.info(
            "agent.respond.done",
            episode_id=ep_id,
            response_chars=len(text),
            recalled_episodes=len(used_episode_ids),
            attachment_parts=len(extra_parts),
        )
        return AgentResponse(
            text=text,
            episode_id=ep_id,
            used_episode_ids=used_episode_ids,
            used_fact_ids=used_fact_ids,
            used_attachment_ids=used_attachment_ids,
        )

    async def stream_respond(
        self,
        message: str,
        *,
        episode_id: int | None = None,
        channel: str = "text",
        attachments: list[AttachmentRef] | None = None,
    ) -> AsyncIterator[str]:
        """Streaming response. Yields chunks; persists assembled text at end."""
        ep_id = await self._ensure_episode(episode_id, channel)
        log.info(
            "agent.stream.start",
            episode_id=ep_id,
            channel=channel,
            attachment_count=len(attachments) if attachments else 0,
        )

        ctx = await recall(self.store, message, self._gemini, episode_id=ep_id)

        # Emit memory_recall events (capped at top-3 each kind — ambient orb only).
        self._publish_recall_events(ctx, channel)

        system_prompt = _build_system_prompt(channel)
        messages, alias_map = self._build_messages(message, ctx, channel)

        extra_parts = parts_from_attachment_refs(attachments) if attachments else []

        accumulated: list[str] = []

        try:
            async for chunk in self._generate_stream_async(
                messages,
                system_prompt,
                extra_parts,
            ):
                if self.enable_anonymization and alias_map:
                    anonymizer = Anonymizer(self.store.list_people())
                    chunk = anonymizer.restore(chunk, alias_map)
                accumulated.append(chunk)
                _publish(
                    self._bus,
                    Event(
                        type="response_chunk",
                        payload={"text": chunk, "channel": channel},
                        ts=time.monotonic(),
                    ),
                )
                yield chunk
        except Exception as exc:
            _publish(
                self._bus,
                Event(
                    type="error",
                    payload={"message": str(exc), "where": "agent.stream_respond"},
                    ts=time.monotonic(),
                ),
            )
            raise

        full_text = "".join(accumulated)
        # SQLite must stay on the originating thread — call inline.
        self._persist_turn(ep_id, message, full_text)

        log.info(
            "agent.stream.done",
            episode_id=ep_id,
            response_chars=len(full_text),
        )

    # ── private helpers ──────────────────────────────────────────────────

    def _generate_sync(
        self,
        messages: list[dict],
        system: str,
        parts: list,
    ) -> str:
        """Call gemini.generate, passing enable_search_grounding if supported.

        Falls back silently if the injected client (e.g. test double) doesn't
        accept that kwarg — keeps existing tests working without modification.
        """
        try:
            return self._gemini.generate(  # type: ignore[union-attr]
                messages,
                system=system,
                parts=parts,
                enable_search_grounding=self._enable_search_grounding,
            )
        except TypeError:
            # Injected test double without the new kwarg — call without it.
            return self._gemini.generate(messages, system=system, parts=parts)  # type: ignore[union-attr]

    async def _generate_stream_async(
        self,
        messages: list[dict],
        system: str,
        parts: list,
    ):  # type: ignore[return]
        """Async generator wrapping gemini.generate_stream with grounding kwarg.

        Falls back if the injected client doesn't accept enable_search_grounding.
        """
        import inspect

        gen = self._gemini.generate_stream  # type: ignore[union-attr]
        sig = inspect.signature(gen)
        if "enable_search_grounding" in sig.parameters:
            async for chunk in gen(
                messages,
                system=system,
                parts=parts,
                enable_search_grounding=self._enable_search_grounding,
            ):
                yield chunk
        else:
            async for chunk in gen(messages, system=system, parts=parts):
                yield chunk

    async def _ensure_episode(self, episode_id: int | None, channel: str) -> int:
        """Return existing episode_id or create a new episode row.

        SQLite connections are single-threaded; we call store.conn directly
        on the event loop thread rather than via asyncio.to_thread.
        """
        if episode_id is not None:
            return episode_id
        return self._create_episode(channel)

    def _create_episode(self, channel: str) -> int:
        """Insert a new episode row and return its id (synchronous)."""
        # Try typed method first (memory-eng may have provided it).
        add_episode = getattr(self.store, "add_episode", None)
        if callable(add_episode):
            try:
                return int(add_episode(channel=channel))
            except TypeError:
                pass  # Signature mismatch — fall through to raw SQL.

        # Raw SQL fallback — always available via store.conn.
        cur = self.store.conn.execute(
            "INSERT INTO episodes (when_at, summary, primary_channel) "
            "VALUES (CURRENT_TIMESTAMP, NULL, ?)",
            (channel,),
        )
        return int(cur.lastrowid or 0)

    def _build_messages(
        self,
        user_message: str,
        ctx: RecallContext,
        channel: str,
    ) -> tuple[list[dict[str, str]], dict[str, str]]:
        """Build the messages list and alias map to send to Gemini."""
        people = self.store.list_people()
        anonymizer = Anonymizer(people)

        memory_block = _format_recall_context(ctx)

        alias_map: dict[str, str] = {}
        if self.enable_anonymization:
            memory_block, _ = anonymizer.redact(memory_block)
            user_message, alias_map = anonymizer.redact(user_message)

        parts: list[str] = []
        if memory_block.strip():
            parts.append(f"[MEMORY]\n{memory_block.strip()}")
        parts.append(user_message)

        final_content = "\n\n".join(parts)

        return [{"role": "user", "content": final_content}], alias_map

    def _persist_turn(self, episode_id: int, user_msg: str, assistant_msg: str) -> None:
        """Update episode summary with a truncated exchange snapshot."""
        snippet = f"U: {user_msg[:80]} | A: {assistant_msg[:80]}"
        snippet = snippet[:_SUMMARY_MAX_CHARS]
        self.store.conn.execute(
            "UPDATE episodes SET summary = ? WHERE id = ?",
            (snippet, episode_id),
        )
        log.debug("agent.persist", episode_id=episode_id)

    def _publish_recall_events(self, ctx: RecallContext, channel: str) -> None:
        """Emit memory_recall events for the top-3 of each recalled kind.

        These drive optional ambient pulses on the orb — not critical-path.
        Capped to avoid spamming the bus on large recall results.
        """
        # Facts: (fact_id, person_name, predicate, object)
        for _, person_name, _pred, _obj in ctx.facts[:_MAX_RECALL_EVENTS_EACH]:
            _publish(
                self._bus,
                Event(
                    type="memory_recall",
                    payload={"kind": "fact", "person_name": person_name or None},
                    ts=time.monotonic(),
                ),
            )
        # Episodes: (episode_id, summary, score)
        for _ep_id, _summary, _score in ctx.episodes[:_MAX_RECALL_EVENTS_EACH]:
            _publish(
                self._bus,
                Event(
                    type="memory_recall",
                    payload={"kind": "episode", "person_name": None},
                    ts=time.monotonic(),
                ),
            )
        # Attachments: (attachment_id, path, description)
        for _att_id, _path, _desc in ctx.attachments[:_MAX_RECALL_EVENTS_EACH]:
            _publish(
                self._bus,
                Event(
                    type="memory_recall",
                    payload={"kind": "attachment", "person_name": None},
                    ts=time.monotonic(),
                ),
            )


# ── module-level helpers ──────────────────────────────────────────────────


def _persona_path() -> Path:
    """Resolve the persona override file path.

    Order: HER_PERSONA_PATH env > ~/.her/persona.md.
    """
    import os

    env = os.environ.get("HER_PERSONA_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".her" / "persona.md"


_persona_cache: tuple[float, str] | None = None


def _load_persona() -> str:
    """Read persona override file if present, else return baked-in default.

    Cached by mtime — so editing the file and restarting picks up the change
    without code reload, while a long-running daemon doesn't re-read on every
    turn.
    """
    global _persona_cache
    path = _persona_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return _SYSTEM_BASE

    if _persona_cache is not None and _persona_cache[0] == mtime:
        return _persona_cache[1]

    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return _SYSTEM_BASE
        _persona_cache = (mtime, text)
        log.info("persona_loaded", path=str(path), chars=len(text))
        return text
    except OSError as exc:
        log.warning("persona_read_failed", path=str(path), error=str(exc))
        return _SYSTEM_BASE


def _build_system_prompt(channel: str) -> str:
    base = _load_persona()
    addon = _SYSTEM_VOICE_ADDON if channel == "voice" else _SYSTEM_TEXT_ADDON
    return base + addon + get_world_state_block()


def _format_recall_context(ctx: RecallContext) -> str:
    """Serialise RecallContext to a compact text block for the LLM."""
    parts: list[str] = []

    if ctx.facts:
        lines = [f"- {name}: {predicate} = {obj}" for _, name, predicate, obj in ctx.facts]
        parts.append("사실 (Facts):\n" + "\n".join(lines))

    if ctx.upcoming_events:
        lines = [f"- {title} ({when})" for _, title, when in ctx.upcoming_events]
        parts.append("예정 이벤트 (Upcoming Events):\n" + "\n".join(lines))

    if ctx.episodes:
        lines = [f"- [{score:.2f}] {summary}" for _, summary, score in ctx.episodes if summary]
        if lines:
            parts.append("관련 대화 (Related Episodes):\n" + "\n".join(lines))

    return "\n\n".join(parts)
