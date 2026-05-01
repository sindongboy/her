"""Agent Core — the main orchestrator.

Wires together: Memory recall → anonymization → Gemini LLM → de-anonymize
→ persist → return AgentResponse.

CLAUDE.md references:
  §2.3  Anonymization only at LLM boundary
  §3.1  Gemini exclusive
  §5.4  Recall strategy
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
from apps.web import Event, EventBus

log = structlog.get_logger(__name__)

# Max memory_recall events emitted per call (facts + sessions + notes + attachments).
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

# Max summary length stored in sessions.summary per turn.
_SUMMARY_MAX_CHARS = 200

_SYSTEM_BASE = """당신은 따뜻하고 가까운 가족 같은 AI 비서입니다.
사용자와 그 가족을 기억하고, 일정과 이벤트를 챙기며, 먼저 제안합니다.
한국어로 자연스럽게 대화하세요. 사용자가 영어로 말하면 영어로 답하세요.
응답은 텍스트 웹 UI 에 표시되므로 마크다운을 적절히 활용해 가독성을 높이세요."""


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
    session_id: int
    used_session_ids: list[int]
    used_fact_ids: list[int]
    used_note_ids: list[int] = field(default_factory=list)
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
        session_id: int | None = None,
        attachments: list[AttachmentRef] | None = None,
    ) -> AgentResponse:
        """Full response cycle: recall → LLM → persist → return."""
        sid = await self._ensure_session(session_id)
        log.info(
            "agent.respond.start",
            session_id=sid,
            attachment_count=len(attachments) if attachments else 0,
        )

        ctx = await recall(self.store, message, self._gemini, session_id=sid)

        system_prompt = _build_system_prompt()
        messages, alias_map = self._build_messages(message, ctx)

        extra_parts = parts_from_attachment_refs(attachments) if attachments else []

        text = await asyncio.to_thread(
            self._generate_sync,
            messages,
            system_prompt,
            extra_parts,
        )

        if self.enable_anonymization and alias_map:
            anonymizer = Anonymizer(self.store.list_people())
            text = anonymizer.restore(text, alias_map)

        self._persist_turn(sid, message, text)

        used_session_ids = [s[0] for s in ctx.sessions]
        used_fact_ids = [f[0] for f in ctx.facts]
        used_note_ids = [n[0] for n in ctx.notes]
        used_attachment_ids = list(ctx.attachment_ids) if ctx.attachment_ids else []

        log.info(
            "agent.respond.done",
            session_id=sid,
            response_chars=len(text),
            recalled_sessions=len(used_session_ids),
            attachment_parts=len(extra_parts),
        )
        return AgentResponse(
            text=text,
            session_id=sid,
            used_session_ids=used_session_ids,
            used_fact_ids=used_fact_ids,
            used_note_ids=used_note_ids,
            used_attachment_ids=used_attachment_ids,
        )

    async def stream_respond(
        self,
        message: str,
        *,
        session_id: int | None = None,
        attachments: list[AttachmentRef] | None = None,
    ) -> AsyncIterator[str]:
        """Streaming response. Yields chunks; persists assembled text at end."""
        sid = await self._ensure_session(session_id)
        log.info(
            "agent.stream.start",
            session_id=sid,
            attachment_count=len(attachments) if attachments else 0,
        )

        ctx = await recall(self.store, message, self._gemini, session_id=sid)

        self._publish_recall_events(ctx)

        system_prompt = _build_system_prompt()
        messages, alias_map = self._build_messages(message, ctx)

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
                        payload={"text": chunk},
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
        self._persist_turn(sid, message, full_text)

        log.info(
            "agent.stream.done",
            session_id=sid,
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

    async def _ensure_session(self, session_id: int | None) -> int:
        """Return existing session_id or create a new session row.

        SQLite connections are single-threaded; we call store.conn directly
        on the event loop thread rather than via asyncio.to_thread.
        """
        if session_id is not None:
            return session_id
        return self.store.add_session()

    def _build_messages(
        self,
        user_message: str,
        ctx: RecallContext,
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

    def _persist_turn(self, session_id: int, user_msg: str, assistant_msg: str) -> None:
        """Append user + assistant rows to messages, refresh session summary."""
        self.store.add_message(session_id, "user", user_msg)
        self.store.add_message(session_id, "assistant", assistant_msg)

        snippet = f"U: {user_msg[:80]} | A: {assistant_msg[:80]}"
        snippet = snippet[:_SUMMARY_MAX_CHARS]
        self.store.set_session_summary(session_id, snippet)
        log.debug("agent.persist", session_id=session_id)

    def _publish_recall_events(self, ctx: RecallContext) -> None:
        """Emit memory_recall events for the top-3 of each recalled kind."""
        for _, person_name, _pred, _obj in ctx.facts[:_MAX_RECALL_EVENTS_EACH]:
            _publish(
                self._bus,
                Event(
                    type="memory_recall",
                    payload={"kind": "fact", "person_name": person_name or None},
                    ts=time.monotonic(),
                ),
            )
        for _sid, _text, _score in ctx.sessions[:_MAX_RECALL_EVENTS_EACH]:
            _publish(
                self._bus,
                Event(
                    type="memory_recall",
                    payload={"kind": "session", "person_name": None},
                    ts=time.monotonic(),
                ),
            )
        for _nid, _content in ctx.notes[:_MAX_RECALL_EVENTS_EACH]:
            _publish(
                self._bus,
                Event(
                    type="memory_recall",
                    payload={"kind": "note", "person_name": None},
                    ts=time.monotonic(),
                ),
            )
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


def _build_system_prompt() -> str:
    base = _load_persona()
    return base + get_world_state_block()


def _format_recall_context(ctx: RecallContext) -> str:
    """Serialise RecallContext to a compact text block for the LLM."""
    parts: list[str] = []

    if ctx.facts:
        lines = [f"- {name}: {predicate} = {obj}" for _, name, predicate, obj in ctx.facts]
        parts.append("사실 (Facts):\n" + "\n".join(lines))

    if ctx.upcoming_events:
        lines = [f"- {title} ({when})" for _, title, when in ctx.upcoming_events]
        parts.append("예정 이벤트 (Upcoming Events):\n" + "\n".join(lines))

    if ctx.notes:
        lines = [f"- {content}" for _, content in ctx.notes if content]
        if lines:
            parts.append("메모 (Notes):\n" + "\n".join(lines))

    if ctx.sessions:
        lines = [
            f"- [{score:.2f}] {text}" for _, text, score in ctx.sessions if text
        ]
        if lines:
            parts.append("관련 대화 (Related Sessions):\n" + "\n".join(lines))

    return "\n\n".join(parts)
