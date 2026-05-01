"""Text channel REPL for the her assistant.

Phase 0 surface: stdin/stdout with /attach, /people, /help, /quit.
Per CLAUDE.md §2.5 and §6.2.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import structlog

from apps.agent import AttachmentRef
from apps.channels.text.attachments import AttachmentError, AttachmentHandler
from apps.memory.store import MemoryStore
from apps.presence import Event, EventBus

logger = structlog.get_logger(__name__)


# ── safe publish helper ───────────────────────────────────────────────────────


def _publish(bus: EventBus | None, event: Event) -> None:
    """Publish *event* to *bus* without ever raising.

    None bus → silent no-op (backward-compat).  Any bus exception is swallowed
    and logged so presence failures never crash text logic.
    """
    if bus is None:
        return
    try:
        bus.publish(event)
    except Exception as exc:  # pragma: no cover
        logger.warning("publish_failed", error=str(exc))

_PROMPT = "나> "
_RESPONSE_PREFIX = " > "
_SEPARATOR = "-" * 60

_HELP_TEXT = """\
사용 가능한 명령어:
  /attach <경로>        파일 첨부 (최대 25MB, 허용 형식: png jpg jpeg pdf txt md ics eml)
  /clear-attachments    대기 중인 첨부 파일 모두 취소
  /people               등록된 가족/사람 목록 보기
  /help                 이 도움말 표시
  /quit, /exit          종료"""


class AgentProtocol(Protocol):
    """Minimal interface that AgentCore must satisfy.

    Injected via constructor so tests can supply fakes.
    """

    async def respond(
        self,
        message: str,
        *,
        episode_id: int | None,
        channel: str,
        attachments: list[AttachmentRef] | None,
    ) -> Any:
        """Return an AgentResponse-like object with .text and .episode_id."""
        ...


class TextChannel:
    """Stdin/stdout REPL for the text channel.

    Accepts natural language messages and /commands.
    Maintains episode continuity across the session.
    """

    def __init__(
        self,
        agent: AgentProtocol,
        store: MemoryStore,
        *,
        attachments_dir: Path = Path("data/attachments"),
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        bus: EventBus | None = None,
    ) -> None:
        self._agent = agent
        self._store = store
        self._attachments_dir = attachments_dir
        self._input_fn = input_fn
        self._output_fn = output_fn
        self._bus: EventBus | None = bus
        self._attachment_handler = AttachmentHandler(store, attachments_dir)
        self._episode_id: int | None = None
        self._pending_attachments: list[AttachmentRef] = []

    # ── public ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main REPL loop: read → dispatch → print."""
        self._output_fn(_SEPARATOR)
        self._output_fn("안녕하세요! 텍스트로 대화를 시작하세요. /help 로 명령어를 확인하세요.")
        self._output_fn(_SEPARATOR)

        while True:
            try:
                user_input = await asyncio.to_thread(self._input_fn, _PROMPT)
            except EOFError:
                # stdin closed (e.g. piped input finished)
                self._output_fn("\n대화를 종료합니다. 안녕히 계세요!")
                break

            line = user_input.strip()
            if not line:
                continue

            if line.lower() in ("/quit", "/exit"):
                self._output_fn("대화를 종료합니다. 안녕히 계세요!")
                break

            if line == "/help":
                self._output_fn(_HELP_TEXT)
                continue

            if line == "/people":
                await self._handle_people()
                continue

            if line.startswith("/attach"):
                await self._handle_attach(line)
                continue

            if line == "/clear-attachments":
                self._handle_clear_attachments()
                continue

            if line.startswith("/"):
                self._output_fn(f"알 수 없는 명령어입니다: '{line}'. /help 를 입력하면 도움말을 볼 수 있습니다.")
                continue

            # Normal message → send to agent
            await self._handle_message(line)

    async def say(self, text: str) -> None:
        """Proactive utterance: print text to stdout with a distinct prefix.

        Called by ProactiveEngine (§2.4). Emits state events to bus.
        """
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "speaking", "channel": "text"},
                ts=time.monotonic(),
            ),
        )
        self._output_fn(f"[먼저] > {text}")
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "idle", "channel": "text"},
                ts=time.monotonic(),
            ),
        )

    # ── private ─────────────────────────────────────────────────────────

    async def _handle_message(self, text: str) -> None:
        """Forward user text to agent, print response, update episode_id."""
        # Record activity for SilenceTrigger.
        try:
            from apps.proactive.activity import record_activity
            record_activity("text")
        except Exception:  # pragma: no cover
            pass

        # Drain pending attachments, then clear the queue.
        pending = self._pending_attachments[:] if self._pending_attachments else None
        self._pending_attachments = []

        # Emit transcript + thinking state before calling agent.
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "thinking", "channel": "text"},
                ts=time.monotonic(),
            ),
        )
        _publish(
            self._bus,
            Event(
                type="transcript",
                payload={"text": text, "final": True, "channel": "text"},
                ts=time.monotonic(),
            ),
        )

        try:
            response = await self._agent.respond(
                text,
                episode_id=self._episode_id,
                channel="text",
                attachments=pending,
            )
            # Update running episode from the agent's response
            if hasattr(response, "episode_id") and response.episode_id is not None:
                self._episode_id = response.episode_id

            reply = response.text if hasattr(response, "text") else str(response)

            # Agent has started producing output — emit speaking state.
            _publish(
                self._bus,
                Event(
                    type="state",
                    payload={"value": "speaking", "channel": "text"},
                    ts=time.monotonic(),
                ),
            )
            self._output_fn(f"{_RESPONSE_PREFIX}{reply}")

            # Done — emit response_end + idle.
            _publish(
                self._bus,
                Event(
                    type="response_end",
                    payload={"channel": "text", "episode_id": self._episode_id},
                    ts=time.monotonic(),
                ),
            )
            _publish(
                self._bus,
                Event(
                    type="state",
                    payload={"value": "idle", "channel": "text"},
                    ts=time.monotonic(),
                ),
            )
        except Exception as exc:
            # On failure, restore the pending list so the user doesn't lose attachments.
            if pending:
                self._pending_attachments = pending + self._pending_attachments
            logger.error("agent_respond_failed", error=str(exc))
            self._output_fn(f"[오류] 응답 생성 중 문제가 발생했습니다: {exc}")

    async def _handle_attach(self, line: str) -> None:
        """/attach <path> — ingest a file and queue it for the next message."""
        parts = line.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            self._output_fn("사용법: /attach <파일 경로>")
            return

        raw_path = parts[1].strip()
        source_path = Path(raw_path).expanduser().resolve()

        # Ensure we have an episode to attach to
        if self._episode_id is None:
            self._episode_id = self._store.add_episode(
                None,
                primary_channel="text",
            )
            logger.info("episode_created_for_attach", episode_id=self._episode_id)

        try:
            # Run synchronously on the event loop thread — the store's SQLite
            # connection is not thread-safe, so we avoid asyncio.to_thread here.
            # File copies are bounded to MAX_BYTES (25 MB) and fast on local disk.
            result = self._attachment_handler.ingest(self._episode_id, source_path)

            # Queue the attachment for forwarding on the next normal message.
            ref = AttachmentRef(
                path=result.stored_path,
                mime=result.mime,
                sha256=result.sha256,
            )
            self._pending_attachments.append(ref)

            self._output_fn(
                f"첨부 완료: {result.original_name} "
                f"(sha256: {result.sha256[:8]}…, "
                f"대기 중인 첨부: {len(self._pending_attachments)}개)"
            )
        except AttachmentError as exc:
            msg = str(exc)
            if msg.startswith("ext_not_allowed"):
                self._output_fn(
                    "[오류] 허용되지 않는 파일 형식입니다. "
                    "허용 형식: png, jpg, jpeg, pdf, txt, md, ics, eml"
                )
            elif msg.startswith("too_large"):
                self._output_fn("[오류] 파일 크기가 25MB 를 초과합니다.")
            elif msg.startswith("file_not_found"):
                self._output_fn(f"[오류] 파일을 찾을 수 없습니다: {raw_path}")
            else:
                self._output_fn(f"[오류] 첨부 실패: {msg}")
        except Exception as exc:
            logger.error("attach_failed", path=raw_path, error=str(exc))
            self._output_fn(f"[오류] 첨부 처리 중 문제가 발생했습니다: {exc}")

    def _handle_clear_attachments(self) -> None:
        """/clear-attachments — discard all queued attachments."""
        count = len(self._pending_attachments)
        self._pending_attachments = []
        if count > 0:
            self._output_fn(f"대기 중인 첨부 파일 {count}개를 모두 취소했습니다.")
        else:
            self._output_fn("대기 중인 첨부 파일이 없습니다.")

    async def _handle_people(self) -> None:
        """/people — list registered people from memory."""
        try:
            people = self._store.list_people()
            if not people:
                self._output_fn("등록된 사람이 없습니다.")
                return
            lines = [f"등록된 사람 ({len(people)}명):"]
            for p in people:
                parts = [f"  • {p.name}"]
                if p.relation:
                    parts.append(f"({p.relation})")
                if p.birthday:
                    parts.append(f"생일: {p.birthday}")
                lines.append(" ".join(parts))
            self._output_fn("\n".join(lines))
        except Exception as exc:
            logger.error("list_people_failed", error=str(exc))
            self._output_fn(f"[오류] 사람 목록을 가져오는 중 문제가 발생했습니다: {exc}")


# ── module-level runner (used by __main__ and presence) ──────────────────────


async def run_repl(bus: "EventBus | None" = None) -> None:
    """Build and run the text REPL from environment.

    Wires MemoryStore + AgentCore + TextChannel using the same env vars as
    ``python -m apps.channels.text``. Accepts an optional event bus so the
    presence server can subscribe to state events.

    Bus selection precedence (highest to lowest):
      1. Explicit ``bus`` argument — e.g. presence/__main__.py co-process.
      2. ``HER_PRESENCE_URL`` env var → RemoteEventBus (cross-process).
      3. No bus (None) — presence events are silently dropped.
    """
    import os
    import sys
    from pathlib import Path

    from apps.memory.store import MemoryStore

    from apps.settings import load_settings
    s = load_settings()

    # Resolve bus: caller wins; else check env for cross-process URL.
    if bus is None:
        presence_url = os.environ.get("HER_PRESENCE_URL", "").strip()
        if presence_url:
            from apps.presence.remote_bus import RemoteEventBus

            bus = RemoteEventBus(presence_url)  # type: ignore[assignment]
            logger.info("text_channel.using_remote_bus", url=presence_url)

    db_path = Path(os.environ.get("HER_DB_PATH", "data/db.sqlite"))
    api_key = os.environ.get("GEMINI_API_KEY", "")
    agent_model = os.environ.get("HER_AGENT_MODEL", s.agent_model)

    if not api_key:
        print(
            "[경고] GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다. "
            "에이전트 응답이 동작하지 않을 수 있습니다.",
            file=sys.stderr,
        )

    store = MemoryStore(db_path)

    try:
        from apps.agent.core import AgentCore  # type: ignore[import]

        agent: Any = AgentCore(store, api_key=api_key or None, model_id=agent_model, bus=bus)
    except ImportError:
        from apps.channels.text._stub_agent import StubAgent

        print(
            "[경고] apps.agent.core 를 불러올 수 없습니다. 임시 에이전트를 사용합니다.",
            file=sys.stderr,
        )
        agent = StubAgent()

    channel = TextChannel(agent, store, bus=bus)

    try:
        await channel.run()
    finally:
        store.close()
