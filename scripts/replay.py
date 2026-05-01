"""scripts/replay.py — JSONL fixture regression replay for §7.2.

Usage:
  python scripts/replay.py <fixture_path> [--db PATH] [--no-llm] [--debug]

JSONL format (one object per line):

  Phase 0 (backward-compatible):
    {"role": "user", "channel": "text", "text": "..."}

  Phase 2 (with optional attachments):
    {"role": "user", "channel": "text", "text": "이 문서 요약해줘",
     "attachments": [
       {"path": "tests/fixtures/sample_attachment.txt",
        "description": "어머니께 받은 메시지"}
     ]}

  The 'attachments' field is optional. Phase 0 fixtures load unchanged.
  Relative paths in 'attachments[].path' are resolved against the directory
  containing the fixture file.

--no-llm mode: exercises recall + persistence using FakeAgentCore (no API key
  needed). Attachment paths are printed so the operator can verify forwarding.
--debug: prints AgentResponse fields (used_episode_ids, used_fact_ids,
  used_attachment_ids) after each turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure repo root is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── AttachmentRef (shim / real import) ───────────────────────────────────────

try:
    from apps.agent.attachments import AttachmentRef  # type: ignore[import]
except ImportError:
    # multimodal-eng hasn't landed yet — use a local shim with the same shape.
    @dataclass(slots=True, frozen=True)
    class AttachmentRef:  # type: ignore[no-redef]
        """Shim for AttachmentRef until multimodal-eng lands apps.agent.attachments."""

        path: Path
        mime: str | None = None
        sha256: str | None = None
        description: str | None = None


# ── Turn dataclass ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    channel: str
    text: str
    raw: dict[str, Any]
    attachments: list[AttachmentRef] = field(default_factory=list)


# ── JSONL loader (pure logic — testable without agent) ───────────────────────


def _parse_attachments(
    raw_list: Any,
    lineno: int,
    fixture_dir: Path,
) -> list[AttachmentRef]:
    """Parse the 'attachments' value from a JSONL line.

    Paths are resolved relative to *fixture_dir*.

    Raises:
        ValueError: if *raw_list* is not a list (with line number in message).
    """
    if not isinstance(raw_list, list):
        raise ValueError(
            f"Line {lineno}: 'attachments' must be a list, got {type(raw_list).__name__}"
        )
    refs: list[AttachmentRef] = []
    for item in raw_list:
        if not isinstance(item, dict):
            raise ValueError(
                f"Line {lineno}: each attachment entry must be a JSON object"
            )
        raw_path = item.get("path", "")
        resolved = (fixture_dir / raw_path).resolve() if raw_path else Path(raw_path)
        refs.append(
            AttachmentRef(
                path=resolved,
                mime=item.get("mime") or None,
                sha256=item.get("sha256") or None,
                description=item.get("description") or None,
            )
        )
    return refs


def load_turns(path: Path) -> list[Turn]:
    """Parse a JSONL fixture file and return a list of Turn objects.

    Phase 2 extension: optional 'attachments' field is parsed into
    ``Turn.attachments`` as a list of ``AttachmentRef``. Relative paths are
    resolved against the directory containing *path*.

    Raises:
        ValueError: if a line is not valid JSON, with the 1-based line number.
        ValueError: if a line is missing required 'role' or 'text' fields.
        ValueError: if 'attachments' is present but not a list.
    """
    fixture_dir = path.parent
    turns: list[Turn] = []
    text = path.read_text(encoding="utf-8")
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj: dict[str, Any] = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {lineno}: invalid JSON — {exc}") from exc
        if "role" not in obj:
            raise ValueError(f"Line {lineno}: missing required field 'role'")
        if "text" not in obj:
            raise ValueError(f"Line {lineno}: missing required field 'text'")

        attachments: list[AttachmentRef] = []
        if "attachments" in obj:
            attachments = _parse_attachments(obj["attachments"], lineno, fixture_dir)

        turns.append(
            Turn(
                role=obj["role"],
                channel=obj.get("channel", "text"),
                text=obj["text"],
                raw=obj,
                attachments=attachments,
            )
        )
    return turns


# ── FakeAgentCore (--no-llm) ─────────────────────────────────────────────────


class FakeAgentCore:
    """Minimal stub used in --no-llm mode.

    Exercises MemoryStore persistence without real LLM calls.
    In Phase 2, also prints attachment paths so the operator can verify
    that attachments were forwarded correctly.
    """

    def __init__(self, db_path: Path) -> None:
        from apps.memory.store import MemoryStore

        self._store = MemoryStore(db_path)

    async def respond(
        self,
        text: str,
        *,
        episode_id: str | int | None = None,
        channel: str = "text",
        attachments: list[AttachmentRef] | None = None,
    ) -> Any:
        """Save the user turn as an episode summary and return a stub reply.

        Prints attachment paths so the operator can visually verify forwarding.
        Returns an object with the same fields as AgentResponse for --debug
        compatibility.
        """
        ep_id = self._store.add_episode(
            summary=text[:120],
            primary_channel=channel,
        )
        if attachments:
            for att in attachments:
                print(f"   [attachment] {att.path}  desc={att.description!r}")

        @dataclass
        class _FakeResponse:
            text: str
            episode_id: int
            used_episode_ids: list[int]
            used_fact_ids: list[int]
            used_attachment_ids: list[int]

        return _FakeResponse(
            text=f"[fake] episode={ep_id} received: {text[:60]}",
            episode_id=ep_id,
            used_episode_ids=[],
            used_fact_ids=[],
            used_attachment_ids=[],
        )

    def close(self) -> None:
        self._store.close()


# ── real AgentCore loader ─────────────────────────────────────────────────────


def _load_real_agent(db_path: Path) -> Any:
    """Import AgentCore and return an initialised instance.

    Returns None and exits with code 3 if the module is not yet available.
    """
    try:
        from apps.agent import AgentCore  # type: ignore[attr-defined]

        return AgentCore(db_path=db_path)
    except ImportError as exc:
        print(
            f"[replay] agent module not yet available: {exc}\n"
            "Run with --no-llm to replay without a real agent.",
            file=sys.stderr,
        )
        sys.exit(3)


# ── attachment validator ──────────────────────────────────────────────────────


def _check_attachments(turns: list[Turn], fixture_path: Path) -> bool:
    """Warn about attachment paths that do not exist on disk.

    Returns True if all attachment paths exist (or there are none).
    Does NOT abort the replay — just prints warnings.
    """
    all_ok = True
    for i, turn in enumerate(turns, start=1):
        for att in turn.attachments:
            if not att.path.exists():
                print(
                    f"[replay] WARNING turn {i}: attachment not found: {att.path}",
                    file=sys.stderr,
                )
                all_ok = False
    return all_ok


# ── replay runner ─────────────────────────────────────────────────────────────


async def replay(
    turns: list[Turn],
    agent: Any,
    episode_id: str | int,
    *,
    debug: bool = False,
) -> tuple[int, int]:
    """Drive turns through agent.respond().

    Returns (user_turns_processed, total_attachments_forwarded).
    """
    processed = 0
    total_attachments = 0
    for turn in turns:
        if turn.role != "user":
            continue
        print(f">> {turn.text}")
        if turn.attachments:
            print(f"   [attachments x{len(turn.attachments)}]")
            total_attachments += len(turn.attachments)

        kwargs: dict[str, Any] = {
            "episode_id": episode_id,
            "channel": turn.channel,
        }
        # Always pass the keyword; pass None when empty so agent can handle it.
        kwargs["attachments"] = turn.attachments if turn.attachments else None

        response: Any = await agent.respond(turn.text, **kwargs)

        # Support both old (str) and new (AgentResponse) return types.
        response_text = response.text if hasattr(response, "text") else str(response)
        print(f"<< {response_text}")

        if debug and hasattr(response, "used_episode_ids"):
            print(
                f"   [debug] episode_id={getattr(response, 'episode_id', '?')}"
                f"  used_episodes={getattr(response, 'used_episode_ids', [])}"
                f"  used_facts={getattr(response, 'used_fact_ids', [])}"
                f"  used_attachments={getattr(response, 'used_attachment_ids', [])}"
            )
        print()
        processed += 1
    return processed, total_attachments


# ── main ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay a JSONL fixture against AgentCore for regression testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("fixture", help="Path to .jsonl fixture file")
    p.add_argument(
        "--db",
        default="data/db.sqlite",
        help="SQLite DB path (default: data/db.sqlite)",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip real LLM calls; use FakeAgentCore (CI-safe, no GEMINI_API_KEY needed)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print AgentResponse fields (used_episode_ids, used_fact_ids, used_attachment_ids) after each turn",
    )
    return p


async def _main_async(args: argparse.Namespace) -> None:
    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"[replay] fixture not found: {fixture_path}", file=sys.stderr)
        sys.exit(1)

    turns = load_turns(fixture_path)
    if not turns:
        print("[replay] fixture is empty — nothing to replay.")
        return

    _check_attachments(turns, fixture_path)

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    episode_id = str(uuid.uuid4())
    print(
        f"[replay] fixture={fixture_path.name}  episode_id={episode_id}"
        f"  no-llm={args.no_llm}  debug={args.debug}\n"
    )

    if args.no_llm:
        agent: Any = FakeAgentCore(db_path)
    else:
        agent = _load_real_agent(db_path)

    try:
        processed, total_attachments = await replay(
            turns, agent, episode_id, debug=args.debug
        )
    finally:
        if hasattr(agent, "close"):
            agent.close()

    user_turns = sum(1 for t in turns if t.role == "user")
    all_att = sum(len(t.attachments) for t in turns if t.role == "user")
    print(
        f"[replay] 완료 — {processed}/{user_turns} 턴 처리"
        f"  첨부파일 {total_attachments}/{all_att}개 전달"
        f"  episode_id={episode_id}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
