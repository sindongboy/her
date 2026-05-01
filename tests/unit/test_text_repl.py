"""Unit tests for TextChannel REPL.

Uses FakeAgent with canned responses and scripted input/output lists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from apps.agent import AttachmentRef
from apps.channels.text.repl import TextChannel
from apps.memory.store import MemoryStore


# ── fakes ─────────────────────────────────────────────────────────────────────


@dataclass
class FakeAgentResponse:
    text: str
    episode_id: int | None = 1
    used_episode_ids: list[int] = field(default_factory=list)
    used_fact_ids: list[int] = field(default_factory=list)


class FakeAgent:
    """Records all calls and returns a canned response."""

    def __init__(self, reply: str = "안녕하세요!") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def respond(
        self,
        message: str,
        *,
        episode_id: int | None,
        channel: str,
        attachments: list[AttachmentRef] | None = None,
    ) -> FakeAgentResponse:
        self.calls.append(
            {
                "message": message,
                "episode_id": episode_id,
                "channel": channel,
                "attachments": attachments,
            }
        )
        return FakeAgentResponse(text=self.reply, episode_id=episode_id or 1)


# ── helpers ──────────────────────────────────────────────────────────────────


def _build_channel(
    inputs: list[str],
    outputs: list[str],
    agent: FakeAgent,
    store: MemoryStore,
    *,
    attachments_dir: Path,
) -> TextChannel:
    """Create a TextChannel with scripted input/output lists."""
    input_iter = iter(inputs)

    def _input(prompt: str) -> str:
        try:
            return next(input_iter)
        except StopIteration:
            raise EOFError("no more input")

    def _output(text: str) -> None:
        outputs.append(text)

    return TextChannel(
        agent,
        store,
        attachments_dir=attachments_dir,
        input_fn=_input,
        output_fn=_output,
    )


def _run(channel: TextChannel) -> None:
    asyncio.run(channel.run())


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def attachments_dir(tmp_path: Path) -> Path:
    d = tmp_path / "attachments"
    d.mkdir()
    return d


# ── tests ─────────────────────────────────────────────────────────────────────


def test_quit_stops_loop(store: MemoryStore, attachments_dir: Path) -> None:
    outputs: list[str] = []
    agent = FakeAgent()
    ch = _build_channel(["/quit"], outputs, agent, store, attachments_dir=attachments_dir)
    _run(ch)

    assert any("종료" in line for line in outputs)
    assert agent.calls == []


def test_exit_stops_loop(store: MemoryStore, attachments_dir: Path) -> None:
    outputs: list[str] = []
    agent = FakeAgent()
    ch = _build_channel(["/exit"], outputs, agent, store, attachments_dir=attachments_dir)
    _run(ch)

    assert any("종료" in line for line in outputs)


def test_help_lists_commands(store: MemoryStore, attachments_dir: Path) -> None:
    outputs: list[str] = []
    agent = FakeAgent()
    ch = _build_channel(["/help", "/quit"], outputs, agent, store, attachments_dir=attachments_dir)
    _run(ch)

    combined = "\n".join(outputs)
    assert "/attach" in combined
    assert "/people" in combined
    assert "/quit" in combined


def test_normal_message_calls_agent_and_prints_response(
    store: MemoryStore, attachments_dir: Path
) -> None:
    outputs: list[str] = []
    agent = FakeAgent(reply="반갑습니다!")
    ch = _build_channel(
        ["안녕", "/quit"], outputs, agent, store, attachments_dir=attachments_dir
    )
    _run(ch)

    assert len(agent.calls) == 1
    assert agent.calls[0]["message"] == "안녕"
    assert agent.calls[0]["channel"] == "text"
    assert any("반갑습니다!" in line for line in outputs)


def test_multiple_messages_share_episode(
    store: MemoryStore, attachments_dir: Path
) -> None:
    """After first message sets episode_id, subsequent messages reuse it."""
    outputs: list[str] = []
    agent = FakeAgent(reply="ok")
    ch = _build_channel(
        ["첫 번째", "두 번째", "/quit"],
        outputs,
        agent,
        store,
        attachments_dir=attachments_dir,
    )
    _run(ch)

    assert len(agent.calls) == 2
    # second call should pass the episode_id from first response (=1)
    assert agent.calls[1]["episode_id"] == 1


def test_eof_stops_loop_gracefully(store: MemoryStore, attachments_dir: Path) -> None:
    """Empty input list → EOFError → graceful shutdown."""
    outputs: list[str] = []
    agent = FakeAgent()
    ch = _build_channel([], outputs, agent, store, attachments_dir=attachments_dir)
    _run(ch)

    assert any("종료" in line for line in outputs)


def test_unknown_command_shows_error(store: MemoryStore, attachments_dir: Path) -> None:
    outputs: list[str] = []
    agent = FakeAgent()
    ch = _build_channel(
        ["/unknown_cmd", "/quit"], outputs, agent, store, attachments_dir=attachments_dir
    )
    _run(ch)

    assert any("알 수 없는 명령어" in line for line in outputs)


def test_people_no_entries(store: MemoryStore, attachments_dir: Path) -> None:
    outputs: list[str] = []
    agent = FakeAgent()
    ch = _build_channel(
        ["/people", "/quit"], outputs, agent, store, attachments_dir=attachments_dir
    )
    _run(ch)

    assert any("없습니다" in line for line in outputs)


def test_people_shows_registered_person(store: MemoryStore, attachments_dir: Path) -> None:
    store.add_person("홍길동", relation="아버지", birthday="1960-01-01")
    outputs: list[str] = []
    agent = FakeAgent()
    ch = _build_channel(
        ["/people", "/quit"], outputs, agent, store, attachments_dir=attachments_dir
    )
    _run(ch)

    combined = "\n".join(outputs)
    assert "홍길동" in combined
    assert "아버지" in combined


def test_attach_valid_file(
    tmp_path: Path, store: MemoryStore, attachments_dir: Path
) -> None:
    src = tmp_path / "note.txt"
    src.write_bytes(b"some content")

    outputs: list[str] = []
    agent = FakeAgent()
    ch = _build_channel(
        [f"/attach {src}", "/quit"],
        outputs,
        agent,
        store,
        attachments_dir=attachments_dir,
    )
    _run(ch)

    combined = "\n".join(outputs)
    assert "첨부 완료" in combined
    assert "note.txt" in combined


def test_attach_failure_shows_korean_error_for_bad_ext(
    tmp_path: Path, store: MemoryStore, attachments_dir: Path
) -> None:
    src = tmp_path / "malware.exe"
    src.write_bytes(b"MZ")

    outputs: list[str] = []
    agent = FakeAgent()
    ch = _build_channel(
        [f"/attach {src}", "/quit"],
        outputs,
        agent,
        store,
        attachments_dir=attachments_dir,
    )
    _run(ch)

    combined = "\n".join(outputs)
    assert "[오류]" in combined
    assert "허용" in combined  # Korean error message contains "허용"


def test_attach_failure_missing_file(
    tmp_path: Path, store: MemoryStore, attachments_dir: Path
) -> None:
    missing = tmp_path / "ghost.txt"

    outputs: list[str] = []
    agent = FakeAgent()
    ch = _build_channel(
        [f"/attach {missing}", "/quit"],
        outputs,
        agent,
        store,
        attachments_dir=attachments_dir,
    )
    _run(ch)

    combined = "\n".join(outputs)
    assert "[오류]" in combined
    assert "찾을 수 없습니다" in combined


def test_attach_without_path_shows_usage(
    store: MemoryStore, attachments_dir: Path
) -> None:
    outputs: list[str] = []
    agent = FakeAgent()
    ch = _build_channel(
        ["/attach", "/quit"], outputs, agent, store, attachments_dir=attachments_dir
    )
    _run(ch)

    combined = "\n".join(outputs)
    assert "사용법" in combined
