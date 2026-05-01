"""Unit tests — REPL forwards pending attachments to the agent.

Covers:
- /attach queues an AttachmentRef.
- Multiple /attach calls extend the queue.
- Next normal message forwards all refs to agent.respond and clears queue.
- /clear-attachments empties the queue.
- /attach with missing file leaves the queue unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from apps.agent import AttachmentRef
from apps.channels.text.repl import TextChannel
from apps.memory.store import MemoryStore


# ── helpers / fakes ───────────────────────────────────────────────────────


class FakeAgentResponse:
    def __init__(self, text: str = "응답", episode_id: int = 1) -> None:
        self.text = text
        self.episode_id = episode_id
        self.used_episode_ids: list[int] = []
        self.used_fact_ids: list[int] = []
        self.used_attachment_ids: list[int] = []


class FakeAgent:
    """Records all calls to respond()."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._episode_counter = 1

    async def respond(
        self,
        message: str,
        *,
        episode_id: int | None,
        channel: str,
        attachments: list[AttachmentRef] | None,
    ) -> FakeAgentResponse:
        self.calls.append(
            {
                "message": message,
                "episode_id": episode_id,
                "channel": channel,
                "attachments": attachments,
            }
        )
        return FakeAgentResponse(episode_id=self._episode_counter)


class FailingAgent:
    """Always raises on respond()."""

    async def respond(
        self,
        message: str,
        *,
        episode_id: int | None,
        channel: str,
        attachments: list[AttachmentRef] | None,
    ) -> FakeAgentResponse:
        raise RuntimeError("agent unavailable")


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "repl_attach_test.db")


@pytest.fixture()
def attachments_dir(tmp_path: Path) -> Path:
    d = tmp_path / "attachments"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def fake_agent() -> FakeAgent:
    return FakeAgent()


@pytest.fixture()
def txt_file(tmp_path: Path) -> Path:
    f = tmp_path / "test.txt"
    f.write_text("테스트 내용", encoding="utf-8")
    return f


@pytest.fixture()
def another_txt_file(tmp_path: Path) -> Path:
    f = tmp_path / "test2.txt"
    f.write_text("두 번째 파일", encoding="utf-8")
    return f


def _make_repl(
    agent: Any,
    store: MemoryStore,
    attachments_dir: Path,
    inputs: list[str],
) -> tuple[TextChannel, list[str]]:
    """Build a TextChannel driven by a scripted input list."""
    output: list[str] = []
    input_iter = iter(inputs)

    def fake_input(prompt: str) -> str:
        try:
            return next(input_iter)
        except StopIteration:
            raise EOFError

    return (
        TextChannel(
            agent=agent,
            store=store,
            attachments_dir=attachments_dir,
            input_fn=fake_input,
            output_fn=output.append,
        ),
        output,
    )


# ── tests ─────────────────────────────────────────────────────────────────


class TestAttachQueue:
    @pytest.mark.asyncio
    async def test_attach_queues_one_ref(
        self,
        fake_agent: FakeAgent,
        store: MemoryStore,
        attachments_dir: Path,
        txt_file: Path,
    ) -> None:
        repl, output = _make_repl(
            fake_agent,
            store,
            attachments_dir,
            [f"/attach {txt_file}", "안녕"],
        )
        await repl.run()

        assert len(fake_agent.calls) == 1
        call = fake_agent.calls[0]
        assert call["attachments"] is not None
        assert len(call["attachments"]) == 1
        ref = call["attachments"][0]
        assert isinstance(ref, AttachmentRef)
        assert ref.path.exists()

    @pytest.mark.asyncio
    async def test_second_attach_extends_queue(
        self,
        fake_agent: FakeAgent,
        store: MemoryStore,
        attachments_dir: Path,
        txt_file: Path,
        another_txt_file: Path,
    ) -> None:
        repl, output = _make_repl(
            fake_agent,
            store,
            attachments_dir,
            [
                f"/attach {txt_file}",
                f"/attach {another_txt_file}",
                "두 파일 모두 보내기",
            ],
        )
        await repl.run()

        assert len(fake_agent.calls) == 1
        call = fake_agent.calls[0]
        assert call["attachments"] is not None
        assert len(call["attachments"]) == 2

    @pytest.mark.asyncio
    async def test_queue_cleared_after_message(
        self,
        fake_agent: FakeAgent,
        store: MemoryStore,
        attachments_dir: Path,
        txt_file: Path,
    ) -> None:
        repl, output = _make_repl(
            fake_agent,
            store,
            attachments_dir,
            [f"/attach {txt_file}", "첫 번째 메시지", "두 번째 메시지"],
        )
        await repl.run()

        assert len(fake_agent.calls) == 2
        # First call had attachments.
        assert fake_agent.calls[0]["attachments"] is not None
        # Second call should have no attachments.
        assert fake_agent.calls[1]["attachments"] is None

    @pytest.mark.asyncio
    async def test_message_without_attach_passes_none(
        self,
        fake_agent: FakeAgent,
        store: MemoryStore,
        attachments_dir: Path,
    ) -> None:
        repl, output = _make_repl(
            fake_agent,
            store,
            attachments_dir,
            ["첨부 없는 메시지"],
        )
        await repl.run()

        assert len(fake_agent.calls) == 1
        assert fake_agent.calls[0]["attachments"] is None

    @pytest.mark.asyncio
    async def test_attach_ack_shows_sha256_and_count(
        self,
        fake_agent: FakeAgent,
        store: MemoryStore,
        attachments_dir: Path,
        txt_file: Path,
    ) -> None:
        repl, output = _make_repl(
            fake_agent,
            store,
            attachments_dir,
            [f"/attach {txt_file}", "ok"],
        )
        await repl.run()

        ack_lines = [l for l in output if "sha256" in l]
        assert ack_lines, "Expected sha256 ack in output"
        assert "1개" in ack_lines[0]


class TestClearAttachments:
    @pytest.mark.asyncio
    async def test_clear_empties_queue(
        self,
        fake_agent: FakeAgent,
        store: MemoryStore,
        attachments_dir: Path,
        txt_file: Path,
    ) -> None:
        repl, output = _make_repl(
            fake_agent,
            store,
            attachments_dir,
            [f"/attach {txt_file}", "/clear-attachments", "메시지"],
        )
        await repl.run()

        assert len(fake_agent.calls) == 1
        assert fake_agent.calls[0]["attachments"] is None

    @pytest.mark.asyncio
    async def test_clear_empty_queue_prints_ack(
        self,
        fake_agent: FakeAgent,
        store: MemoryStore,
        attachments_dir: Path,
    ) -> None:
        repl, output = _make_repl(
            fake_agent,
            store,
            attachments_dir,
            ["/clear-attachments"],
        )
        await repl.run()

        clear_lines = [l for l in output if "없습니다" in l or "취소" in l]
        assert clear_lines

    @pytest.mark.asyncio
    async def test_clear_with_files_reports_count(
        self,
        fake_agent: FakeAgent,
        store: MemoryStore,
        attachments_dir: Path,
        txt_file: Path,
    ) -> None:
        repl, output = _make_repl(
            fake_agent,
            store,
            attachments_dir,
            [f"/attach {txt_file}", "/clear-attachments"],
        )
        await repl.run()

        clear_lines = [l for l in output if "취소" in l]
        assert clear_lines
        assert "1" in clear_lines[0]


class TestAttachFailure:
    @pytest.mark.asyncio
    async def test_missing_file_leaves_queue_unchanged(
        self,
        fake_agent: FakeAgent,
        store: MemoryStore,
        attachments_dir: Path,
        txt_file: Path,
    ) -> None:
        repl, output = _make_repl(
            fake_agent,
            store,
            attachments_dir,
            [
                f"/attach {txt_file}",                    # succeeds
                "/attach /nonexistent/missing.txt",       # fails
                "메시지 전송",
            ],
        )
        await repl.run()

        # Only the first (successful) attachment should be forwarded.
        assert len(fake_agent.calls) == 1
        assert fake_agent.calls[0]["attachments"] is not None
        assert len(fake_agent.calls[0]["attachments"]) == 1

    @pytest.mark.asyncio
    async def test_attach_bad_ext_error_message(
        self,
        fake_agent: FakeAgent,
        store: MemoryStore,
        attachments_dir: Path,
        tmp_path: Path,
    ) -> None:
        bad = tmp_path / "script.sh"
        bad.write_text("#!/bin/bash", encoding="utf-8")
        repl, output = _make_repl(
            fake_agent,
            store,
            attachments_dir,
            [f"/attach {bad}"],
        )
        await repl.run()

        error_lines = [l for l in output if "오류" in l]
        assert error_lines
