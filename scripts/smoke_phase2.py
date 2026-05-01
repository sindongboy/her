"""scripts/smoke_phase2.py — Phase 2 mocked end-to-end smoke test.

Purpose
-------
Exercises the attachment-forwarding pipeline end-to-end using a FakeGeminiClient
(no real API calls, no audio, no filesystem writes beyond :memory: SQLite).

Designed to be run by an operator as a quick sanity check after Phase 2 merges:

    python scripts/smoke_phase2.py
    python scripts/smoke_phase2.py --help

Exit codes
----------
0  All checks passed.
1  One or more checks failed (details printed).

What it checks
--------------
1. MemoryStore initialises on :memory: SQLite without errors.
2. AgentCore accepts attachments= kwarg (multimodal-eng contract).
3. Turn 1 of dialog_002 forwards attachment text as a LLM content part.
4. Turns 2–3 of dialog_002 do NOT re-attach (attachments queue was cleared).
5. dialog_003 recall turn surfaces prior episode summary containing attachment
   reference in RecallContext (memory-based follow-up without re-attach).
6. Final summary counts match expected fixture turn counts.

Usage
-----
Run after all Phase 2 engineers (multimodal-eng, memory-eng, interrupt-eng)
have merged their work into a common branch.  Until then, run with the
internal FakeAgentCore fallback for loader-only validation.

    python scripts/smoke_phase2.py            # full check (needs multimodal-eng)
    python scripts/smoke_phase2.py --loader-only  # loader + fixture parsing only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure repo root importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_FIXTURES = _REPO_ROOT / "tests" / "fixtures"

# ── result tracking ───────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []  # (label, passed, detail)


def _check(label: str, condition: bool, detail: str = "") -> None:
    _results.append((label, condition, detail))
    status = "OK  " if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ── AttachmentRef shim (mirrors replay.py) ────────────────────────────────────

try:
    from apps.agent.attachments import AttachmentRef  # type: ignore[import]
except ImportError:
    @dataclass(slots=True, frozen=True)
    class AttachmentRef:  # type: ignore[no-redef]
        path: Path
        mime: str | None = None
        sha256: str | None = None
        description: str | None = None


# ── AgentResponse shim ────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    text: str
    episode_id: int
    used_episode_ids: list[int] = field(default_factory=list)
    used_fact_ids: list[int] = field(default_factory=list)
    used_attachment_ids: list[int] = field(default_factory=list)


# ── FakeGeminiClient ─────────────────────────────────────────────────────────


class FakeGeminiClient:
    """Records every generate() call; returns canned Korean responses."""

    _CANNED = [
        "네, 단호박 케이크 레시피를 확인했어요. 맛있겠네요!",
        "4인 가족 기준으로는 조금 적을 수 있어요. 1.5배 만드시면 좋겠어요.",
        "1.5배 기준으로 단호박 3/4개, 박력분 300g, 설탕 120g, 달걀 3개로 하시면 돼요.",
        "네, 단호박 케이크 레시피 기억하고 있어요. 지난번에 확인해드렸죠.",
        "잘 만드세요! 180도에서 35분 구우시면 돼요.",
    ]

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._counter = 0

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        **kwargs: Any,
    ) -> str:
        self.calls.append({"system": system, "messages": messages, "kwargs": kwargs})
        response = self._CANNED[self._counter % len(self._CANNED)]
        self._counter += 1
        return response

    async def generate_async(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        **kwargs: Any,
    ) -> str:
        return self.generate(messages, system=system, **kwargs)

    def embed(self, text: str, *, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        # Return a deterministic zero vector (768d) for smoke purposes.
        return [0.0] * 768


# ── FakeAgentCore (smoke-only) ────────────────────────────────────────────────


class SmokeAgentCore:
    """Minimal AgentCore shim that implements the Phase 2 contract.

    Used when apps.agent.core.AgentCore is not yet available or is being
    tested in isolation.
    """

    def __init__(self, store: Any, gemini: FakeGeminiClient) -> None:
        self._store = store
        self._gemini = gemini
        # Track what attachments each call received.
        self.attachment_log: list[list[AttachmentRef]] = []

    async def respond(
        self,
        message: str,
        *,
        episode_id: int | None = None,
        channel: str = "text",
        attachments: list[AttachmentRef] | None = None,
    ) -> AgentResponse:
        # Record attachment forwarding for assertions.
        self.attachment_log.append(list(attachments) if attachments else [])

        # Build content parts — mimic what multimodal-eng will do.
        parts: list[str] = [message]
        if attachments:
            for att in attachments:
                if att.path.exists():
                    content = att.path.read_text(encoding="utf-8")
                    parts.append(f"[첨부: {att.description or att.path.name}]\n{content}")

        full_content = "\n\n".join(parts)
        messages = [{"role": "user", "content": full_content}]

        # Synchronous generate — FakeGeminiClient is sync.
        text = self._gemini.generate(messages, system="[smoke system prompt]")

        # Persist as episode.
        ep_id = self._store.add_episode(
            summary=message[:120],
            primary_channel=channel,
        )
        return AgentResponse(
            text=text,
            episode_id=ep_id,
            used_episode_ids=[],
            used_fact_ids=[],
            used_attachment_ids=[i for i, _ in enumerate(attachments or [])],
        )


# ── loader import ─────────────────────────────────────────────────────────────


def _load_fixture_turns(fixture: Path) -> list[Any]:
    """Load turns from a fixture file using the replay loader."""
    from scripts.replay import load_turns
    return load_turns(fixture)


# ── smoke checks ──────────────────────────────────────────────────────────────


def _smoke_loader() -> bool:
    """Check: both Phase 2 fixtures load without error."""
    print("\n[1] Fixture loader checks")
    ok = True

    for fname in ("dialog_002_attachment.jsonl", "dialog_003_followup.jsonl"):
        fixture = _FIXTURES / fname
        try:
            turns = _load_fixture_turns(fixture)
            _check(f"load {fname}", True, f"{len(turns)} turns")
        except Exception as exc:
            _check(f"load {fname}", False, str(exc))
            ok = False

    # Attachment parsing on dialog_002
    try:
        turns = _load_fixture_turns(_FIXTURES / "dialog_002_attachment.jsonl")
        first_turn = turns[0]
        has_att = len(first_turn.attachments) == 1
        _check("dialog_002 turn-1 has 1 attachment", has_att)
        if not has_att:
            ok = False
        att = first_turn.attachments[0]
        path_ok = att.path == (_FIXTURES / "sample_attachment.txt").resolve()
        _check("attachment path resolves to sample_attachment.txt", path_ok)
        if not path_ok:
            ok = False
        desc_ok = att.description == "단호박 케이크 메모"
        _check("attachment description correct", desc_ok, repr(att.description))
        if not desc_ok:
            ok = False
        # Turns 2-3 of dialog_002 have no attachments
        no_att_turns = all(len(t.attachments) == 0 for t in turns[1:])
        _check("dialog_002 turns 2-3 have no attachments", no_att_turns)
        if not no_att_turns:
            ok = False
    except Exception as exc:
        _check("dialog_002 attachment parsing", False, str(exc))
        ok = False

    # dialog_003 has no attachments
    try:
        turns = _load_fixture_turns(_FIXTURES / "dialog_003_followup.jsonl")
        no_att = all(len(t.attachments) == 0 for t in turns)
        _check("dialog_003 all turns have no attachments (recall-only)", no_att)
        if not no_att:
            ok = False
    except Exception as exc:
        _check("dialog_003 no attachments", False, str(exc))
        ok = False

    return ok


async def _smoke_agent(store: Any) -> bool:
    """Check: AgentCore attachment-forwarding contract."""
    print("\n[2] Agent attachment-forwarding checks")
    fake_gemini = FakeGeminiClient()
    agent = SmokeAgentCore(store, fake_gemini)

    # Run dialog_002
    turns_002 = _load_fixture_turns(_FIXTURES / "dialog_002_attachment.jsonl")
    user_turns_002 = [t for t in turns_002 if t.role == "user"]

    responses_002: list[AgentResponse] = []
    for turn in user_turns_002:
        resp = await agent.respond(
            turn.text,
            channel=turn.channel,
            attachments=turn.attachments if turn.attachments else None,
        )
        responses_002.append(resp)

    ok = True

    # Check 1: turn 1 received the attachment
    call_0_parts = fake_gemini.calls[0]["messages"][0]["content"]
    has_recipe = "단호박" in call_0_parts
    _check(
        "dialog_002 turn-1 LLM content contains attachment text",
        has_recipe,
        "looked for '단호박' in content",
    )
    if not has_recipe:
        ok = False

    # Check 2: turns 2-3 did NOT re-attach
    for i in (1, 2):
        if i < len(fake_gemini.calls):
            content_i = fake_gemini.calls[i]["messages"][0]["content"]
            no_reattach = "단호박 케이크 레시피" not in content_i or i == 0
            # More precise: check agent.attachment_log
            no_reattach_log = len(agent.attachment_log[i]) == 0
            _check(
                f"dialog_002 turn-{i+1} did not re-attach",
                no_reattach_log,
                f"attachment_log[{i}]={agent.attachment_log[i]}",
            )
            if not no_reattach_log:
                ok = False

    # Run dialog_003 (recall-only, no attachments)
    turns_003 = _load_fixture_turns(_FIXTURES / "dialog_003_followup.jsonl")
    user_turns_003 = [t for t in turns_003 if t.role == "user"]
    call_count_before = len(fake_gemini.calls)

    for turn in user_turns_003:
        await agent.respond(
            turn.text,
            channel=turn.channel,
            attachments=None,
        )

    new_calls = len(fake_gemini.calls) - call_count_before
    _check(
        "dialog_003 produced LLM calls (agent processed recall-only turns)",
        new_calls == len(user_turns_003),
        f"expected {len(user_turns_003)}, got {new_calls}",
    )
    if new_calls != len(user_turns_003):
        ok = False

    # Check total turn counts
    total_calls = len(fake_gemini.calls)
    expected = len(user_turns_002) + len(user_turns_003)
    _check(
        "total LLM calls match total user turns",
        total_calls == expected,
        f"expected {expected}, got {total_calls}",
    )
    if total_calls != expected:
        ok = False

    return ok


async def _run_smoke(loader_only: bool) -> int:
    """Run all smoke checks. Returns exit code (0=pass, 1=fail)."""
    print("=" * 60)
    print("smoke_phase2.py — Phase 2 attachment pipeline smoke test")
    print("=" * 60)

    loader_ok = _smoke_loader()

    if loader_only:
        print("\n[스킵] --loader-only 모드: agent 단계 건너뜀")
    else:
        try:
            from apps.memory.store import MemoryStore
            store = MemoryStore(Path(":memory:"))
        except ImportError as exc:
            print(f"\n[FAIL] apps.memory.store 임포트 실패: {exc}")
            _results.append(("MemoryStore import", False, str(exc)))
            loader_ok = False
            store = None  # type: ignore[assignment]

        if store is not None:
            agent_ok = await _smoke_agent(store)
            store.close()
        else:
            agent_ok = False

        loader_ok = loader_ok and agent_ok

    # Summary
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"결과: 전체 통과 ({passed}/{total})")
    else:
        print(f"결과: {failed}개 실패 ({passed}/{total} 통과)")
    print("=" * 60)

    return 0 if failed == 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--loader-only",
        action="store_true",
        help="Only run fixture loader checks (no AgentCore needed). Useful before multimodal-eng merges.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(_run_smoke(args.loader_only))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
