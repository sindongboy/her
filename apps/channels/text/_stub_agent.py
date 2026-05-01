"""Stub agent used when AgentCore is not yet available.

Only for development bootstrapping — not tested, not shipped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StubResponse:
    text: str
    episode_id: int | None = None
    used_episode_ids: list[int] = ()  # type: ignore[assignment]
    used_fact_ids: list[int] = ()  # type: ignore[assignment]


class StubAgent:
    """Returns a placeholder message. Replace with real AgentCore."""

    async def respond(
        self,
        message: str,
        *,
        episode_id: int | None,
        channel: str,
    ) -> StubResponse:
        return StubResponse(
            text=(
                "[StubAgent] AgentCore 가 아직 준비되지 않았습니다. "
                "apps/agent/core.py 를 구현한 후 다시 시도하세요."
            ),
            episode_id=episode_id,
        )
