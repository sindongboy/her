"""Agent Core — public surface.

Exports:
    AgentCore      — main orchestrator; call respond() or stream_respond().
    AgentResponse  — returned by respond().
    RecallContext  — passed to callers who need to inspect recalled memory.
    AttachmentRef  — stable handle for an on-disk attachment passed to respond().
"""

from apps.agent.core import AgentCore, AgentResponse, AttachmentRef
from apps.agent.recall import RecallContext

__all__ = ["AgentCore", "AgentResponse", "AttachmentRef", "RecallContext"]
