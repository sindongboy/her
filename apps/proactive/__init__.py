"""Proactive Engine — fires proactive utterances based on triggers.

CLAUDE.md §2.4: time-based, silence, recurring-pattern, and (stub) external.
"""

from apps.proactive.engine import ProactiveConfig, ProactiveEngine
from apps.proactive.triggers import (
    RecurringPatternTrigger,
    SilenceTrigger,
    TimeBasedTrigger,
    Trigger,
    TriggerProposal,
)

__all__ = [
    "ProactiveConfig",
    "ProactiveEngine",
    "RecurringPatternTrigger",
    "SilenceTrigger",
    "TimeBasedTrigger",
    "Trigger",
    "TriggerProposal",
]
