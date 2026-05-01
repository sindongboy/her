"""Consolidator package — daily batch memory consolidation (CLAUDE.md §5.3).

Usage:
    from apps.consolidator import run_consolidation, ConsolidationReport
"""

from apps.consolidator.runner import ConsolidationReport, run_consolidation

__all__ = ["run_consolidation", "ConsolidationReport"]
