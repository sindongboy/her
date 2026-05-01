"""Anonymization at the LLM boundary.

Per CLAUDE.md §2.3: real names exist only inside Memory Layer.
At the Agent Core → LLM adapter boundary, names are replaced with stable
aliases (P001, P002, …) before being sent to Gemini. The response is
de-anonymized before being returned to the caller.

Alias mapping is rebuilt per request — stable within a single exchange,
but not persisted. This is intentional for Phase 0.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from apps.memory.store import Person


class Anonymizer:
    """Builds an alias map from a snapshot of people and performs redaction/restore."""

    def __init__(self, people: Sequence[Person]) -> None:
        # Sort by name length descending to avoid prefix collisions.
        # E.g. "할머니" must be replaced before "머니" if they both exist.
        self._people = sorted(people, key=lambda p: len(p.name), reverse=True)

    # ── public ──────────────────────────────────────────────────────────

    def redact(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace person names with aliases.

        Returns:
            (redacted_text, alias_to_real_map)
            The map is required to call restore() later.
        """
        alias_to_real: dict[str, str] = {}
        real_to_alias: dict[str, str] = {}
        counter = 1

        result = text
        for person in self._people:
            name = person.name
            if not name or name not in result:
                continue
            if name not in real_to_alias:
                alias = f"P{counter:03d}"
                real_to_alias[name] = alias
                alias_to_real[alias] = name
                counter += 1
            result = result.replace(name, real_to_alias[name])

        return result, alias_to_real

    def restore(self, text: str, alias_to_real: dict[str, str]) -> str:
        """Replace aliases back with real names.

        Applies longest alias first to avoid partial replacements
        (e.g. P001 inside P0010).
        """
        if not alias_to_real:
            return text

        result = text
        # Sort by alias length descending for safe multi-pass replace.
        sorted_aliases = sorted(alias_to_real.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            real = alias_to_real[alias]
            # Use word-boundary-aware replacement so "P001s" isn't touched.
            result = re.sub(re.escape(alias), real, result)
        return result

    # ── convenience ─────────────────────────────────────────────────────

    @staticmethod
    def build_alias_block(alias_to_real: dict[str, str]) -> str:
        """Return a compact string listing active aliases for debugging."""
        lines = [f"{alias}={real}" for alias, real in sorted(alias_to_real.items())]
        return ", ".join(lines)
