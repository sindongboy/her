"""Unit tests for apps/settings/consent.py."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

import pytest

from apps.settings.consent import (
    _DISCLOSURE_LINES,
    grant_mic_consent,
    is_consent_granted,
    prompt_mic_consent,
)
from apps.settings.store import Settings


# ── is_consent_granted ─────────────────────────────────────────────────────

class TestIsConsentGranted:
    def test_false_by_default(self) -> None:
        s = Settings()
        assert is_consent_granted(s) is False

    def test_true_when_field_set(self) -> None:
        from dataclasses import replace

        s = replace(Settings(), mic_consent_granted=True)
        assert is_consent_granted(s) is True

    def test_respects_false_explicitly(self) -> None:
        from dataclasses import replace

        s = replace(Settings(), mic_consent_granted=False)
        assert is_consent_granted(s) is False


# ── grant_mic_consent ──────────────────────────────────────────────────────

class TestGrantMicConsent:
    def test_returns_new_settings_with_consent_true(self) -> None:
        s = Settings()
        updated = grant_mic_consent(s)
        assert updated.mic_consent_granted is True

    def test_original_unchanged(self) -> None:
        s = Settings()
        grant_mic_consent(s)
        assert s.mic_consent_granted is False  # pure function

    def test_mic_consent_at_is_iso_timestamp(self) -> None:
        s = Settings()
        updated = grant_mic_consent(s)
        assert updated.mic_consent_at is not None
        # Should be parseable as an ISO datetime
        parsed = datetime.fromisoformat(updated.mic_consent_at)
        assert parsed.year >= 2026

    def test_custom_when_preserved(self) -> None:
        s = Settings()
        fixed_time = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
        updated = grant_mic_consent(s, when=fixed_time)
        assert updated.mic_consent_at is not None
        assert "2026-05-01" in updated.mic_consent_at

    def test_other_fields_preserved(self) -> None:
        from dataclasses import replace

        s = replace(Settings(), quiet_mode=True, wake_keyword="hey her", daily_proactive_limit=7)
        updated = grant_mic_consent(s)
        assert updated.quiet_mode is True
        assert updated.wake_keyword == "hey her"
        assert updated.daily_proactive_limit == 7


# ── prompt_mic_consent ─────────────────────────────────────────────────────

class TestPromptMicConsent:
    """Test the interactive consent prompt with stub I/O."""

    @staticmethod
    def _make_io(answer: str) -> tuple[list[str], list[str]]:
        """Returns (output_lines, []) — caller fills in output lines."""
        return [], []

    def _run(self, answer: str) -> tuple[bool, str]:
        """Run prompt with stub input, capture output."""
        output_lines: list[str] = []
        result = prompt_mic_consent(
            input_fn=lambda _: answer,
            output_fn=lambda *args, **kwargs: output_lines.append(" ".join(str(a) for a in args)),
        )
        return result, "\n".join(output_lines)

    # ── acceptance ────────────────────────────────────────────────────────

    def test_dongi_accepted(self) -> None:
        ok, _ = self._run("동의")
        assert ok is True

    def test_yes_accepted(self) -> None:
        ok, _ = self._run("yes")
        assert ok is True

    def test_y_accepted(self) -> None:
        ok, _ = self._run("y")
        assert ok is True

    def test_case_insensitive_YES(self) -> None:
        ok, _ = self._run("YES")
        assert ok is True

    def test_case_insensitive_Y_upper(self) -> None:
        ok, _ = self._run("Y")
        assert ok is True

    # ── denial ────────────────────────────────────────────────────────────

    def test_chwisohanmeyo_denied(self) -> None:
        ok, _ = self._run("취소")
        assert ok is False

    def test_n_denied(self) -> None:
        ok, _ = self._run("n")
        assert ok is False

    def test_no_denied(self) -> None:
        ok, _ = self._run("no")
        assert ok is False

    def test_empty_input_denied(self) -> None:
        ok, _ = self._run("")
        assert ok is False

    def test_random_string_denied(self) -> None:
        ok, _ = self._run("어쩌고저쩌고")
        assert ok is False

    # ── disclosure text shown ─────────────────────────────────────────────

    def test_disclosure_lines_shown(self) -> None:
        _, output = self._run("동의")
        for line in _DISCLOSURE_LINES:
            assert line in output

    def test_disclosure_shown_even_on_denial(self) -> None:
        _, output = self._run("n")
        for line in _DISCLOSURE_LINES:
            assert line in output

    # ── EOFError / KeyboardInterrupt handling ─────────────────────────────

    def test_eoferror_returns_false(self) -> None:
        output_lines: list[str] = []

        def raise_eof(prompt: str) -> str:
            raise EOFError

        result = prompt_mic_consent(
            input_fn=raise_eof,
            output_fn=lambda *a, **kw: output_lines.append(" ".join(str(x) for x in a)),
        )
        assert result is False

    def test_keyboard_interrupt_returns_false(self) -> None:
        output_lines: list[str] = []

        def raise_kb(prompt: str) -> str:
            raise KeyboardInterrupt

        result = prompt_mic_consent(
            input_fn=raise_kb,
            output_fn=lambda *a, **kw: output_lines.append(" ".join(str(x) for x in a)),
        )
        assert result is False
