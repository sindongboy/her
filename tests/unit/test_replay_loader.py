"""Unit tests for the JSONL loader in scripts/replay.py.

Covers:
- valid fixture file → list of Turn objects
- malformed JSON line → ValueError with line number
- empty file → empty list
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.replay import Turn, load_turns  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.jsonl"
    p.write_text(content, encoding="utf-8")
    return p


# ── tests ─────────────────────────────────────────────────────────────────────


class TestLoadTurns:
    def test_valid_fixture(self, tmp_path: Path) -> None:
        content = (
            '{"role": "user", "channel": "text", "text": "안녕?"}\n'
            '{"role": "user", "channel": "voice", "text": "케이크 사야 돼."}\n'
        )
        turns = load_turns(_write(tmp_path, content))
        assert len(turns) == 2
        assert isinstance(turns[0], Turn)
        assert turns[0].role == "user"
        assert turns[0].channel == "text"
        assert turns[0].text == "안녕?"
        assert turns[1].channel == "voice"

    def test_default_channel_is_text(self, tmp_path: Path) -> None:
        """Lines without 'channel' field default to 'text'."""
        content = '{"role": "user", "text": "hello"}\n'
        turns = load_turns(_write(tmp_path, content))
        assert turns[0].channel == "text"

    def test_raw_dict_preserved(self, tmp_path: Path) -> None:
        """The raw dict is stored on the Turn for future extension."""
        content = '{"role": "user", "text": "hi", "extra": 42}\n'
        turns = load_turns(_write(tmp_path, content))
        assert turns[0].raw["extra"] == 42

    def test_malformed_json_raises_with_line_number(self, tmp_path: Path) -> None:
        content = (
            '{"role": "user", "text": "ok"}\n'
            'NOT JSON\n'
            '{"role": "user", "text": "also ok"}\n'
        )
        with pytest.raises(ValueError, match="Line 2"):
            load_turns(_write(tmp_path, content))

    def test_missing_role_raises_with_line_number(self, tmp_path: Path) -> None:
        content = '{"text": "no role here"}\n'
        with pytest.raises(ValueError, match="Line 1"):
            load_turns(_write(tmp_path, content))

    def test_missing_text_raises_with_line_number(self, tmp_path: Path) -> None:
        content = '{"role": "user"}\n'
        with pytest.raises(ValueError, match="Line 1"):
            load_turns(_write(tmp_path, content))

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        turns = load_turns(_write(tmp_path, ""))
        assert turns == []

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        content = (
            "\n"
            '{"role": "user", "text": "hello"}\n'
            "\n"
            '{"role": "user", "text": "world"}\n'
            "\n"
        )
        turns = load_turns(_write(tmp_path, content))
        assert len(turns) == 2

    def test_malformed_line_error_reports_correct_line(self, tmp_path: Path) -> None:
        """Blank lines are counted in the line number for accurate reporting."""
        content = (
            '{"role": "user", "text": "line 1"}\n'
            "\n"
            "bad json here\n"
        )
        with pytest.raises(ValueError, match="Line 3"):
            load_turns(_write(tmp_path, content))

    def test_actual_fixture_file_parses(self) -> None:
        """The committed dialog_001.jsonl must parse without errors."""
        fixture = _REPO_ROOT / "tests" / "fixtures" / "dialog_001.jsonl"
        if not fixture.exists():
            pytest.skip("dialog_001.jsonl not yet present")
        turns = load_turns(fixture)
        assert len(turns) > 0
        for turn in turns:
            assert turn.role in {"user", "assistant"}
            assert turn.channel in {"text", "voice", "mixed"}
            assert turn.text
