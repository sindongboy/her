"""Unit tests for Phase 2 JSONL loader extensions in scripts/replay.py.

Covers:
1. Loader parses 'attachments' field; absent field → empty list.
2. Path resolution: fixture at tests/fixtures/dialog_002_attachment.jsonl with
   attachment path 'sample_attachment.txt' resolves to
   tests/fixtures/sample_attachment.txt.
3. Malformed 'attachments' (not a list) raises ValueError with line number.
4. Missing referenced file → loader returns the Turn (no existence check at
   parse time; replay warns at runtime).
5. Backward compat: dialog_001.jsonl still loads with zero attachments per turn.
6. AttachmentRef objects are produced correctly: path, description, mime defaults.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.replay import AttachmentRef, Turn, load_turns  # noqa: E402

_FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures"


# ── helpers ────────────────────────────────────────────────────────────────────


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.jsonl"
    p.write_text(content, encoding="utf-8")
    return p


# ── 1. attachments field parsing ──────────────────────────────────────────────


class TestAttachmentsField:
    def test_absent_attachments_gives_empty_list(self, tmp_path: Path) -> None:
        """Turns without 'attachments' field get an empty list (Phase 0 compat)."""
        content = '{"role": "user", "channel": "text", "text": "안녕?"}\n'
        turns = load_turns(_write(tmp_path, content))
        assert len(turns) == 1
        assert turns[0].attachments == []

    def test_present_attachments_parsed(self, tmp_path: Path) -> None:
        """'attachments' field is parsed into a list of AttachmentRef."""
        att_file = tmp_path / "note.txt"
        att_file.write_text("내용", encoding="utf-8")
        obj = {
            "role": "user",
            "channel": "text",
            "text": "봐줘",
            "attachments": [{"path": "note.txt", "description": "메모"}],
        }
        p = _write(tmp_path, json.dumps(obj, ensure_ascii=False))
        turns = load_turns(p)
        assert len(turns[0].attachments) == 1
        att = turns[0].attachments[0]
        assert isinstance(att, AttachmentRef)
        assert att.description == "메모"

    def test_multiple_attachments(self, tmp_path: Path) -> None:
        """Multiple attachment entries in a single turn are all parsed."""
        obj = {
            "role": "user",
            "channel": "text",
            "text": "두 파일",
            "attachments": [
                {"path": "a.txt"},
                {"path": "b.txt", "description": "두 번째"},
            ],
        }
        p = _write(tmp_path, json.dumps(obj, ensure_ascii=False))
        turns = load_turns(p)
        assert len(turns[0].attachments) == 2

    def test_empty_attachments_list(self, tmp_path: Path) -> None:
        """Empty attachments array gives empty list (not an error)."""
        obj = {"role": "user", "text": "hi", "attachments": []}
        p = _write(tmp_path, json.dumps(obj))
        turns = load_turns(p)
        assert turns[0].attachments == []


# ── 2. path resolution ────────────────────────────────────────────────────────


class TestPathResolution:
    def test_relative_path_resolved_against_fixture_dir(self, tmp_path: Path) -> None:
        """Relative attachment paths are resolved against the fixture file's parent."""
        obj = {
            "role": "user",
            "text": "봐줘",
            "attachments": [{"path": "sample.txt"}],
        }
        fixture = tmp_path / "fixtures" / "dialog.jsonl"
        fixture.parent.mkdir(parents=True)
        fixture.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

        turns = load_turns(fixture)
        expected = (tmp_path / "fixtures" / "sample.txt").resolve()
        assert turns[0].attachments[0].path == expected

    def test_dialog_002_attachment_paths_resolve_correctly(self) -> None:
        """dialog_002_attachment.jsonl attachment paths resolve inside fixtures/."""
        fixture = _FIXTURES_DIR / "dialog_002_attachment.jsonl"
        if not fixture.exists():
            pytest.skip("dialog_002_attachment.jsonl not yet present")
        turns = load_turns(fixture)
        # First turn has an attachment
        first_user = next(t for t in turns if t.role == "user")
        assert len(first_user.attachments) == 1
        resolved = first_user.attachments[0].path
        assert resolved == (_FIXTURES_DIR / "sample_attachment.txt").resolve()

    def test_absolute_path_kept_as_is(self, tmp_path: Path) -> None:
        """An absolute path in the fixture stays absolute (just resolved)."""
        abs_path = str((tmp_path / "absolute.txt").resolve())
        obj = {"role": "user", "text": "x", "attachments": [{"path": abs_path}]}
        p = _write(tmp_path, json.dumps(obj))
        turns = load_turns(p)
        assert turns[0].attachments[0].path == Path(abs_path).resolve()


# ── 3. malformed attachments raises with line number ─────────────────────────


class TestMalformedAttachments:
    def test_attachments_not_a_list_raises(self, tmp_path: Path) -> None:
        """'attachments' that is not a list raises ValueError with line number."""
        obj = {"role": "user", "text": "hi", "attachments": "not-a-list"}
        p = _write(tmp_path, json.dumps(obj))
        with pytest.raises(ValueError, match="Line 1"):
            load_turns(p)

    def test_attachments_dict_raises(self, tmp_path: Path) -> None:
        """'attachments' as a bare dict (not list) raises ValueError."""
        obj = {"role": "user", "text": "hi", "attachments": {"path": "x.txt"}}
        p = _write(tmp_path, json.dumps(obj))
        with pytest.raises(ValueError, match="Line 1"):
            load_turns(p)

    def test_malformed_on_second_line_reports_line_2(self, tmp_path: Path) -> None:
        """Line number in error reflects the actual line with bad 'attachments'."""
        line1 = json.dumps({"role": "user", "text": "ok"})
        line2 = json.dumps({"role": "user", "text": "bad", "attachments": 42})
        content = f"{line1}\n{line2}\n"
        p = _write(tmp_path, content)
        with pytest.raises(ValueError, match="Line 2"):
            load_turns(p)

    def test_attachment_entry_not_dict_raises(self, tmp_path: Path) -> None:
        """Attachment list entries that are not dicts raise ValueError."""
        obj = {"role": "user", "text": "hi", "attachments": ["not-a-dict"]}
        p = _write(tmp_path, json.dumps(obj))
        with pytest.raises(ValueError, match="Line 1"):
            load_turns(p)


# ── 4. missing file does not raise at parse time ──────────────────────────────


class TestMissingFile:
    def test_missing_attachment_file_does_not_raise_on_load(
        self, tmp_path: Path
    ) -> None:
        """File existence is NOT checked during load_turns (replay warns at runtime)."""
        obj = {
            "role": "user",
            "text": "봐줘",
            "attachments": [{"path": "nonexistent_file_xyz.txt"}],
        }
        p = _write(tmp_path, json.dumps(obj, ensure_ascii=False))
        # Must not raise
        turns = load_turns(p)
        assert len(turns) == 1
        assert len(turns[0].attachments) == 1
        # Path is set even though file doesn't exist
        assert turns[0].attachments[0].path.name == "nonexistent_file_xyz.txt"


# ── 5. backward compatibility ─────────────────────────────────────────────────


class TestBackwardCompat:
    def test_dialog_001_loads_with_zero_attachments(self) -> None:
        """Phase 0 fixture dialog_001.jsonl loads without error; all turns have empty attachments."""
        fixture = _FIXTURES_DIR / "dialog_001.jsonl"
        if not fixture.exists():
            pytest.skip("dialog_001.jsonl not yet present")
        turns = load_turns(fixture)
        assert len(turns) > 0
        for turn in turns:
            assert turn.attachments == [], f"Expected no attachments on turn: {turn}"

    def test_no_attachments_key_means_empty_list(self, tmp_path: Path) -> None:
        """Fixtures with no 'attachments' key always get default empty list."""
        content = (
            '{"role": "user", "channel": "text", "text": "hello"}\n'
            '{"role": "user", "channel": "voice", "text": "world"}\n'
        )
        turns = load_turns(_write(tmp_path, content))
        for t in turns:
            assert t.attachments == []


# ── 6. AttachmentRef field correctness ───────────────────────────────────────


class TestAttachmentRefFields:
    def test_path_field_is_path_object(self, tmp_path: Path) -> None:
        """AttachmentRef.path is a resolved Path, not a raw string."""
        obj = {
            "role": "user",
            "text": "x",
            "attachments": [{"path": "test.txt"}],
        }
        p = _write(tmp_path, json.dumps(obj))
        turns = load_turns(p)
        att = turns[0].attachments[0]
        assert isinstance(att.path, Path)

    def test_description_set_from_fixture(self, tmp_path: Path) -> None:
        """'description' key in fixture maps to AttachmentRef.description."""
        obj = {
            "role": "user",
            "text": "x",
            "attachments": [{"path": "f.txt", "description": "설명입니다"}],
        }
        p = _write(tmp_path, json.dumps(obj, ensure_ascii=False))
        turns = load_turns(p)
        assert turns[0].attachments[0].description == "설명입니다"

    def test_mime_defaults_to_none(self, tmp_path: Path) -> None:
        """Attachment without 'mime' key gets mime=None."""
        obj = {"role": "user", "text": "x", "attachments": [{"path": "f.txt"}]}
        p = _write(tmp_path, json.dumps(obj))
        turns = load_turns(p)
        assert turns[0].attachments[0].mime is None

    def test_sha256_defaults_to_none(self, tmp_path: Path) -> None:
        """Attachment without 'sha256' key gets sha256=None."""
        obj = {"role": "user", "text": "x", "attachments": [{"path": "f.txt"}]}
        p = _write(tmp_path, json.dumps(obj))
        turns = load_turns(p)
        assert turns[0].attachments[0].sha256 is None

    def test_description_defaults_to_none(self, tmp_path: Path) -> None:
        """Attachment without 'description' key gets description=None."""
        obj = {"role": "user", "text": "x", "attachments": [{"path": "f.txt"}]}
        p = _write(tmp_path, json.dumps(obj))
        turns = load_turns(p)
        assert turns[0].attachments[0].description is None

    def test_mime_set_from_fixture(self, tmp_path: Path) -> None:
        """'mime' key in fixture maps to AttachmentRef.mime."""
        obj = {
            "role": "user",
            "text": "x",
            "attachments": [{"path": "f.pdf", "mime": "application/pdf"}],
        }
        p = _write(tmp_path, json.dumps(obj))
        turns = load_turns(p)
        assert turns[0].attachments[0].mime == "application/pdf"

    def test_sha256_set_from_fixture(self, tmp_path: Path) -> None:
        """'sha256' key in fixture maps to AttachmentRef.sha256."""
        digest = "abc123"
        obj = {
            "role": "user",
            "text": "x",
            "attachments": [{"path": "f.txt", "sha256": digest}],
        }
        p = _write(tmp_path, json.dumps(obj))
        turns = load_turns(p)
        assert turns[0].attachments[0].sha256 == digest

    def test_dialog_002_first_turn_attachment_ref(self) -> None:
        """dialog_002_attachment.jsonl first turn has correct AttachmentRef."""
        fixture = _FIXTURES_DIR / "dialog_002_attachment.jsonl"
        if not fixture.exists():
            pytest.skip("dialog_002_attachment.jsonl not yet present")
        turns = load_turns(fixture)
        first = turns[0]
        assert len(first.attachments) == 1
        att = first.attachments[0]
        assert att.path == (_FIXTURES_DIR / "sample_attachment.txt").resolve()
        assert att.description == "단호박 케이크 메모"
        assert att.mime is None

    def test_dialog_003_has_no_attachments(self) -> None:
        """dialog_003_followup.jsonl turns have no attachments (recall-only scenario)."""
        fixture = _FIXTURES_DIR / "dialog_003_followup.jsonl"
        if not fixture.exists():
            pytest.skip("dialog_003_followup.jsonl not yet present")
        turns = load_turns(fixture)
        assert len(turns) > 0
        for t in turns:
            assert t.attachments == []
