"""Unit tests for apps/settings/store.py."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

from apps.settings.store import Settings, load_settings, save_settings, settings_path


# ── Helpers ────────────────────────────────────────────────────────────────

def _round_trip(tmp_path: Path, s: Settings) -> Settings:
    target = tmp_path / "settings.toml"
    save_settings(s, target)
    return load_settings(target)


# ── Tests ──────────────────────────────────────────────────────────────────

class TestLoadDefaults:
    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.toml"
        s = load_settings(missing)
        defaults = Settings()
        assert s.mic_consent_granted == defaults.mic_consent_granted
        assert s.quiet_mode == defaults.quiet_mode
        assert s.wake_keyword == defaults.wake_keyword
        assert s.daily_proactive_limit == defaults.daily_proactive_limit
        assert s.schema_version == defaults.schema_version
        assert s.schema_version == 7
        assert s.tts == ["say", "Jian (Premium)"]

    def test_missing_key_uses_default(self, tmp_path: Path) -> None:
        target = tmp_path / "partial.toml"
        target.write_text('mic_consent_granted = true\n', encoding="utf-8")
        s = load_settings(target)
        assert s.mic_consent_granted is True
        assert s.wake_keyword == Settings().wake_keyword  # default applied

    def test_invalid_toml_returns_defaults(self, tmp_path: Path) -> None:
        target = tmp_path / "broken.toml"
        target.write_bytes(b"\xff\xfe not valid toml !!!!")
        s = load_settings(target)
        assert s == Settings()


class TestRoundTrip:
    def test_default_round_trip(self, tmp_path: Path) -> None:
        s = Settings()
        loaded = _round_trip(tmp_path, s)
        assert loaded.mic_consent_granted == s.mic_consent_granted
        assert loaded.quiet_mode == s.quiet_mode
        assert loaded.wake_keyword == s.wake_keyword
        assert loaded.daily_proactive_limit == s.daily_proactive_limit
        assert loaded.schema_version == s.schema_version

    def test_modified_values_round_trip(self, tmp_path: Path) -> None:
        from dataclasses import replace

        s = replace(
            Settings(),
            mic_consent_granted=True,
            mic_consent_at="2026-05-01T00:00:00+00:00",
            quiet_mode=True,
            wake_keyword="비서야",
            daily_proactive_limit=5,
        )
        loaded = _round_trip(tmp_path, s)
        assert loaded.mic_consent_granted is True
        assert loaded.mic_consent_at == "2026-05-01T00:00:00+00:00"
        assert loaded.quiet_mode is True
        assert loaded.wake_keyword == "비서야"
        assert loaded.daily_proactive_limit == 5


class TestUnknownKeys:
    def test_unknown_key_is_ignored(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        target = tmp_path / "extra.toml"
        target.write_text(
            'unknown_future_field = "hello"\nquiet_mode = true\n',
            encoding="utf-8",
        )
        s = load_settings(target)
        assert s.quiet_mode is True
        # structlog prints to stdout; verify the unknown key name appears in output
        captured = capsys.readouterr()
        assert "unknown_future_field" in captured.out

    def test_unknown_keys_do_not_raise(self, tmp_path: Path) -> None:
        target = tmp_path / "extra2.toml"
        target.write_text('foo = 1\nbar = "baz"\n', encoding="utf-8")
        s = load_settings(target)
        assert isinstance(s, Settings)


class TestSchemaMigration:
    def test_v1_file_with_wake_keyword_path_loads_cleanly(self, tmp_path: Path) -> None:
        """Schema v1 file that includes the removed wake_keyword_path key loads
        without error.  The removed key is silently dropped (info-logged, not
        an error/warning) and the resulting Settings is valid v2."""
        target = tmp_path / "v1_settings.toml"
        target.write_text(
            'schema_version = 1\n'
            'mic_consent_granted = true\n'
            'wake_keyword = "자기야"\n'
            'wake_keyword_path = "/old/path/custom.ppn"\n'
            'quiet_mode = false\n'
            'daily_proactive_limit = 3\n',
            encoding="utf-8",
        )
        s = load_settings(target)
        assert isinstance(s, Settings)
        assert s.mic_consent_granted is True
        assert s.wake_keyword == "자기야"
        # wake_keyword_path must not exist on the new Settings dataclass
        assert not hasattr(s, "wake_keyword_path")

    def test_v1_file_wake_keyword_path_does_not_cause_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """wake_keyword_path from v1 is treated as a known-removed field, so
        it must NOT appear in the 'unknown keys' warning output."""
        target = tmp_path / "v1_compat.toml"
        target.write_text(
            'wake_keyword_path = "/old/path/custom.ppn"\n'
            'quiet_mode = true\n',
            encoding="utf-8",
        )
        s = load_settings(target)
        assert s.quiet_mode is True
        captured = capsys.readouterr()
        # Should NOT appear as an unknown key warning
        assert "wake_keyword_path" not in captured.out or "settings_unknown_keys" not in captured.out


class TestAtomicWrite:
    def test_file_created_in_new_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "settings.toml"
        assert not nested.exists()
        save_settings(Settings(), nested)
        assert nested.exists()

    def test_crash_mid_write_preserves_original(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataclasses import replace

        target = tmp_path / "settings.toml"
        original = replace(Settings(), quiet_mode=False, daily_proactive_limit=3)
        save_settings(original, target)

        # Simulate crash: monkeypatch os.replace to raise on second call
        call_count = 0
        real_replace = os.replace

        def flaky_replace(src: str, dst: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("simulated crash")
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", flaky_replace)
        with pytest.raises(OSError, match="simulated crash"):
            save_settings(replace(Settings(), quiet_mode=True, daily_proactive_limit=99), target)

        # Original must survive intact
        loaded = load_settings(target)
        assert loaded.quiet_mode is False
        assert loaded.daily_proactive_limit == 3

    def test_written_file_is_valid_toml(self, tmp_path: Path) -> None:
        target = tmp_path / "settings.toml"
        save_settings(Settings(), target)
        with open(target, "rb") as fh:
            parsed = tomllib.load(fh)
        assert "mic_consent_granted" in parsed


class TestPathOverride:
    def test_her_settings_path_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        custom = tmp_path / "custom_settings.toml"
        monkeypatch.setenv("HER_SETTINGS_PATH", str(custom))
        result = settings_path()
        assert result == custom

    def test_default_path_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HER_SETTINGS_PATH", raising=False)
        result = settings_path()
        assert result == Path.home() / ".her" / "settings.toml"

    def test_load_uses_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from dataclasses import replace

        custom = tmp_path / "env_settings.toml"
        save_settings(replace(Settings(), quiet_mode=True), custom)
        monkeypatch.setenv("HER_SETTINGS_PATH", str(custom))
        s = load_settings()
        assert s.quiet_mode is True
