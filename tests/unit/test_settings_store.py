"""Unit tests for apps/settings/store.py."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

from apps.settings.store import Settings, load_settings, save_settings, settings_path


def _round_trip(tmp_path: Path, s: Settings) -> Settings:
    target = tmp_path / "settings.toml"
    save_settings(s, target)
    return load_settings(target)


class TestLoadDefaults:
    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.toml"
        s = load_settings(missing)
        defaults = Settings()
        assert s.web_host == defaults.web_host == "127.0.0.1"
        assert s.web_port == defaults.web_port == 8765
        assert s.agent_model == defaults.agent_model
        assert s.schema_version == defaults.schema_version == 8

    def test_missing_key_uses_default(self, tmp_path: Path) -> None:
        target = tmp_path / "partial.toml"
        target.write_text('web_port = 9000\n', encoding="utf-8")
        s = load_settings(target)
        assert s.web_port == 9000
        assert s.web_host == Settings().web_host

    def test_invalid_toml_returns_defaults(self, tmp_path: Path) -> None:
        target = tmp_path / "broken.toml"
        target.write_bytes(b"\xff\xfe not valid toml !!!!")
        s = load_settings(target)
        assert s == Settings()


class TestRoundTrip:
    def test_default_round_trip(self, tmp_path: Path) -> None:
        s = Settings()
        loaded = _round_trip(tmp_path, s)
        assert loaded == s

    def test_modified_values_round_trip(self, tmp_path: Path) -> None:
        from dataclasses import replace

        s = replace(
            Settings(),
            web_port=9000,
            agent_model="gemini-3.1-pro-preview",
            location_name="부산",
            enable_calendar=False,
        )
        loaded = _round_trip(tmp_path, s)
        assert loaded.web_port == 9000
        assert loaded.location_name == "부산"
        assert loaded.enable_calendar is False


class TestUnknownKeys:
    def test_unknown_key_is_ignored(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "extra.toml"
        target.write_text(
            'unknown_future_field = "hello"\nweb_port = 9000\n',
            encoding="utf-8",
        )
        s = load_settings(target)
        assert s.web_port == 9000
        captured = capsys.readouterr()
        assert "unknown_future_field" in captured.out


class TestSchemaMigration:
    def test_v7_voice_keys_dropped_silently(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A pre-v8 file with mic_consent/quiet_mode/wake_keyword/tts/etc loads
        cleanly: removed keys are info-logged, not warned, and the resulting
        Settings is a valid v8 with default values for the new fields."""
        target = tmp_path / "v7_settings.toml"
        target.write_text(
            'schema_version = 7\n'
            'mic_consent_granted = true\n'
            'mic_consent_at = "2026-04-30T00:00:00+00:00"\n'
            'quiet_mode = false\n'
            'wake_keyword = "자기야"\n'
            'daily_proactive_limit = 3\n'
            'silence_threshold_hours = 4.0\n'
            'stt_model = "medium"\n'
            'tts = ["say", "Jian (Premium)"]\n'
            'echo_gate_ms = 400\n'
            'agent_model = "gemini-3.1-pro-preview"\n'
            'location_name = "서울"\n',
            encoding="utf-8",
        )
        s = load_settings(target)
        assert isinstance(s, Settings)
        assert s.agent_model == "gemini-3.1-pro-preview"
        assert s.location_name == "서울"
        # Removed keys must not exist on the new Settings dataclass
        for removed in ("mic_consent_granted", "quiet_mode", "wake_keyword", "tts"):
            assert not hasattr(s, removed)

    def test_v7_removed_keys_are_not_unknown_warnings(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "v7_compat.toml"
        target.write_text(
            'wake_keyword = "비서야"\n'
            'mic_consent_granted = true\n'
            'web_port = 9000\n',
            encoding="utf-8",
        )
        s = load_settings(target)
        assert s.web_port == 9000
        captured = capsys.readouterr()
        # Removed keys go through info path, not unknown_keys warning.
        assert "settings_unknown_keys" not in captured.out


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
        original = replace(Settings(), web_port=8765)
        save_settings(original, target)

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
            save_settings(replace(Settings(), web_port=9999), target)

        loaded = load_settings(target)
        assert loaded.web_port == 8765

    def test_written_file_is_valid_toml(self, tmp_path: Path) -> None:
        target = tmp_path / "settings.toml"
        save_settings(Settings(), target)
        with open(target, "rb") as fh:
            parsed = tomllib.load(fh)
        assert "web_host" in parsed
        assert "web_port" in parsed
        assert parsed["schema_version"] == 8


class TestPathOverride:
    def test_her_settings_path_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        custom = tmp_path / "custom_settings.toml"
        monkeypatch.setenv("HER_SETTINGS_PATH", str(custom))
        result = settings_path()
        assert result == custom

    def test_default_path_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HER_SETTINGS_PATH", raising=False)
        result = settings_path()
        assert result == Path.home() / ".her" / "settings.toml"

    def test_load_uses_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataclasses import replace

        custom = tmp_path / "env_settings.toml"
        save_settings(replace(Settings(), web_port=9100), custom)
        monkeypatch.setenv("HER_SETTINGS_PATH", str(custom))
        s = load_settings()
        assert s.web_port == 9100
