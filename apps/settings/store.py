"""Settings store — read/write ~/.her/settings.toml using stdlib only.

Reads with tomllib (stdlib, Python 3.11+).
Writes by hand-formatting (flat scalar fields only — no tomli_w dep needed).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_SETTINGS_PATH = Path.home() / ".her" / "settings.toml"

_SCHEMA_VERSION = 8

# Keys removed from older schema versions — silently dropped during load.
# v8 ripped out the voice/daemon/proactive subsystem. The keys below all
# come from earlier voice-era settings files; they're tolerated for one
# migration cycle so existing TOML files load cleanly.
_REMOVED_KEYS: frozenset[str] = frozenset(
    {
        # v1–v4 transient keys
        "wake_keyword_path",
        "tts_provider",
        "tts_model",
        "tts_voice",
        # v8 voice/daemon/proactive removal
        "mic_consent_granted",
        "mic_consent_at",
        "quiet_mode",
        "wake_keyword",
        "daily_proactive_limit",
        "silence_threshold_hours",
        "stt_model",
        "tts",
        "echo_gate_ms",
    }
)


@dataclass(slots=True)
class Settings:
    # Web UI
    web_host: str = "127.0.0.1"
    web_port: int = 8765
    # Model
    agent_model: str = "gemini-3.1-pro-preview"
    # Location (default: Seoul). Used for weather lookup and system-prompt context.
    location_name: str = "서울"
    location_lat: float = 37.5665
    location_lon: float = 126.9780
    # Gemini Google Search grounding — enables factual/current-events grounding.
    enable_search_grounding: bool = True
    # macOS Calendar integration — reads today + tomorrow events via osascript.
    enable_calendar: bool = True
    # Widget views go ~2 weeks ahead by default. The agent's recall path
    # uses its own narrower window (24h) so this only affects the calendar
    # widget / system-prompt context block.
    calendar_lookahead_days: int = 14
    calendar_max_events: int = 8
    schema_version: int = _SCHEMA_VERSION


def settings_path() -> Path:
    """Return the active settings path, honoring HER_SETTINGS_PATH override."""
    override = os.environ.get("HER_SETTINGS_PATH")
    if override:
        return Path(override)
    return DEFAULT_SETTINGS_PATH


def load_settings(path: Path | None = None) -> Settings:
    """Read TOML if it exists, else return defaults.

    Missing keys → default.  Unknown keys → warn and ignore.
    """
    target = path or settings_path()

    if not target.exists():
        logger.debug("settings_file_not_found", path=str(target))
        return Settings()

    try:
        with open(target, "rb") as fh:
            raw: dict = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        logger.warning("settings_load_error", path=str(target), error=str(exc))
        return Settings()

    known = {f.name for f in fields(Settings)}

    # Migration: keys removed in older schema versions are silently dropped.
    removed = set(raw) & _REMOVED_KEYS
    if removed:
        logger.info("settings_removed_keys_dropped", keys=sorted(removed))

    unknown = set(raw) - known - _REMOVED_KEYS
    if unknown:
        logger.warning("settings_unknown_keys", keys=sorted(unknown))

    kwargs: dict = {}
    defaults = Settings()
    for f in fields(Settings):
        if f.name not in raw:
            kwargs[f.name] = getattr(defaults, f.name)
        else:
            kwargs[f.name] = raw[f.name]

    return Settings(**kwargs)


def _to_toml_value(value: object) -> str:
    """Render a single scalar value as a TOML literal."""
    if value is None:
        return "none_placeholder"  # unreachable — we skip None fields below
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise TypeError(f"Unsupported TOML type: {type(value)}")


def _serialize_settings(s: Settings) -> str:
    """Hand-format Settings as a flat TOML string."""
    lines: list[str] = [
        "# her settings — managed by apps/settings/store.py",
        "# Edit manually or via `her` CLI.  Do NOT commit this file.",
        "",
    ]
    for f in fields(Settings):
        value = getattr(s, f.name)
        if value is None:
            # TOML has no native null; use an empty string to preserve the key
            lines.append(f"# {f.name} is unset (null)")
            continue
        lines.append(f"{f.name} = {_to_toml_value(value)}")
    lines.append("")  # trailing newline
    return "\n".join(lines)


def save_settings(s: Settings, path: Path | None = None) -> None:
    """Atomic write: write to .tmp, fsync, rename.  Creates parent dirs."""
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    # Stamp the on-disk schema_version to whatever the current code understands.
    # Migrations of older files happen via _REMOVED_KEYS + new-field defaults at
    # load time; persisting the new version makes the round-trip idempotent.
    s.schema_version = _SCHEMA_VERSION

    tmp = target.with_suffix(".toml.tmp")
    content = _serialize_settings(s)

    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except OSError as exc:
        logger.error("settings_save_error", path=str(target), error=str(exc))
        raise
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    logger.debug("settings_saved", path=str(target))
