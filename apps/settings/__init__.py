"""Settings module — public surface.

Exports:
    Settings       — user settings dataclass (slots=True).
    load_settings  — read ~/.her/settings.toml; returns defaults if absent.
    save_settings  — atomic write to ~/.her/settings.toml.
    settings_path  — resolve active path (honors HER_SETTINGS_PATH).
"""

from apps.settings.store import Settings, load_settings, save_settings, settings_path

__all__ = [
    "Settings",
    "load_settings",
    "save_settings",
    "settings_path",
]
