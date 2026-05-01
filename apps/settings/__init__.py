"""Settings module — public surface.

Exports:
    Settings            — user settings dataclass (slots=True).
    load_settings       — read ~/.her/settings.toml; returns defaults if absent.
    save_settings       — atomic write to ~/.her/settings.toml.
    settings_path       — resolve active path (honors HER_SETTINGS_PATH).
    is_consent_granted  — check mic consent field on a Settings instance.
    grant_mic_consent   — pure: return updated Settings with consent set.
    prompt_mic_consent  — interactive Korean consent prompt; returns bool.
"""

from apps.settings.consent import grant_mic_consent, is_consent_granted, prompt_mic_consent
from apps.settings.store import Settings, load_settings, save_settings, settings_path

__all__ = [
    "Settings",
    "load_settings",
    "save_settings",
    "settings_path",
    "is_consent_granted",
    "grant_mic_consent",
    "prompt_mic_consent",
]
