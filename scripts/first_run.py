"""her — first-run setup.

Checks:
  1. GEMINI_API_KEY is set (else: tell user to edit .envrc).
  2. Mic consent granted in settings — if not, run prompt_mic_consent → save.
  3. macOS portaudio available (best-effort: try `import sounddevice`).
  4. Print summary: "Ready" or list of items still needed.

Exit 0 on ready, 1 on missing config (after writing what could be written).

Usage:
  uv run python scripts/first_run.py
  uv run python scripts/first_run.py --non-interactive --auto-grant-consent
  uv run python scripts/first_run.py --settings-path /tmp/test_settings.toml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is importable when run as a script
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from apps.settings import (
    Settings,
    grant_mic_consent,
    is_consent_granted,
    load_settings,
    prompt_mic_consent,
    save_settings,
    settings_path,
)


def _check_gemini_key() -> tuple[bool, str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key and key != "REPLACE_ME":
        return True, "GEMINI_API_KEY 설정됨"
    return False, "GEMINI_API_KEY 없음 — .envrc 에 키를 입력하고 `direnv allow` 실행"


def _check_sounddevice() -> tuple[bool, str]:
    try:
        import sounddevice  # noqa: F401

        return True, "sounddevice (PortAudio) 사용 가능"
    except ImportError:
        return False, (
            "sounddevice 없음 — `uv sync` 를 실행하거나 "
            "`brew install portaudio && uv add sounddevice` 필요 (Phase 1+)"
        )


def _handle_consent(
    settings: Settings,
    *,
    non_interactive: bool,
    auto_grant: bool,
    settings_file: Path,
) -> Settings:
    if is_consent_granted(settings):
        return settings

    if non_interactive or auto_grant:
        updated = grant_mic_consent(settings)
        save_settings(updated, settings_file)
        print("[자동] 마이크 동의가 자동으로 설정되었습니다 (--auto-grant-consent).")
        return updated

    granted = prompt_mic_consent()
    if granted:
        updated = grant_mic_consent(settings)
        save_settings(updated, settings_file)
        return updated

    return settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="her first-run setup: checks env vars and mic consent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip all interactive prompts (for CI). Does NOT auto-grant consent.",
    )
    parser.add_argument(
        "--auto-grant-consent",
        action="store_true",
        help="Automatically grant mic consent without prompting (for tests/CI).",
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=None,
        help="Override settings file path (default: ~/.her/settings.toml).",
    )

    args = parser.parse_args(argv)

    settings_file: Path = args.settings_path or settings_path()
    settings = load_settings(settings_file)

    print("=" * 60)
    print("her — first-run 점검")
    print("=" * 60)

    # ── 1. Gemini key ──────────────────────────────────────────
    gemini_ok, gemini_msg = _check_gemini_key()
    status1 = "OK" if gemini_ok else "MISSING"
    print(f"[{status1}] {gemini_msg}")

    # ── 2. Mic consent ─────────────────────────────────────────
    settings = _handle_consent(
        settings,
        non_interactive=args.non_interactive,
        auto_grant=args.auto_grant_consent,
        settings_file=settings_file,
    )
    consent_ok = is_consent_granted(settings)
    status2 = "OK" if consent_ok else "PENDING"
    if consent_ok:
        print(f"[{status2}] 마이크 상시 청취 동의 완료 (at: {settings.mic_consent_at})")
    else:
        print(f"[{status2}] 마이크 동의 미완료 — `make first-run` 을 다시 실행하거나 수동으로 동의하세요.")

    # ── 3. sounddevice / PortAudio (optional, warn only) ───────
    sd_ok, sd_msg = _check_sounddevice()
    status3 = "OK" if sd_ok else "WARN"
    print(f"[{status3}] {sd_msg}")

    print("=" * 60)

    # Required items: GEMINI_API_KEY + mic consent
    required_ok = gemini_ok and consent_ok
    if required_ok:
        print("Ready — her 를 시작할 수 있습니다. `make text` 로 텍스트 채널을 열어보세요.")
        print("Wake word: 로컬 Whisper 사용 (별도 키 불필요)")
        return 0

    print("아직 설정이 완료되지 않았습니다. 위의 MISSING/PENDING 항목을 해결하세요.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
