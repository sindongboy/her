"""Daemon main loop — runs the voice channel with wake-word enabled.

Checks mic consent before starting. Exits with EX_CONFIG (78) if consent
has not been granted (see CLAUDE.md §10, §3.2).

The ``stop_event`` is set externally (SIGTERM / SIGINT via lifecycle.py)
to trigger a clean shutdown.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# EX_CONFIG from sysexits.h — misconfigured environment
_EX_CONFIG = 78


async def run_daemon(
    *,
    db_path: Path,
    log_dir: Path,
    settings: object,  # apps.settings.Settings (or shim)
    stop_event: asyncio.Event,
) -> None:
    """Main daemon loop.

    Steps
    -----
    1. Verify mic consent.  Missing → log + exit(EX_CONFIG).
    2. Build VoiceChannel with wake_word enabled.
    3. Run channel concurrently with a stop_event watcher.
    4. On stop: call channel.stop(), flush logs, close store.
    """
    # ── 1. Consent check ────────────────────────────────────────────────
    if not getattr(settings, "mic_consent_granted", False):
        logger.error(
            "daemon.consent_missing",
            reason="mic_consent_granted is False — run 'bin/her consent' to grant permission",
        )
        sys.exit(_EX_CONFIG)

    # ── 2. MemoryStore ───────────────────────────────────────────────────
    from apps.memory.store import MemoryStore

    store = MemoryStore(db_path)

    # ── 3. AgentCore ─────────────────────────────────────────────────────
    import os

    api_key = os.environ.get("GEMINI_API_KEY", "")
    agent_model = os.environ.get(
        "HER_AGENT_MODEL", getattr(settings, "agent_model", "gemini-3.1-pro-preview")
    )
    if not api_key:
        logger.warning("daemon.no_api_key", hint="Set GEMINI_API_KEY in environment")

    from apps.agent.core import AgentCore

    agent = AgentCore(store, api_key=api_key or None, model_id=agent_model)

    # ── 4. Build VoiceChannel ────────────────────────────────────────────
    channel = _build_voice_channel(agent, store, settings)
    if channel is None:
        logger.error("daemon.voice_channel_unavailable")
        store.close()
        sys.exit(1)

    # ── 5. Build ProactiveEngine ─────────────────────────────────────────
    from apps.proactive.engine import ProactiveConfig, ProactiveEngine

    proactive_config = ProactiveConfig(
        daily_limit=getattr(settings, "daily_proactive_limit", 3),
        silence_threshold_hours=getattr(settings, "silence_threshold_hours", 4.0),
        quiet_mode=getattr(settings, "quiet_mode", False),
    )
    proactive = ProactiveEngine(
        store,
        agent,
        voice_channel=channel,  # type: ignore[arg-type]
        config=proactive_config,
    )

    # ── 6. Run channel concurrently with stop watcher ────────────────────
    logger.info("daemon.start", db=str(db_path), log_dir=str(log_dir))
    try:
        await asyncio.gather(
            channel.run(),
            proactive.run(stop_event=stop_event),
            _watch_stop(stop_event, channel),
        )
    finally:
        logger.info("daemon.shutdown")
        store.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _watch_stop(stop_event: asyncio.Event, channel: object) -> None:
    """Wait for stop_event then call channel.stop()."""
    await stop_event.wait()
    logger.info("daemon.stop_requested")
    stop_fn = getattr(channel, "stop", None)
    if callable(stop_fn):
        result = stop_fn()
        if asyncio.iscoroutine(result):
            await result


def _build_voice_channel(
    agent: object,
    store: object,
    settings: object,
) -> object | None:
    """Construct a VoiceChannel with wake_word enabled.

    Returns None if required audio libraries are unavailable.
    """
    try:
        from apps.channels.voice.audio import open_microphone, open_speaker
        from apps.channels.voice.channel import VoiceChannel
        from apps.channels.voice.stt import STT  # type: ignore[import]
        from apps.channels.voice.tts import GeminiTTS, SayFallbackTTS, TTS, TTSConfig  # type: ignore[import]
        from apps.channels.voice.vad import VAD  # type: ignore[import]
    except ImportError as exc:
        logger.error("daemon.import_error", exc=str(exc))
        return None

    import os

    api_key = os.environ.get("GEMINI_API_KEY", "")
    stt_model = os.environ.get("HER_STT_MODEL", getattr(settings, "stt_model", "medium"))
    echo_gate_ms = int(
        os.environ.get("HER_ECHO_GATE_MS", str(getattr(settings, "echo_gate_ms", 400)))
    )

    # TTS as [engine, voice]. Env override: HER_TTS="engine:voice".
    settings_tts = getattr(settings, "tts", ["say", "Jian (Premium)"])
    tts_env = os.environ.get("HER_TTS", "").strip()
    if tts_env and ":" in tts_env:
        tts_engine, tts_voice = tts_env.split(":", 1)
    else:
        tts_engine, tts_voice = settings_tts[0], settings_tts[1]

    mic = open_microphone()
    speaker = open_speaker()
    vad = VAD()
    stt = STT(model_size=stt_model, language="ko")

    if tts_engine == "say":
        tts: object = SayFallbackTTS(voice=tts_voice)
    else:
        tts_config = TTSConfig(voice=tts_voice, model_id=tts_engine)
        primary_tts = GeminiTTS(config=tts_config, api_key=api_key or None)
        fallback_tts = SayFallbackTTS()
        tts = TTS(primary=primary_tts, fallback=fallback_tts)

    wake_keyword: str = getattr(settings, "wake_keyword", "자기야")
    quiet_mode: bool = getattr(settings, "quiet_mode", False)

    channel = VoiceChannel(
        agent,
        store,
        mic=mic,
        speaker=speaker,
        vad=vad,
        stt=stt,
        tts=tts,
        enable_wake_word=True,
        enable_interruption=True,
        wake_keyword=wake_keyword,
        is_quiet=quiet_mode,
        echo_gate_ms=echo_gate_ms,
    )
    return channel
