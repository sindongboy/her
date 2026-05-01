"""Manual smoke test for the voice channel hardware pipeline.

NOT a pytest test — run manually with real hardware:
    make voice-smoke
    # or:
    uv run python scripts/smoke_voice.py

Requires:
    - Real microphone + speaker (portaudio / sounddevice)
    - GEMINI_API_KEY set in environment (for Gemini TTS)
    - faster-whisper model downloaded (auto-downloads on first run)

Steps:
    1. Open default mic + speaker.
    2. Say something via TTS: greeting + 5-second prompt.
    3. Capture 5 seconds of raw audio from the mic (no VAD).
    4. Transcribe via STT (faster-whisper).
    5. Print transcript to stdout.
    6. Play the transcript back via TTS.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(
            "[경고] GEMINI_API_KEY 가 없습니다. Gemini TTS 가 동작하지 않을 수 있습니다.",
            file=sys.stderr,
        )

    # ── Audio devices ───────────────────────────────────────────────────
    try:
        from apps.channels.voice.audio import (
            AudioDeviceError,
            AudioFormat,
            open_microphone,
            open_speaker,
        )
    except ImportError as exc:
        print(
            f"마이크/스피커 라이브러리(portaudio)가 없어요. "
            f"brew install portaudio 후 다시 시도하세요. ({exc})"
        )
        sys.exit(1)

    mic_fmt = AudioFormat(sample_rate=16_000, channels=1, sample_width_bytes=2)
    mic = open_microphone(fmt=mic_fmt)
    speaker = open_speaker()

    # ── STT ─────────────────────────────────────────────────────────────
    try:
        from apps.channels.voice.stt import STT  # type: ignore[import]
    except ImportError as exc:
        print(f"[오류] STT 모듈을 불러올 수 없습니다: {exc}")
        sys.exit(1)

    print("[smoke] STT 모델을 불러오는 중...")
    stt = STT(model_size="small", language="ko")
    await stt.warmup()
    print("[smoke] STT 준비 완료")

    # ── TTS ─────────────────────────────────────────────────────────────
    try:
        from apps.channels.voice.tts import GeminiTTS, SayFallbackTTS, TTS, TTSConfig  # type: ignore[import]
    except ImportError as exc:
        print(f"[오류] TTS 모듈을 불러올 수 없습니다: {exc}")
        sys.exit(1)

    tts_config = TTSConfig(voice_name="Kore", api_key=api_key or None)
    primary_tts = GeminiTTS(config=tts_config)
    fallback_tts = SayFallbackTTS()
    tts = TTS(primary=primary_tts, fallback=fallback_tts)

    # ── Step 1: Greet ───────────────────────────────────────────────────
    greeting = "안녕하세요. 마이크 테스트입니다. 5초간 말씀해주세요."
    print(f"[smoke] TTS 재생: {greeting!r}")
    try:
        await tts.speak_text(greeting, speaker)
    except AudioDeviceError as exc:
        print(
            f"마이크/스피커 라이브러리(portaudio)가 없어요. "
            f"brew install portaudio 후 다시 시도하세요. ({exc})"
        )
        sys.exit(1)

    # ── Step 2: Record 5 seconds of raw audio ──────────────────────────
    print("[smoke] 마이크 녹음 시작 (5초)...")
    capture_seconds = 5.0
    frame_ms = 20
    expected_frames = int(capture_seconds * 1000 / frame_ms)
    captured: bytearray = bytearray()

    t_start = time.monotonic()
    frame_count = 0
    try:
        async for frame in mic.frames(frame_ms=frame_ms):
            captured.extend(frame)
            frame_count += 1
            if frame_count >= expected_frames:
                break
    except AudioDeviceError as exc:
        print(
            f"마이크/스피커 라이브러리(portaudio)가 없어요. "
            f"brew install portaudio 후 다시 시도하세요. ({exc})"
        )
        sys.exit(1)

    elapsed = time.monotonic() - t_start
    print(f"[smoke] 녹음 완료: {len(captured)} bytes, {elapsed:.1f}s, {frame_count} frames")

    await mic.close()

    # ── Step 3: Transcribe ──────────────────────────────────────────────
    print("[smoke] STT 변환 중...")
    transcript = await stt.transcribe(bytes(captured), fmt=mic_fmt)
    print(f"[smoke] 전사 결과: {transcript!r}")

    # ── Step 4: TTS playback of transcript ──────────────────────────────
    if transcript.strip():
        reply = f"이렇게 들었어요: {transcript}"
    else:
        reply = "아무 말씀도 들리지 않았어요. 다시 시도해주세요."

    print(f"[smoke] TTS 재생: {reply!r}")
    await tts.speak_text(reply, speaker)

    await speaker.flush()
    await speaker.close()

    print("[smoke] 완료.")


if __name__ == "__main__":
    asyncio.run(main())
