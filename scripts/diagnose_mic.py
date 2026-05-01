"""Diagnose why voice channel isn't reacting.

Checks:
  1. macOS mic permission (sounddevice can open mic).
  2. Available audio devices + default device.
  3. Capture 5 seconds, report peak/RMS levels (was anything actually picked up?).
  4. Run VAD on the captured audio: how many frames look like speech?

Run:
    uv run python scripts/diagnose_mic.py
"""

from __future__ import annotations

import asyncio
import math
import struct
import sys
import time


async def main() -> int:
    # ── 1. Sounddevice import + device list ───────────────────────────────
    try:
        import sounddevice as sd
    except ImportError as exc:
        print(f"[FAIL] sounddevice 모듈 없음: {exc}", file=sys.stderr)
        print("       brew install portaudio && uv sync", file=sys.stderr)
        return 1

    print("[ok] sounddevice import")

    print("\n[devices]")
    for idx, dev in enumerate(sd.query_devices()):
        ch_in = dev.get("max_input_channels", 0)
        ch_out = dev.get("max_output_channels", 0)
        marker = ""
        try:
            default_in, default_out = sd.default.device
            if idx == default_in:
                marker += " <- DEFAULT IN"
            if idx == default_out:
                marker += " <- DEFAULT OUT"
        except Exception:
            pass
        print(f"  [{idx:2d}] in={ch_in} out={ch_out}  {dev['name']!r}{marker}")

    # ── 2. Open mic, capture 5 seconds ─────────────────────────────────────
    from apps.channels.voice.audio import AudioFormat, open_microphone

    fmt = AudioFormat(sample_rate=16000, channels=1, sample_width_bytes=2)
    print(f"\n[capture] 5초간 녹음 — 마이크에 평소 음량으로 말씀하세요...")
    mic = open_microphone(fmt=fmt)

    captured = bytearray()
    frame_ms = 20
    expected_frames = int(5000 / frame_ms)

    t0 = time.monotonic()
    n = 0
    async for frame in mic.frames(frame_ms=frame_ms):
        captured.extend(frame)
        n += 1
        if n >= expected_frames:
            break
    elapsed = time.monotonic() - t0
    await mic.close()

    print(f"[capture] 완료 — {len(captured)} bytes, {elapsed:.1f}초, {n} 프레임")

    # ── 3. Audio level analysis ────────────────────────────────────────────
    samples = struct.unpack(f"{len(captured) // 2}h", bytes(captured))
    if not samples:
        print("[FAIL] 캡처 결과 비어있음 — 마이크 권한 또는 디바이스 문제")
        return 2

    peak = max(abs(s) for s in samples)
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))

    # int16 max = 32767. Peak < 100 = essentially silent.
    print(f"\n[level] peak={peak} (max=32767), rms={rms:.0f}")
    if peak < 100:
        print("[FAIL] 무음 수준. 가능한 원인:")
        print("       - macOS 마이크 권한 미허용 (시스템 설정 → 개인정보 → 마이크)")
        print("       - 잘못된 마이크 디바이스 선택됨")
        print("       - 마이크 음소거 또는 케이블 문제")
        return 3
    if peak < 1000:
        print("[WARN] 매우 작음. 마이크에 더 가깝게 말씀하시거나 시스템 입력 게인을 올리세요.")

    # ── 4. VAD test ─────────────────────────────────────────────────────────
    print("\n[vad] silero-vad 로 음성 활동 분석 중...")
    try:
        from apps.channels.voice.vad import VAD
    except ImportError as exc:
        print(f"[FAIL] VAD 모듈 import 실패: {exc}")
        return 4

    vad = VAD()
    # silero v5: 32ms frames (512 samples @ 16kHz)
    silero_frame_bytes = 512 * 2
    vad_frames_total = len(captured) // silero_frame_bytes
    vad_speech = 0
    for i in range(vad_frames_total):
        start = i * silero_frame_bytes
        end = start + silero_frame_bytes
        if vad.is_speech(bytes(captured[start:end])):
            vad_speech += 1

    pct = 100 * vad_speech / max(vad_frames_total, 1)
    print(f"[vad] {vad_speech}/{vad_frames_total} 프레임이 음성으로 감지됨 ({pct:.0f}%)")

    if vad_speech == 0:
        print("[FAIL] VAD 가 아예 음성 검출 못함.")
        print("       - 음량은 잡혔지만 VAD 임계값(0.5)에 못 미침")
        print("       - 더 또렷하게 말씀하거나 마이크에 가깝게")
        return 5
    if pct < 5:
        print("[WARN] 음성 비율 매우 낮음. 정상 발화는 보통 30%+ 입니다.")

    print("\n[ok] 마이크 + VAD 정상 동작. make voice 가 작동해야 합니다.")
    print("     그래도 반응 없으면 STT 모델/네트워크 문제일 수 있어요.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
