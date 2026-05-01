"""VoiceChannel orchestrator — Phase 2/3 with barge-in interruption and wake word.

CLAUDE.md references:
  §2.1  Voice-first; first audio out as soon as first sentence ready.
  §2.4  Quiet mode — suppresses TTS playback; toggleable at runtime.
  §2.5  Multi-channel: same AgentCore as text, channel='voice'.
  §6.1  Pipeline: VAD-gated mic → STT → stream_respond → TTS.
        Interruption: user speaks while assistant speaks → cancel TTS + agent
        within 300ms; captured speech becomes next turn input.
        Wake Word: VAD + faster-whisper polling (reuses existing STT/VAD).
  §4    Agent Core knows no channel details; envelope only.
  §12   Decision 2026-05-01: Porcupine replaced by Whisper-polling detector.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

import structlog

from apps.channels.voice.audio import AudioInputStream, AudioOutputStream
from apps.presence import Event, EventBus

log = structlog.get_logger(__name__)


# ── safe publish helper ───────────────────────────────────────────────────────


def _publish(bus: EventBus | None, event: Event) -> None:
    """Publish *event* to *bus* without ever raising.

    If bus is None the call is a silent no-op (backward-compat for tests that
    don't inject a bus).  Any exception from the real bus is swallowed and
    logged so that presence failures never crash voice logic.
    """
    if bus is None:
        return
    try:
        bus.publish(event)
    except Exception as exc:  # pragma: no cover
        log.warning("publish_failed", error=str(exc))

# ── Protocols ────────────────────────────────────────────────────────────────
# Written against the sibling contracts (vad.py / stt.py / tts.py).
# We use Protocol instead of direct imports so the channel compiles even
# before those files exist, and tests can inject fakes freely.


class _VADProtocol(Protocol):
    """Minimal interface required from vad.VAD."""

    pass  # VAD is passed to speech_segments; we don't call it directly here.


class _SpeechSegmentsProtocol(Protocol):
    """Callable that yields speech segment bytes from an AudioInputStream."""

    def __call__(
        self,
        stream: AudioInputStream,
        vad: Any,
        *,
        min_speech_ms: int,
        max_silence_ms: int,
        max_segment_ms: int,
    ) -> AsyncIterator[bytes]:
        ...


class _STTProtocol(Protocol):
    """Minimal interface required from stt.STT."""

    async def transcribe(self, pcm_bytes: bytes) -> str:
        ...

    async def warmup(self) -> None:
        ...


class _TTSProtocol(Protocol):
    """Minimal interface required from tts.TTS."""

    async def speak(self, text_stream: AsyncIterator[str], output: AudioOutputStream) -> None:
        ...

    async def speak_text(self, text: str, output: AudioOutputStream) -> None:
        ...


class _AgentProtocol(Protocol):
    """Minimal AgentCore interface required by VoiceChannel."""

    async def stream_respond(
        self,
        message: str,
        *,
        episode_id: int | None,
        channel: str,
    ) -> AsyncIterator[str]:
        ...


# ── InterruptionDetector factory protocol ────────────────────────────────────


class _InterruptDetectorProtocol(Protocol):
    """Factory that creates a detector for one turn.

    Called at the start of each TTS turn with the interrupt queue.
    Returns an object with wait_for_interrupt() coroutine.
    """

    def __call__(self, queue: asyncio.Queue[bytes | None]) -> Any:
        ...


# ── WakeDetector factory protocol ────────────────────────────────────────────


class _WakeDetectorProtocol(Protocol):
    """Factory that creates a WakeDetector instance.

    Injected by tests to avoid requiring real STT/VAD in unit tests.
    Signature: (stt, vad, *, keyword) -> WakeDetector-compatible object.
    """

    def __call__(self, stt: Any, vad: Any, *, keyword: str) -> Any:
        ...


# ── VoiceChannel ─────────────────────────────────────────────────────────────


class VoiceChannel:
    """Phase 2: voice loop with barge-in interruption.

    Loop (per turn):
      1. Await next speech segment from shared mic stream.
      2. STT: transcribe PCM bytes → text.
      3. Skip turn if transcript is empty (log warning, don't call agent).
      4. agent.stream_respond(transcript, episode_id, channel='voice') → text chunks.
      5. With interruption enabled:
           - speak_task = asyncio.create_task(tts.speak(...))
           - interrupt_task = asyncio.create_task(detector.wait_for_interrupt())
           - Wait for first to complete.
           - If interrupt wins: cancel speak_task; stop speaker; captured bytes
             become next turn input.
           - Else: cancel interrupt_task; continue normally.
      6. Carry episode_id forward for the next turn (multi-turn continuity).

    Parameters
    ----------
    agent:
        AgentCore (or compatible fake). Must implement stream_respond.
    store:
        MemoryStore — held for future use (e.g. proactive retrieval) and
        to mirror TextChannel's constructor signature.
    mic:
        AudioInputStream sourcing PCM int16 frames.
    speaker:
        AudioOutputStream for PCM playback.
    vad:
        VAD instance forwarded to speech_segments.
    stt:
        STT instance; transcribes raw PCM bytes.
    tts:
        TTS instance; plays text stream through speaker.
    speech_segments_fn:
        Async generator factory.  Defaults to importing from vad module at
        runtime so the channel can be constructed before vad.py exists in
        tests (fakes override this parameter).
    status_fn:
        Called with a plain-text Korean status string for each phase
        transition.  Defaults to printing to stderr.  Tests capture via list.
    enable_interruption:
        When True (default False), barge-in interruption is active during
        TTS playback.  Set True in __main__.py and in interruption tests.
        Existing Phase 1 tests leave this as False for backward-compat.
    interrupt_detector_fn:
        Factory called with (queue) → detector object per turn.  Injected
        by tests; defaults to building a real InterruptionDetector at runtime.
    enable_wake_word:
        When True, the channel runs in background mode: sleeps until the wake
        word fires, handles an active window of ``wake_active_window_s``
        seconds, then returns to sleep.  Implies a fanout mic stream.
    wake_detector_fn:
        Factory ``(stt, vad, *, keyword) -> WakeDetector``-compatible object.
        Injected by tests to avoid requiring real STT/VAD inference.  When
        None and ``enable_wake_word=True``, a real WakeDetector is built using
        the channel's existing STT and VAD instances plus ``wake_keyword``.
    wake_keyword:
        Korean wake phrase forwarded to WakeConfig (e.g. ``"자기야"``).
        Any natural-language Korean phrase works — Whisper understands it
        natively without custom training.  Only used when
        ``wake_detector_fn`` is None.
    wake_active_window_s:
        How many seconds to stay "awake" after the wake word fires without a
        new utterance.  Resets on each user turn.  Defaults to 30 s.
    is_quiet:
        Initial quiet-mode state.  When True, the agent is still called but
        TTS output is suppressed.  Toggle at runtime with ``set_quiet()``.
    """

    def __init__(
        self,
        agent: _AgentProtocol,
        store: Any,  # MemoryStore — typed loosely to avoid circular imports
        *,
        mic: AudioInputStream,
        speaker: AudioOutputStream,
        vad: Any,
        stt: _STTProtocol,
        tts: _TTSProtocol,
        speech_segments_fn: _SpeechSegmentsProtocol | None = None,
        status_fn: Callable[[str], None] | None = None,
        enable_interruption: bool = False,
        interrupt_detector_fn: _InterruptDetectorProtocol | None = None,
        enable_wake_word: bool = False,
        wake_detector_fn: _WakeDetectorProtocol | None = None,
        wake_keyword: str = "자기야",
        wake_active_window_s: float = 30.0,
        is_quiet: bool = False,
        bus: EventBus | None = None,
        echo_gate_ms: int = 400,
    ) -> None:
        self._agent = agent
        self._store = store
        self._mic = mic
        self._speaker = speaker
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._speech_segments_fn = speech_segments_fn
        self._status_fn = status_fn or _default_status
        self._episode_id: int | None = None
        self._running = False
        self._enable_interruption = enable_interruption
        self._interrupt_detector_fn = interrupt_detector_fn
        self._enable_wake_word = enable_wake_word
        self._wake_detector_fn = wake_detector_fn
        self._wake_keyword = wake_keyword
        self._wake_active_window_s = wake_active_window_s
        self.is_quiet: bool = is_quiet
        self._bus: EventBus | None = bus
        # Post-TTS sleep before draining mic — accounts for speaker tail and
        # room reverb so the assistant's own audio doesn't loop back through
        # the mic and self-trigger STT. Tests pass 0 for instant turns.
        self._echo_gate_ms = echo_gate_ms

    # ── public ───────────────────────────────────────────────────────────────

    def set_quiet(self, on: bool) -> None:
        """Toggle quiet mode.  Thread-safe (simple bool assignment).

        When quiet mode is on the assistant still calls the agent (so memory
        is updated) but TTS playback is suppressed.  This method is the
        public interface for daemon-eng's IPC toggle.
        """
        self.is_quiet = on
        log.info("quiet_mode_set", is_quiet=on)

    async def say(self, text: str) -> None:
        """Proactive utterance: push text through TTS without a user turn.

        Called by ProactiveEngine (§2.4).  Skipped if quiet mode is on or
        another proactive speak is already in progress (non-blocking lock).
        Does NOT interfere with the main run() loop turns.
        """
        if not hasattr(self, "_proactive_lock"):
            self._proactive_lock = asyncio.Lock()  # type: ignore[attr-defined]

        if self.is_quiet:
            log.debug("voice_channel.say_skipped_quiet")
            return

        # Non-blocking acquire: if a proactive speak is already in progress, skip.
        acquired = self._proactive_lock.locked()  # type: ignore[attr-defined]
        if acquired:
            log.debug("voice_channel.say_skipped_busy")
            return

        async with self._proactive_lock:  # type: ignore[attr-defined]
            _publish(
                self._bus,
                Event(
                    type="state",
                    payload={"value": "speaking", "channel": "voice"},
                    ts=time.monotonic(),
                ),
            )
            try:
                await self._tts.speak_text(text, self._speaker)
            except Exception as exc:
                log.error("voice_channel.say_failed", error=str(exc))
            finally:
                _publish(
                    self._bus,
                    Event(
                        type="state",
                        payload={"value": "idle", "channel": "voice"},
                        ts=time.monotonic(),
                    ),
                )

    async def run(self) -> None:
        """Main voice loop.  Returns on stop() or KeyboardInterrupt."""
        self._running = True
        self._status_fn("[시작] 음성 채널을 시작합니다.")
        # Publish loop-start idle state.
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "idle", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )

        self._status_fn("[STT 준비 중] Whisper 모델을 불러옵니다 (첫 실행은 다운로드 때문에 30-60초)...")
        try:
            # 60s ceiling — first-time medium load in same process as uvicorn
            # can be slow but should never exceed this.
            await asyncio.wait_for(self._stt.warmup(), timeout=60.0)
        except asyncio.TimeoutError:
            self._status_fn("[STT 경고] 워밍업이 60초 안에 끝나지 않았어요. 작은 모델로 바꿔보세요 (HER_STT_MODEL=small).")
            log.warning("stt_warmup_timeout")
        except Exception as exc:
            self._status_fn(f"[STT 경고] {exc}")
            log.warning("stt_warmup_failed", error=str(exc))

        # Optional TTS warmup — Gemini needs it to avoid cold-start fallback;
        # macOS `say` is a no-op (subprocess spawns are fast enough).
        warmup_fn = getattr(self._tts, "warmup", None)
        if callable(warmup_fn):
            self._status_fn("[TTS 준비 중] 음성 합성기를 준비하고 있어요...")
            try:
                await asyncio.wait_for(warmup_fn(), timeout=10.0)
            except asyncio.TimeoutError:
                self._status_fn("[TTS 경고] 워밍업 10초 초과. 첫 응답에서 폴백 음성이 나올 수 있어요.")
                log.warning("tts_warmup_timeout")
            except Exception as exc:
                self._status_fn(f"[TTS 경고] {exc}")
                log.warning("tts_warmup_failed", error=str(exc))

        self._status_fn(
            "[준비 완료] 마이크에 대고 평소 음량으로 말씀하세요. "
            "말씀 후 잠깐(0.7초) 멈추면 인식해요."
        )

        if self._enable_wake_word:
            await self._run_with_wake_word()
        elif self._enable_interruption:
            await self._run_with_interruption()
        else:
            await self._run_linear()

    async def stop(self) -> None:
        """Signal the run loop to exit cleanly after the current segment."""
        self._running = False
        log.info("voice_channel.stop_requested")

    # ── linear (Phase 1 compat) ───────────────────────────────────────────────

    async def _run_linear(self) -> None:
        """Original single-turn loop; no interruption.  Phase 1 behaviour."""
        speech_segments = self._resolve_speech_segments()

        try:
            async for segment_bytes in speech_segments(
                self._mic,
                self._vad,
                min_speech_ms=200,
                max_silence_ms=700,
                max_segment_ms=30_000,
            ):
                if not self._running:
                    break
                await self._handle_segment_linear(segment_bytes)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self._status_fn("[끝] 음성 채널을 종료합니다.")
            log.info("voice_channel.stopped")

    async def _handle_segment_linear(self, pcm_bytes: bytes) -> None:
        """Process one VAD-gated speech segment end-to-end (no interruption)."""
        # 1. STT — emit listening state before transcribe
        self._status_fn(f"[변환 중] 음성 {len(pcm_bytes) // 32000}초 분량을 텍스트로 바꾸고 있어요...")
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "listening", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )
        try:
            transcript = await self._stt.transcribe(pcm_bytes)
        except Exception as exc:
            log.error("stt_transcribe_failed", error=str(exc))
            self._status_fn(f"[STT 오류] {exc}")
            return

        # 2. Skip empty transcripts
        if not transcript.strip():
            log.warning("stt_empty_transcript")
            self._status_fn("[못 들었어요] 다시 말씀해 주세요.")
            return

        log.info("voice_channel.transcript_ready", length=len(transcript))
        self._status_fn(f'[들었어요] "{transcript}"')

        # Record user activity for SilenceTrigger.
        try:
            from apps.proactive.activity import record_activity
            record_activity("voice")
        except Exception:  # pragma: no cover
            pass

        # Emit transcript + thinking state
        _publish(
            self._bus,
            Event(
                type="transcript",
                payload={"text": transcript, "final": True, "channel": "voice"},
                ts=time.monotonic(),
            ),
        )

        # 3. Agent streaming response
        self._status_fn("[생각 중] 답변을 준비하고 있어요...")
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "thinking", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )
        try:
            text_stream = self._agent.stream_respond(
                transcript,
                episode_id=self._episode_id,
                channel="voice",
            )
            if asyncio.iscoroutine(text_stream):
                text_stream = await text_stream  # type: ignore[assignment]
        except Exception as exc:
            log.error("agent_stream_failed", error=str(exc))
            return

        # 4. TTS: bridge text stream → speaker (or drain silently in quiet mode)
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "speaking", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )
        await self._speak_or_drain(text_stream)

        # 5. Echo gate: discard any audio captured during STT/agent/TTS so the
        #    assistant's own speech doesn't loop back through the mic.
        #    Wait briefly for speaker tail + room reverb to die, then drain.
        if self._echo_gate_ms > 0:
            await asyncio.sleep(self._echo_gate_ms / 1000)
        try:
            dropped = self._mic.drain()
            if dropped:
                log.debug("mic_drained_after_tts", frames=dropped)
        except AttributeError:
            pass  # FakeMicrophone in old tests may not have drain()
        try:
            self._vad.reset()
        except AttributeError:
            pass

        _publish(
            self._bus,
            Event(
                type="response_end",
                payload={"channel": "voice", "episode_id": self._episode_id},
                ts=time.monotonic(),
            ),
        )
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "idle", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )

    # ── interruption-aware loop (Phase 2) ────────────────────────────────────

    async def _run_with_interruption(self) -> None:
        """Phase 2 loop: single shared mic stream + per-turn interruption."""
        from apps.channels.voice.interrupt import _FrameFanout, speech_segments_from_queue

        fanout = _FrameFanout(self._mic)
        stt_queue = fanout.subscribe()
        interrupt_queue = fanout.subscribe()

        fanout_task = asyncio.create_task(fanout.run(), name="fanout")

        try:
            # pending_pcm: speech captured during interruption, fed to next turn
            pending_pcm: bytes | None = None

            while self._running:
                if pending_pcm is not None:
                    # Previous turn was interrupted — the captured bytes are the
                    # current turn's input (no need to wait for next segment).
                    segment_bytes = pending_pcm
                    pending_pcm = None
                else:
                    # Wait for the next natural speech segment
                    got_segment = False
                    async for segment_bytes in speech_segments_from_queue(
                        stt_queue,
                        self._vad,
                        min_speech_ms=200,
                        max_silence_ms=700,
                        max_segment_ms=30_000,
                    ):
                        got_segment = True
                        break  # one segment at a time

                    if not got_segment:
                        break  # mic stream ended

                if not self._running:
                    break

                # Drain the interrupt queue so the detector sees fresh frames
                _drain_queue(interrupt_queue)

                captured = await self._handle_segment_with_interruption(
                    segment_bytes, interrupt_queue
                )
                if captured is not None:
                    # Interruption fired; captured bytes are next turn's input
                    pending_pcm = captured if captured else None

        except KeyboardInterrupt:
            pass
        finally:
            fanout_task.cancel()
            try:
                await fanout_task
            except asyncio.CancelledError:
                pass
            self._running = False
            self._status_fn("[끝] 음성 채널을 종료합니다.")
            log.info("voice_channel.stopped")

    async def _handle_segment_with_interruption(
        self,
        pcm_bytes: bytes,
        interrupt_queue: asyncio.Queue[bytes | None],
    ) -> bytes | None:
        """Process one turn with barge-in support.

        Returns:
            None  — TTS completed normally (no interruption).
            bytes — interruption fired; value is the captured prebuffer PCM
                    (may be empty bytes if detector returned empty).
        """
        # 1. STT — emit listening state before transcribe
        self._status_fn("[듣는 중] 음성을 인식하고 있어요...")
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "listening", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )
        try:
            transcript = await self._stt.transcribe(pcm_bytes)
        except Exception as exc:
            log.error("stt_transcribe_failed", error=str(exc))
            return None

        # 2. Skip empty transcripts
        if not transcript.strip():
            log.warning("stt_empty_transcript")
            return None

        log.info("voice_channel.transcript_ready", length=len(transcript))

        # Record user activity for SilenceTrigger.
        try:
            from apps.proactive.activity import record_activity
            record_activity("voice")
        except Exception:  # pragma: no cover
            pass

        # Emit transcript event then thinking state
        _publish(
            self._bus,
            Event(
                type="transcript",
                payload={"text": transcript, "final": True, "channel": "voice"},
                ts=time.monotonic(),
            ),
        )

        # 3. Agent streaming response
        self._status_fn("[생각 중] 답변을 준비하고 있어요...")
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "thinking", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )
        try:
            text_stream = self._agent.stream_respond(
                transcript,
                episode_id=self._episode_id,
                channel="voice",
            )
            if asyncio.iscoroutine(text_stream):
                text_stream = await text_stream  # type: ignore[assignment]
        except Exception as exc:
            log.error("agent_stream_failed", error=str(exc))
            return None

        # Update episode_id before TTS starts (agent exposes last_episode_id)
        if hasattr(self._agent, "last_episode_id"):
            ep_id = getattr(self._agent, "last_episode_id", None)
            if ep_id is not None:
                self._episode_id = ep_id

        # 4a. Quiet mode: drain stream silently, skip TTS entirely
        if self.is_quiet:
            await self._drain_stream(text_stream)
            return None

        # 4b. Build detector
        detector = self._build_detector(interrupt_queue)

        # 5. Race TTS against interruption detector
        self._status_fn("[말하는 중] 음성으로 답변합니다...")
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "speaking", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )

        speak_task = asyncio.create_task(
            self._stream_speak(text_stream), name="speak"
        )
        interrupt_task = asyncio.create_task(
            detector.wait_for_interrupt(), name="interrupt"
        )

        done, pending = await asyncio.wait(
            {speak_task, interrupt_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if interrupt_task in done:
            # Barge-in: stop TTS immediately
            captured = interrupt_task.result()

            speak_task.cancel()
            try:
                await speak_task
            except (asyncio.CancelledError, Exception):
                pass

            await self._speaker.stop()
            self._status_fn("[중단됨] 말을 끊고 다시 듣습니다...")
            log.info("voice_channel.interrupted", captured_bytes=len(captured))
            return captured

        # Normal completion: cancel the interrupt detector
        interrupt_task.cancel()
        try:
            await interrupt_task
        except asyncio.CancelledError:
            pass

        # Retrieve any speak_task exception
        try:
            await speak_task
        except Exception as exc:
            log.error("tts_speak_failed", error=str(exc))

        # Emit response_end + idle after successful TTS completion
        _publish(
            self._bus,
            Event(
                type="response_end",
                payload={"channel": "voice", "episode_id": self._episode_id},
                ts=time.monotonic(),
            ),
        )
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "idle", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )

        return None

    async def _stream_speak(self, text_stream: AsyncIterator[str]) -> None:
        """Collect the text stream and drive TTS → speaker."""
        episode_id, chunks = await self._collect_episode_and_stream(text_stream)
        if episode_id is not None:
            self._episode_id = episode_id
        await self._tts.speak(_iter_list(chunks), self._speaker)

    # ── wake-word loop (Phase 3) ──────────────────────────────────────────────

    async def _run_with_wake_word(self) -> None:
        """Phase 3: sleep until wake word fires, handle active window, repeat.

        Architecture:
        - One ``_FrameFanout`` is created **once** and lives for the entire run.
          It fans the mic stream (consumed once) to three queues:
          * ``stt_queue``       — VAD-gated speech segments for STT.
          * ``wake_queue``      — raw frames for WakeDetector at all times.
          * ``interrupt_queue`` — raw frames for InterruptionDetector during TTS.
        - Outer loop: await wake word → enter active window → return to sleep.
        - Active window: handle utterances until ``wake_active_window_s`` seconds
          of silence, then return to sleeping (re-block on wake word).
        - Quiet mode and interruption apply inside the active window.
        """
        from apps.channels.voice.interrupt import _FrameFanout, speech_segments_from_queue

        wake_detector = self._build_wake_detector()

        # One fanout for the lifetime of this run.
        fanout = _FrameFanout(self._mic)
        stt_queue = fanout.subscribe()
        wake_queue = fanout.subscribe()
        interrupt_queue = fanout.subscribe() if self._enable_interruption else None

        fanout_task = asyncio.create_task(fanout.run(), name="fanout_wake")

        try:
            while self._running:
                # sleep → wake → active-window cycle
                mic_ended = await self._run_wake_cycle(
                    wake_detector=wake_detector,
                    wake_queue=wake_queue,
                    stt_queue=stt_queue,
                    interrupt_queue=interrupt_queue,
                    speech_segments_from_queue_fn=speech_segments_from_queue,
                )
                if mic_ended:
                    break  # mic stream exhausted — stop outer loop

        except KeyboardInterrupt:
            pass
        finally:
            fanout_task.cancel()
            try:
                await fanout_task
            except asyncio.CancelledError:
                pass
            wake_detector.close()
            self._running = False
            self._status_fn("[끝] 음성 채널을 종료합니다.")
            log.info("voice_channel.stopped")

    async def _run_wake_cycle(
        self,
        *,
        wake_detector: Any,
        wake_queue: asyncio.Queue[bytes | None],
        stt_queue: asyncio.Queue[bytes | None],
        interrupt_queue: asyncio.Queue[bytes | None] | None,
        speech_segments_from_queue_fn: Any,
    ) -> bool:
        """One sleep → wake → active-window → sleep cycle.

        Returns ``True`` when the mic stream has ended (EOF sentinel received),
        so the caller can break the outer loop.  Returns ``False`` on a normal
        window-timeout exit (caller should loop to re-sleep on wake word).
        """
        # ── sleeping: wait for wake word ──────────────────────────────────
        self._status_fn("[잠듦] 호출어를 기다립니다...")
        log.info("voice_channel.wake_listening")
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "sleep", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )
        from apps.channels.voice.wake import WakeWordError

        try:
            await wake_detector.wait_for_wake(wake_queue)
        except WakeWordError:
            # Queue closed (mic stream ended) before any wake word detected.
            log.info("voice_channel.wake_queue_closed")
            return True  # treat as EOF — exit outer loop

        if not self._running:
            return True  # stop() called while sleeping

        # ── active window: wake fired ─────────────────────────────────────
        self._status_fn("[깨어남] 말씀하세요.")
        log.info("voice_channel.wake_active")
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "wake", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "listening", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )

        window_timeout = self._wake_active_window_s

        while self._running:
            # Wait for next speech segment with a timeout.
            try:
                segment_bytes = await asyncio.wait_for(
                    self._get_one_segment(stt_queue, speech_segments_from_queue_fn),
                    timeout=window_timeout,
                )
            except asyncio.TimeoutError:
                # No utterance within the window — return to sleeping.
                self._status_fn("[다시 잠듭니다] 호출어를 기다립니다...")
                log.info("voice_channel.wake_window_timeout")
                _publish(
                    self._bus,
                    Event(
                        type="state",
                        payload={"value": "sleep", "channel": "voice"},
                        ts=time.monotonic(),
                    ),
                )
                return False  # not EOF — outer loop re-sleeps

            if segment_bytes is None:
                # Mic stream ended (EOF sentinel propagated through queue).
                return True

            if not self._running:
                return True

            # ── process one utterance ──────────────────────────────────────
            if interrupt_queue is not None:
                _drain_queue(interrupt_queue)
                await self._handle_wake_segment_with_interruption(
                    segment_bytes, interrupt_queue
                )
            else:
                await self._handle_wake_segment_linear(segment_bytes)

        return True  # self._running became False

    async def _get_one_segment(
        self,
        stt_queue: asyncio.Queue[bytes | None],
        speech_segments_from_queue_fn: Any,
    ) -> bytes | None:
        """Return a single speech segment bytes from the queue, or None on EOF."""
        async for segment in speech_segments_from_queue_fn(
            stt_queue,
            self._vad,
            min_speech_ms=200,
            max_silence_ms=700,
            max_segment_ms=30_000,
        ):
            return segment
        return None  # stream ended (sentinel received)

    async def _handle_wake_segment_linear(self, pcm_bytes: bytes) -> None:
        """Process one wake-window utterance without interruption support."""
        from apps.channels.voice.wake import detect_quiet_intent

        transcript = await self._transcribe(pcm_bytes)
        if transcript is None:
            return

        intent = detect_quiet_intent(transcript)
        if intent == "on":
            self.is_quiet = True
            self._status_fn("[조용 모드] 조용히 할게요.")
            log.info("voice_channel.quiet_mode_on")
            _publish(
                self._bus,
                Event(
                    type="state",
                    payload={"value": "quiet", "channel": "voice"},
                    ts=time.monotonic(),
                ),
            )
            return
        if intent == "off":
            self.is_quiet = False
            self._status_fn("[조용 모드 해제] 다시 말씀드릴게요.")
            log.info("voice_channel.quiet_mode_off")
            _publish(
                self._bus,
                Event(
                    type="state",
                    payload={"value": "idle", "channel": "voice"},
                    ts=time.monotonic(),
                ),
            )
            if not self.is_quiet:
                try:
                    await self._tts.speak_text(
                        "네, 다시 말씀드릴게요.", self._speaker
                    )
                except Exception as exc:
                    log.error("tts_ack_failed", error=str(exc))
            return

        text_stream = await self._agent_stream(transcript)
        if text_stream is None:
            return

        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "speaking", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )
        await self._speak_or_drain(text_stream)
        _publish(
            self._bus,
            Event(
                type="response_end",
                payload={"channel": "voice", "episode_id": self._episode_id},
                ts=time.monotonic(),
            ),
        )
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "idle", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )

    async def _handle_wake_segment_with_interruption(
        self,
        pcm_bytes: bytes,
        interrupt_queue: asyncio.Queue[bytes | None],
    ) -> None:
        """Process one wake-window utterance with barge-in interruption."""
        from apps.channels.voice.wake import detect_quiet_intent

        transcript = await self._transcribe(pcm_bytes)
        if transcript is None:
            return

        intent = detect_quiet_intent(transcript)
        if intent == "on":
            self.is_quiet = True
            self._status_fn("[조용 모드] 조용히 할게요.")
            log.info("voice_channel.quiet_mode_on")
            _publish(
                self._bus,
                Event(
                    type="state",
                    payload={"value": "quiet", "channel": "voice"},
                    ts=time.monotonic(),
                ),
            )
            return
        if intent == "off":
            self.is_quiet = False
            self._status_fn("[조용 모드 해제] 다시 말씀드릴게요.")
            log.info("voice_channel.quiet_mode_off")
            _publish(
                self._bus,
                Event(
                    type="state",
                    payload={"value": "idle", "channel": "voice"},
                    ts=time.monotonic(),
                ),
            )
            if not self.is_quiet:
                try:
                    await self._tts.speak_text(
                        "네, 다시 말씀드릴게요.", self._speaker
                    )
                except Exception as exc:
                    log.error("tts_ack_failed", error=str(exc))
            return

        text_stream = await self._agent_stream(transcript)
        if text_stream is None:
            return

        if self.is_quiet:
            await self._drain_stream(text_stream)
            return

        # Race TTS vs interruption detector
        detector = self._build_detector(interrupt_queue)
        self._status_fn("[말하는 중] 음성으로 답변합니다...")
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "speaking", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )

        speak_task = asyncio.create_task(
            self._stream_speak(text_stream), name="speak"
        )
        interrupt_task = asyncio.create_task(
            detector.wait_for_interrupt(), name="interrupt"
        )
        done, _ = await asyncio.wait(
            {speak_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
        )

        if interrupt_task in done:
            speak_task.cancel()
            try:
                await speak_task
            except (asyncio.CancelledError, Exception):
                pass
            await self._speaker.stop()
            self._status_fn("[중단됨] 말을 끊고 다시 듣습니다...")
        else:
            interrupt_task.cancel()
            try:
                await interrupt_task
            except asyncio.CancelledError:
                pass
            try:
                await speak_task
            except Exception as exc:
                log.error("tts_speak_failed", error=str(exc))
            # Emit response_end + idle on normal completion
            _publish(
                self._bus,
                Event(
                    type="response_end",
                    payload={"channel": "voice", "episode_id": self._episode_id},
                    ts=time.monotonic(),
                ),
            )
            _publish(
                self._bus,
                Event(
                    type="state",
                    payload={"value": "idle", "channel": "voice"},
                    ts=time.monotonic(),
                ),
            )

    # ── shared per-turn helpers ───────────────────────────────────────────────

    async def _transcribe(self, pcm_bytes: bytes) -> str | None:
        """Run STT; return transcript or None on error / empty result."""
        self._status_fn(f"[변환 중] 음성 {len(pcm_bytes) // 32000}초 분량을 텍스트로 바꾸고 있어요...")
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "listening", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )
        try:
            transcript = await self._stt.transcribe(pcm_bytes)
        except Exception as exc:
            log.error("stt_transcribe_failed", error=str(exc))
            self._status_fn(f"[STT 오류] {exc}")
            return None
        if not transcript.strip():
            log.warning("stt_empty_transcript")
            self._status_fn("[못 들었어요] 다시 말씀해 주세요.")
            return None
        log.info("voice_channel.transcript_ready", length=len(transcript))
        self._status_fn(f'[들었어요] "{transcript}"')
        _publish(
            self._bus,
            Event(
                type="transcript",
                payload={"text": transcript, "final": True, "channel": "voice"},
                ts=time.monotonic(),
            ),
        )
        return transcript

    async def _agent_stream(self, transcript: str) -> AsyncIterator[str] | None:
        """Call agent.stream_respond; return the text stream or None on error."""
        self._status_fn("[생각 중] 답변을 준비하고 있어요...")
        _publish(
            self._bus,
            Event(
                type="state",
                payload={"value": "thinking", "channel": "voice"},
                ts=time.monotonic(),
            ),
        )
        try:
            text_stream = self._agent.stream_respond(
                transcript,
                episode_id=self._episode_id,
                channel="voice",
            )
            if asyncio.iscoroutine(text_stream):
                text_stream = await text_stream  # type: ignore[assignment]
        except Exception as exc:
            log.error("agent_stream_failed", error=str(exc))
            return None
        return text_stream  # type: ignore[return-value]

    async def _speak_or_drain(self, text_stream: AsyncIterator[str]) -> None:
        """Speak the stream via TTS, or silently drain it if quiet mode is on."""
        if self.is_quiet:
            await self._drain_stream(text_stream)
        else:
            self._status_fn("[답하는 중] 음성으로 답변합니다...")
            try:
                episode_id, chunks = await self._collect_episode_and_stream(text_stream)
                if episode_id is not None:
                    self._episode_id = episode_id
                await self._tts.speak(_iter_list(chunks), self._speaker)
            except Exception as exc:
                log.error("tts_speak_failed", error=str(exc))

    async def _drain_stream(self, text_stream: AsyncIterator[str]) -> None:
        """Consume the agent stream silently (memory update without TTS)."""
        episode_id, _chunks = await self._collect_episode_and_stream(text_stream)
        if episode_id is not None:
            self._episode_id = episode_id
        log.debug("voice_channel.stream_drained_quiet")

    def _build_wake_detector(self) -> Any:
        """Build a WakeDetector for this channel.

        Uses the injected factory (for tests) or constructs a real
        WakeDetector reusing the channel's STT and VAD instances.
        The factory signature is ``(stt, vad, *, keyword) -> WakeDetector``.
        """
        if self._wake_detector_fn is not None:
            return self._wake_detector_fn(self._stt, self._vad, keyword=self._wake_keyword)
        from apps.channels.voice.wake import WakeConfig, WakeDetector

        return WakeDetector(
            stt=self._stt,
            vad=self._vad,
            config=WakeConfig(keyword=self._wake_keyword),
        )

    def _build_detector(self, queue: asyncio.Queue[bytes | None]) -> Any:
        """Build an InterruptionDetector for one turn.

        Uses the injected factory (for tests) or imports at runtime.
        """
        if self._interrupt_detector_fn is not None:
            return self._interrupt_detector_fn(queue)
        from apps.channels.voice.interrupt import InterruptionDetector

        return InterruptionDetector(queue, self._vad)

    # ── shared helpers ────────────────────────────────────────────────────────

    async def _collect_episode_and_stream(
        self,
        text_stream: AsyncIterator[str],
    ) -> tuple[int | None, list[str]]:
        """Consume the text stream, collecting chunks.

        AgentCore.stream_respond is an async generator that yields str chunks.
        We collect them all so TTS can replay from a fresh iterator.
        The AgentCore persists the episode after the last yield, so we read
        the episode_id from the agent's internal state indirectly via the
        protocol — since stream_respond doesn't return episode_id, we keep
        track via a best-effort approach: if the agent exposes last_episode_id
        we use it; otherwise episode continuity relies on passing the same
        episode_id back on the next call.
        """
        chunks: list[str] = []
        async for chunk in text_stream:
            chunks.append(chunk)

        # Try to get episode_id from agent if it exposes it
        episode_id: int | None = None
        if hasattr(self._agent, "last_episode_id"):
            episode_id = getattr(self._agent, "last_episode_id", None)

        return episode_id, chunks

    def _resolve_speech_segments(self) -> _SpeechSegmentsProtocol:
        """Return the speech_segments function to use.

        Uses the injected factory (for tests) or imports at runtime.
        """
        if self._speech_segments_fn is not None:
            return self._speech_segments_fn
        # Runtime import — vad.py must exist when voice channel actually runs
        from apps.channels.voice.vad import speech_segments  # type: ignore[import]

        return speech_segments  # type: ignore[return-value]


# ── helpers ──────────────────────────────────────────────────────────────────


def _default_status(msg: str) -> None:
    """Print status to stderr so it doesn't pollute stdout."""
    print(msg, file=sys.stderr, flush=True)


async def _iter_list(items: list[str]) -> AsyncIterator[str]:  # type: ignore[misc]
    """Yield items from a list as an async iterator."""
    for item in items:
        yield item


def _drain_queue(q: asyncio.Queue[bytes | None]) -> None:
    """Discard all currently-queued frames without blocking.

    Called before starting the interruption detector so stale frames from the
    previous turn don't trigger a false-positive interruption.
    """
    drained = 0
    while not q.empty():
        try:
            q.get_nowait()
            drained += 1
        except asyncio.QueueEmpty:
            break
    if drained:
        log.debug("interrupt_queue_drained", frames=drained)


# ── module-level runner (used by __main__ and presence) ──────────────────────


async def run_voice(bus: "EventBus | None" = None) -> None:
    """Build and run the voice channel from environment.

    Wires MemoryStore + AgentCore + audio I/O + VAD/STT/TTS + VoiceChannel
    using the same env vars as ``python -m apps.channels.voice``. Accepts an
    optional event bus so callers (presence server) can subscribe to state
    events. Used by both the standalone ``__main__`` and the presence server.

    Bus selection precedence (highest to lowest):
      1. Explicit ``bus`` argument — e.g. presence/__main__.py co-process.
      2. ``HER_PRESENCE_URL`` env var → RemoteEventBus (cross-process).
      3. No bus (None) — presence events are silently dropped.
    """
    import os
    import sys
    from pathlib import Path

    from apps.agent.core import AgentCore
    from apps.channels.voice.audio import AudioDeviceError, open_microphone, open_speaker
    from apps.channels.voice.stt import STT
    from apps.channels.voice.tts import GeminiTTS, SayFallbackTTS, TTS, TTSConfig
    from apps.channels.voice.vad import VAD
    from apps.memory.store import MemoryStore

    # Settings file > env var > hardcoded default. Env vars override settings
    # so a one-off `HER_AGENT_MODEL=... make voice` still works.
    from apps.settings import load_settings
    s = load_settings()

    # Resolve bus: caller wins; else check env for cross-process URL.
    if bus is None:
        presence_url = os.environ.get("HER_PRESENCE_URL", "").strip()
        if presence_url:
            from apps.presence.remote_bus import RemoteEventBus

            bus = RemoteEventBus(presence_url)  # type: ignore[assignment]
            print(f"[음성 채널] 프레즌스 서버 연결: {presence_url}", file=sys.stderr)

    db_path = Path(os.environ.get("HER_DB_PATH", "data/db.sqlite"))
    api_key = os.environ.get("GEMINI_API_KEY", "")
    stt_model = os.environ.get("HER_STT_MODEL", s.stt_model)
    agent_model = os.environ.get("HER_AGENT_MODEL", s.agent_model)
    echo_gate_ms = int(os.environ.get("HER_ECHO_GATE_MS", str(s.echo_gate_ms)))

    # TTS as [engine, voice]. Env override: HER_TTS="say:Jian (Premium)" or
    # HER_TTS="gemini-2.5-pro-preview-tts:Kore". Falls back to settings.tts.
    tts_env = os.environ.get("HER_TTS", "").strip()
    if tts_env and ":" in tts_env:
        tts_engine, tts_voice = tts_env.split(":", 1)
    else:
        tts_engine, tts_voice = s.tts[0], s.tts[1]

    if not api_key:
        print(
            "[경고] GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다. "
            "에이전트 응답이 동작하지 않을 수 있습니다.",
            file=sys.stderr,
        )

    store = MemoryStore(db_path)
    agent = AgentCore(store, api_key=api_key or None, model_id=agent_model, bus=bus)

    mic = open_microphone()
    speaker = open_speaker()
    vad = VAD()
    stt = STT(model_size=stt_model, language="ko")

    tts: Any
    if tts_engine == "say":
        # macOS local synthesis — no quota, no network, no fallback voice swap.
        tts = SayFallbackTTS(voice=tts_voice)
        tts_label = f"say -v '{tts_voice}'"
    else:
        tts_config = TTSConfig(voice=tts_voice, model_id=tts_engine)
        primary_tts = GeminiTTS(config=tts_config, api_key=api_key or None)
        fallback_tts = SayFallbackTTS()
        tts = TTS(primary=primary_tts, fallback=fallback_tts)
        tts_label = f"gemini ({tts_voice} / {tts_engine}) + say fallback"

    print(f"[음성 채널] LLM: {agent_model}", file=sys.stderr)
    print(f"[음성 채널] STT 모델: {stt_model}", file=sys.stderr)
    print(f"[음성 채널] TTS: {tts_label}", file=sys.stderr)
    print("[음성 채널] 마이크/스피커: 기본 장치", file=sys.stderr)
    location_name = getattr(s, "location_name", "서울")
    location_lat = getattr(s, "location_lat", 37.5665)
    location_lon = getattr(s, "location_lon", 126.9780)
    print(f"[음성 채널] 위치: {location_name} ({location_lat:.2f}, {location_lon:.2f})", file=sys.stderr)
    print("[음성 채널] 시작합니다. 말씀해주세요.", file=sys.stderr)

    channel = VoiceChannel(
        agent,
        store,
        mic=mic,
        speaker=speaker,
        vad=vad,
        stt=stt,
        tts=tts,
        bus=bus,
        echo_gate_ms=echo_gate_ms,
    )

    try:
        await channel.run()
    except AudioDeviceError as exc:
        print(
            f"마이크/스피커 라이브러리(portaudio)가 없어요. "
            f"brew install portaudio 후 다시 시도하세요. ({exc})",
            file=sys.stderr,
        )
    finally:
        store.close()
