"""Voice channel package.

Re-exports the public audio API so other engineers can write:

    from apps.channels.voice import (
        AudioFormat, AudioInputStream, AudioOutputStream,
        FakeMicrophone, FakeSpeaker,
        open_microphone, open_speaker,
    )

Concrete sounddevice implementations (SoundDeviceMicrophone,
SoundDeviceSpeaker) are available via `apps.channels.voice.audio` directly.

Other modules in this package (vad, stt, tts, channel) are owned by
separate engineers and will be added in Phase 1.
"""

from apps.channels.voice.audio import (
    AudioDeviceError,
    AudioFormat,
    AudioInputStream,
    AudioOutputStream,
    FakeMicrophone,
    FakeSpeaker,
    open_microphone,
    open_speaker,
)

__all__ = [
    "AudioDeviceError",
    "AudioFormat",
    "AudioInputStream",
    "AudioOutputStream",
    "FakeMicrophone",
    "FakeSpeaker",
    "open_microphone",
    "open_speaker",
]
