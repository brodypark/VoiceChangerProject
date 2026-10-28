"""CLI entry point for the real-time AI voice changer.

Modes: `--list-devices`, `--mode passthrough` (mic -> BlackHole, no AI),
`--mode sts` (Pipeline B: VAD phrases -> ElevenLabs speech-to-speech -> BlackHole) and
`--mode stt-tts` (Pipeline A: realtime speech-to-text -> streaming text-to-speech -> BlackHole, the default).
The GUI (`python -m gui`) runs the same VoiceSession (session.py).

Threads: PortAudio callbacks capture/play audio, the pipeline has its own worker
thread, and the main thread is the only one that prints (it drains the session's events).
"""

from __future__ import annotations

import sys
import time
from typing import Sequence

from audio_io import AudioIOError
from config import Config, ConfigError, load_config
from devices import DeviceError, get_devices, list_devices
from pipelines.base import ApiFatalError, check_account, make_client
from session import METER_WIDTH, LevelMeter, VoiceSession, resolve_devices  # noqa: F401 - re-exported for tests

CLEAR_LINE = "\r\x1b[2K"  # carriage return + ANSI "erase line", so messages don't mix with the meter


def run_session(config: Config) -> int:
    """Run one session in the terminal until Ctrl+C (or until the pipeline stops itself)."""
    # Module-level names are passed in (and time is looked up per call) so tests can replace them on main.
    session = VoiceSession(config, get_devices=get_devices, make_client=make_client, check_account=check_account,
                           clock=lambda: time.monotonic())
    for line in session.prepare():
        print(line)

    try:
        info = session.start()
        for warning in info.warnings:
            print(warning, file=sys.stderr)
        print(f"{session.title}: {info.route}")
        steps = " ".join(s for s in (session.how_to, "Ctrl+C to stop.") if s)
        print(f"In Discord/Zoom/etc., choose '{info.output_name}' as the microphone. {steps}")
        while True:
            time.sleep(0.1)
            for line in session.poll():
                print(f"{CLEAR_LINE}{line}")
            if session.done:  # cap reached / fatal error, and its last audio finished playing
                break
            print(f"{CLEAR_LINE}mic level [{session.meter.bar()}]", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for line in session.stop():
            print(f"{CLEAR_LINE}{line}")
    # A fatal error was already printed as an "ERROR: ..." event line.
    return 2 if session.error is not None else 0


def run_passthrough(config: Config) -> int:
    """Mic -> BlackHole (and optional monitor) with no AI."""
    return run_session(config)


def run_sts(config: Config) -> int:
    """Pipeline B: speak a phrase, hear it back in your ElevenLabs voice on BlackHole."""
    return run_session(config)


def run_stt_tts(config: Config) -> int:
    """Pipeline A: realtime transcription -> your ElevenLabs voice reads it out."""
    return run_session(config)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = load_config(argv)

        if config.list_devices:
            print(list_devices())
            return 0
        if config.mode == "passthrough":
            return run_passthrough(config)
        if config.mode == "sts":
            return run_sts(config)
        return run_stt_tts(config)
    except (ConfigError, DeviceError, AudioIOError, ApiFatalError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
