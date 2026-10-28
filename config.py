"""Configuration: loads .env + CLI args into one immutable Config object.

Audio format (used end to end; any conversion happens only in audio_io.py):
    16 kHz, mono, signed 16-bit little-endian PCM ("pcm_16000").
    16 kHz is accepted by Scribe v2 Realtime and the speech-to-speech endpoint,
    and is one of the few rates webrtcvad supports (8/16/32/48 kHz).
    Frames are 20 ms (320 samples = 640 bytes); webrtcvad only accepts 10/20/30 ms.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

# --- ElevenLabs model IDs (verified against elevenlabs.io/docs/models, 2026-10-27) ---
STT_MODEL_ID = "scribe_v2_realtime"  # Pipeline A: realtime speech-to-text
STT_BATCH_MODEL_ID = "scribe_v2"  # Pipeline B word check: is this phrase real speech or just "hmm"/a sigh?
LANGUAGE_CODE = "en"
TTS_MODEL_ID = "eleven_flash_v2_5"  # Pipeline A: text-to-speech websocket
STS_MODEL_ID = "eleven_english_sts_v2"  # Pipeline B: speech-to-speech (Voice Changer)

# --- Audio format ---
SAMPLE_RATE = 16_000  # Hz
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (int16)
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH  # 640

# Passthrough (no AI) skips VAD/STT, so it runs at full quality instead of 16 kHz.
PASSTHROUGH_SAMPLE_RATE = 48_000

# --- AI output (both pipelines) ---
# Raw PCM output formats only (no decoding needed). pcm_44100 requires the Pro tier.
OUTPUT_FORMATS = ("pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100", "pcm_48000")
DEFAULT_OUTPUT_FORMAT = "pcm_48000"

# --- Pipeline B (speech-to-speech) ---
# `optimize_streaming_latency` is marked deprecated in the docs (2026-10) but still accepted.
STS_LATENCY_OPTIMIZATION = 3
DEFAULT_MAX_SPEECH_S = 60.0  # credit safety cap: seconds of speech sent per run (0 = no cap)

# --- Pipeline A (realtime STT -> TTS websocket) ---
STT_SEND_CHUNK_MS = 100  # mic audio is batched into ~100 ms websocket messages
STT_VAD_SILENCE_S = 0.5  # server-side silence that commits a segment (same feel as Pipeline B)
# --pause: silence that ends a phrase, in both AI modes (sts: local VAD; stt-tts: the STT server's VAD).
# Shorter = less latency, but a mid-sentence breath can split the sentence. The docs (2026-10) give no
# range for vad_silence_threshold_secs, so this is a conservative one.
DEFAULT_PAUSE_S = STT_VAD_SILENCE_S
PAUSE_RANGE = (0.3, 1.5)
PARTIAL_HOLDBACK_WORDS = 2  # newest words of a partial transcript are still likely to change; don't speak them yet
# "Balanced" timing: TTS starts generating once ~50 characters (6-10 words) are buffered, or on a flush at a pause.
TTS_CHUNK_LENGTH_SCHEDULE = (50, 90, 120, 160)
TTS_INACTIVITY_TIMEOUT_S = 180  # max allowed; the TTS socket closes after this much silence and is reopened lazily

# Timing (--timing):
#   smooth   = each phrase is spoken whole after you pause; one streaming HTTP request per phrase, so any model
#              works and the voice can plan the whole sentence (natural rhythm). Default.
#   balanced = starts mid-sentence over the TTS websocket: (holdback words, chunk_length_schedule) below.
#   fast     = like balanced, starts sooner, choppier joins.
TIMINGS = ("smooth", "balanced", "fast")
DEFAULT_TIMING = "smooth"
STREAMING_TIMINGS = {"balanced": (PARTIAL_HOLDBACK_WORDS, TTS_CHUNK_LENGTH_SCHEDULE), "fast": (1, (50, 70, 100, 140))}

# TTS model for stt-tts (--tts-model). Measured 2026-10-27 on the same sentence: v4 turbo ~150 ms to first audio
# at the user's natural pace (3.5 s, matching the sts output); Flash v2.5 also ~150 ms but ~30% faster speech;
# v4 ~440 ms; v3 ~1.1 s. Only Flash is accepted on the word-by-word websocket (v4 turbo: HTTP 400 there).
TTS_MODELS = ("eleven_v4_turbo", "eleven_v4", "eleven_v3", "eleven_flash_v2_5")
SMOOTH_TTS_MODEL = "eleven_v4_turbo"
WEBSOCKET_TTS_MODELS = (TTS_MODEL_ID,)
# Instant voice clones keep more of their likeness with a higher similarity boost (voice default is 0.75).
TTS_VOICE_SETTINGS = {"stability": 0.5, "similarity_boost": 0.9, "style": 0.0, "use_speaker_boost": True, "speed": 1.0}
# v3/v4 models follow inline delivery tags like "[calm, relaxed] text" without reading them aloud (tested
# 2026-10-28: this tag brought v4 turbo's median pitch from ~180 Hz to ~161 Hz, close to the user's sts output).
# Flash would read the tag out loud, so it's only added for these models. Stability barely affected v4 turbo.
DELIVERY_TAG_MODELS = ("eleven_v4_turbo", "eleven_v4", "eleven_v3")
DEFAULT_DELIVERY = "calm, relaxed, conversational"
SPEED_RANGE = (0.7, 1.2)  # ElevenLabs voice_settings.speed limits
# --stability / --similarity (0-1). None = stt-tts uses TTS_VOICE_SETTINGS, sts uses the voice's stored settings.

MODES = ("passthrough", "stt-tts", "sts")
DEFAULT_MODE = "stt-tts"
DEFAULT_OUTPUT_DEVICE = "BlackHole 2ch"

ENV_PATH = Path(__file__).resolve().parent / ".env"


class ConfigError(Exception):
    """Raised for missing/invalid configuration, with a user-facing message."""


@dataclass(frozen=True)
class Config:
    api_key: str | None
    voice_id: str | None
    mode: str
    input_device: str | None  # name substring or index as string; None = system default
    output_device: str  # virtual mic (BlackHole)
    monitor_device: str | None  # headphones; None = monitor off
    vad_aggressiveness: int  # webrtcvad 0 (least) .. 3 (most aggressive)
    dry_run: bool
    list_devices: bool
    output_format: str = DEFAULT_OUTPUT_FORMAT
    max_speech_s: float = DEFAULT_MAX_SPEECH_S  # 0 = no cap
    denoise: bool = False
    word_check: bool = True  # drop phrases with no real words (hmm, uh, sighs) via speech-to-text
    timing: str = DEFAULT_TIMING  # stt-tts only: smooth | balanced | fast
    tts_model_override: str | None = None  # stt-tts only: --tts-model; None = pick for the timing
    delivery: str = DEFAULT_DELIVERY  # stt-tts only: delivery tag for v3/v4 models ("" = none)
    speed: float = 1.0  # stt-tts only: voice speed, 0.7-1.2
    stability: float | None = None  # both AI modes; None = default (see TTS_VOICE_SETTINGS)
    similarity: float | None = None  # both AI modes: similarity_boost; None = default
    pause_s: float = DEFAULT_PAUSE_S  # both AI modes: silence that ends a phrase
    latency_log: str | None = None  # CSV file to append one row per utterance to; None = off

    @property
    def tts_model(self) -> str:
        """TTS model for stt-tts: --tts-model if given, else v4 turbo (smooth) or Flash (websocket timings)."""
        if self.tts_model_override:
            return self.tts_model_override
        return SMOOTH_TTS_MODEL if self.timing == "smooth" else TTS_MODEL_ID

    @property
    def output_rate(self) -> int:
        """Sample rate of the AI voice output, parsed from e.g. 'pcm_48000'."""
        return int(self.output_format.split("_")[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Real-time AI voice changer: mic -> ElevenLabs voice -> virtual mic (BlackHole).",
    )
    parser.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    parser.add_argument("--mode", choices=MODES, default=None, help=f"pipeline (default: {DEFAULT_MODE})")
    parser.add_argument("--input", dest="input_device", default=None, help="mic name substring or index")
    parser.add_argument(
        "--output", dest="output_device", default=None, help=f"virtual mic (default: {DEFAULT_OUTPUT_DEVICE})"
    )
    parser.add_argument(
        "--monitor", dest="monitor_device", default=None, help="also play to this device (headphones only!)"
    )
    parser.add_argument("--vad", dest="vad_aggressiveness", type=int, choices=range(4), default=3,
                        help="VAD aggressiveness 0-3; higher ignores more non-speech (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="run without calling ElevenLabs (no credits used); phrases are echoed back as-is")
    parser.add_argument("--output-format", dest="output_format", choices=OUTPUT_FORMATS,
                        default=DEFAULT_OUTPUT_FORMAT,
                        help=f"AI voice audio quality (default: {DEFAULT_OUTPUT_FORMAT})")
    parser.add_argument("--max-speech", dest="max_speech_s", type=float, default=DEFAULT_MAX_SPEECH_S,
                        help=f"stop after this many seconds of speech are sent (default: {DEFAULT_MAX_SPEECH_S:.0f}, "
                             "0 = no cap)")
    parser.add_argument("--denoise", action="store_true",
                        help="ask ElevenLabs to remove background noise (adds latency)")
    parser.add_argument("--timing", choices=TIMINGS, default=DEFAULT_TIMING,
                        help="stt-tts: smooth = speak each phrase after you pause (natural, default); "
                             "balanced/fast = start mid-sentence (Flash model only, choppier)")
    parser.add_argument("--tts-model", dest="tts_model_override", choices=TTS_MODELS, default=None,
                        help=f"stt-tts voice model (default: {SMOOTH_TTS_MODEL} for smooth, "
                             f"{TTS_MODEL_ID} for balanced/fast)")
    parser.add_argument("--delivery", default=DEFAULT_DELIVERY,
                        help=f'stt-tts: how the voice should sound, for v3/v4 models (default: "{DEFAULT_DELIVERY}"; '
                             '"" = off). E.g. "low, steady voice, laid-back"')
    parser.add_argument("--speed", type=float, default=1.0,
                        help=f"stt-tts: voice speed {SPEED_RANGE[0]}-{SPEED_RANGE[1]} (default: 1.0)")
    parser.add_argument("--stability", type=float, default=None,
                        help="voice stability 0-1: lower = more expressive, higher = more even "
                             f"(default: {TTS_VOICE_SETTINGS['stability']} for stt-tts, the voice's own setting for sts)")
    parser.add_argument("--similarity", type=float, default=None,
                        help="voice similarity boost 0-1: how closely to match your voice "
                             f"(default: {TTS_VOICE_SETTINGS['similarity_boost']} for stt-tts, the voice's own setting for sts)")
    parser.add_argument("--pause", dest="pause_s", type=float, default=DEFAULT_PAUSE_S,
                        help=f"seconds of silence that end a phrase, {PAUSE_RANGE[0]}-{PAUSE_RANGE[1]} "
                             f"(default: {DEFAULT_PAUSE_S}); shorter = faster, but may split sentences")
    parser.add_argument("--latency-log", dest="latency_log", default=None, metavar="CSV",
                        help="append each utterance's latency breakdown to this CSV file")
    parser.add_argument("--no-word-check", dest="word_check", action="store_false",
                        help="don't drop non-word sounds (hmm, uh, sighs); skips the speech-to-text check")
    return parser


def load_config(argv: Sequence[str] | None = None, env_path: Path = ENV_PATH) -> Config:
    """Parse CLI args and .env. Env vars override nothing that the CLI sets explicitly."""
    args = build_parser().parse_args(argv)
    load_dotenv(env_path, override=False)

    mode = args.mode or os.getenv("VOICE_CHANGER_MODE", DEFAULT_MODE)
    if mode not in MODES:
        raise ConfigError(f"Invalid mode {mode!r}. Use one of: {', '.join(MODES)}.")
    if args.max_speech_s < 0:
        raise ConfigError("--max-speech must be >= 0 (0 = no cap).")
    if not SPEED_RANGE[0] <= args.speed <= SPEED_RANGE[1]:
        raise ConfigError(f"--speed must be between {SPEED_RANGE[0]} and {SPEED_RANGE[1]}, got {args.speed}.")
    for flag, value in (("--stability", args.stability), ("--similarity", args.similarity)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ConfigError(f"{flag} must be between 0 and 1, got {value}.")
    if not PAUSE_RANGE[0] <= args.pause_s <= PAUSE_RANGE[1]:
        raise ConfigError(f"--pause must be between {PAUSE_RANGE[0]} and {PAUSE_RANGE[1]} s, got {args.pause_s}.")
    if args.timing != "smooth" and args.tts_model_override not in (None, *WEBSOCKET_TTS_MODELS):
        raise ConfigError(f"--timing {args.timing} streams word by word, which only works with "
                          f"{', '.join(WEBSOCKET_TTS_MODELS)}. Use --timing smooth with {args.tts_model_override}.")

    return Config(
        api_key=os.getenv("ELEVENLABS_API_KEY") or None,
        voice_id=os.getenv("ELEVENLABS_VOICE_ID") or None,
        mode=mode,
        input_device=args.input_device,
        output_device=args.output_device or DEFAULT_OUTPUT_DEVICE,
        monitor_device=args.monitor_device,
        vad_aggressiveness=args.vad_aggressiveness,
        dry_run=args.dry_run,
        list_devices=args.list_devices,
        output_format=args.output_format,
        max_speech_s=args.max_speech_s,
        denoise=args.denoise,
        word_check=args.word_check,
        timing=args.timing,
        tts_model_override=args.tts_model_override,
        delivery=args.delivery.strip().strip("[]").strip(),
        speed=args.speed,
        stability=args.stability,
        similarity=args.similarity,
        pause_s=args.pause_s,
        latency_log=args.latency_log,
    )


def voice_settings(config: Config) -> dict:
    """Full TTS voice settings: TTS_VOICE_SETTINGS with --speed, --stability and --similarity applied."""
    settings = {**TTS_VOICE_SETTINGS, "speed": config.speed}
    if config.stability is not None:
        settings["stability"] = config.stability
    if config.similarity is not None:
        settings["similarity_boost"] = config.similarity
    return settings


def require_credentials(config: Config) -> None:
    """Call before any ElevenLabs request. Never includes the key itself in messages."""
    missing = [name for name, value in (("ELEVENLABS_API_KEY", config.api_key),
                                        ("ELEVENLABS_VOICE_ID", config.voice_id)) if not value]
    if missing:
        raise ConfigError(
            f"Missing {', '.join(missing)}. Copy .env.example to .env and fill it in."
        )
