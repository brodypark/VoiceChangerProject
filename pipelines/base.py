"""Shared pipeline interface, ElevenLabs client setup, and API error messages.

Every pipeline takes 16 kHz mono int16 little-endian PCM in 20 ms frames via
feed_audio() (called on the PortAudio callback thread, so it must never block)
and delivers converted mono int16 PCM at `output_sample_rate` through the
`on_audio_out` callback (called from the pipeline's own worker thread).

Threads: worker threads never print. They put user-facing lines on `events`,
which the main thread drains and prints. A pipeline that stops itself (credit
cap reached, fatal API error) sets `finished`; on a fatal error `error` is set too.

Error messages never include the API key or any request/response headers
(ApiError.__str__ includes response headers, so it is never used here).
"""

from __future__ import annotations

import dataclasses
import queue
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable

import httpx
from elevenlabs import ElevenLabs
from elevenlabs.core.api_error import ApiError
from elevenlabs.core.parse_error import ParsingError

from config import Config

MAX_API_MESSAGE_CHARS = 300  # error bodies can be whole HTML pages on 5xx
RETRY_DELAY_S = 0.5  # wait before the one retry of a phrase after a temporary error (5xx, 429, network)
KEY_HINT = ("Check ELEVENLABS_API_KEY in .env and that the key has the Speech to Speech "
            "(Voice Changer) permission enabled.")


class ApiFatalError(Exception):
    """An ElevenLabs error that retrying won't fix (bad key, no credits, unknown voice). User-facing message."""


class Pipeline(ABC):
    """Mic frames in, converted audio out. Implementations: Pipeline A (stt_tts.py), Pipeline B (sts.py)."""

    config: Config
    input_sample_rate: int  # Hz of the frames passed to feed_audio()
    output_sample_rate: int  # Hz of the PCM passed to on_audio_out

    def __init__(self, on_audio_out: Callable[[bytes], None]) -> None:
        self.on_audio_out = on_audio_out
        self.events: queue.Queue[str] = queue.Queue()  # user-facing lines for the main thread to print
        self.error: Exception | None = None  # set on a fatal error (the pipeline has stopped)
        self.finished = threading.Event()  # set when the pipeline stopped itself

    @abstractmethod
    def start(self) -> None:
        """Start background work. Idempotent."""

    @abstractmethod
    def stop(self) -> None:
        """Stop background work promptly. Safe to call more than once."""

    @abstractmethod
    def feed_audio(self, frame: bytes) -> None:
        """Accept one 20 ms input frame. Runs on the audio callback thread: must never block."""

    def set_voice(self, voice_id: str) -> None:
        """Switch voices while running: the next phrase uses `voice_id`. Callable from any thread.

        Swapping the (immutable) config is one atomic assignment, and each phrase reads its voice
        when it starts, so a phrase already being spoken finishes in the old voice.
        """
        self.config = dataclasses.replace(self.config, voice_id=voice_id)

    def _emit(self, msg: str) -> None:
        self.events.put_nowait(msg)

    def drain_events(self) -> list[str]:
        """Return and remove all pending event lines (main thread)."""
        lines = []
        while True:
            try:
                lines.append(self.events.get_nowait())
            except queue.Empty:
                return lines


# --- ElevenLabs helpers ---

def make_client(api_key: str) -> ElevenLabs:
    return ElevenLabs(api_key=api_key)


def _parse_error_body(body: Any) -> tuple[str, str]:
    """Return (API message, lowercase status/code/type tags) from any error body shape.

    Shapes seen: {"detail": {"status", "code", "type", "message"}}, {"detail": [<422
    validation items>]}, {"detail": "text"}, a bare string, or None.
    """
    detail = body.get("detail", body) if isinstance(body, dict) else body
    tags = ""
    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("msg") or "")
        tags = " ".join(str(detail.get(k) or "") for k in ("status", "code", "type"))
    elif isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(x) for x in item.get("loc", []) if x != "body")
                msg = str(item.get("msg") or item.get("message") or "")
                parts.append(f"{loc}: {msg}" if loc else msg)
            else:
                parts.append(str(item))
        message = "; ".join(p for p in parts if p)
    elif detail is None:
        message = ""
    else:
        message = str(detail)
    message = " ".join(message.split())  # collapse newlines/whitespace
    if len(message) > MAX_API_MESSAGE_CHARS:
        message = message[:MAX_API_MESSAGE_CHARS] + "..."
    return message, tags.lower()


def _is_missing_permission(exc: Exception) -> bool:
    if not isinstance(exc, ApiError) or exc.status_code not in (401, 403):
        return False
    message, tags = _parse_error_body(exc.body)
    return "permission" in message.lower() or "permission" in tags


def describe_api_error(exc: Exception) -> tuple[str, bool]:
    """Map an exception from an ElevenLabs call to (user-facing message, is_fatal)."""
    if isinstance(exc, (ApiError, ParsingError)):
        status = exc.status_code
        message, tags = _parse_error_body(exc.body)
        said = f": {message}" if message else ""

        # Quota errors historically arrive as 401 with status "quota_exceeded", so check them first.
        if status == 402 or "quota" in tags or "credit" in tags or "quota" in message.lower():
            return f"Out of ElevenLabs credits{said}. Check your usage at elevenlabs.io.", True
        if status in (401, 403):
            if _is_missing_permission(exc):
                return f"ElevenLabs API key is missing a permission{said}. {KEY_HINT}", True
            return f"ElevenLabs rejected the API key{said}. {KEY_HINT}", True
        if status == 404:
            return f"Voice not found{said}. Check ELEVENLABS_VOICE_ID in .env.", True
        if status in (400, 422):
            hint = ""
            if any(word in message.lower() for word in ("output_format", "tier", "subscription")):
                hint = " Your plan may not allow this output format; try --output-format pcm_24000."
            return f"ElevenLabs rejected the request ({status}){said}.{hint}", True
        if status == 429:
            return f"ElevenLabs rate/concurrency limit hit{said}. Slow down or wait a moment.", False
        if status is not None and 500 <= status < 600:
            return f"ElevenLabs server error ({status}){said}. Try again shortly.", False
        return f"ElevenLabs API error ({status}){said}.", False

    if isinstance(exc, (httpx.HTTPError, ConnectionError, TimeoutError)):
        return f"Network problem talking to ElevenLabs ({type(exc).__name__}: {exc}). Check your connection.", False
    return f"Unexpected error ({type(exc).__name__}: {exc}).", False


def lookup_voice(client: ElevenLabs, voice_id: str) -> str | None:
    """Name of a voice (free call), or None if the key has no voices_read permission (can't check).

    Raises ApiFatalError with a user-facing message for anything else (unknown voice, bad key, network).
    """
    try:
        voice = client.voices.get(voice_id)
    except Exception as exc:  # noqa: BLE001 - classified below
        if _is_missing_permission(exc):
            return None
        if isinstance(exc, ApiError) and exc.status_code in (400, 404, 422):
            raise ApiFatalError(f"No voice with ID {voice_id!r} in your ElevenLabs account. Copy it from "
                                "ElevenLabs > Voices > (your voice) > Copy voice ID.") from None
        msg, _ = describe_api_error(exc)
        raise ApiFatalError(msg) from None
    return getattr(voice, "name", None) or voice_id


def check_account(client: ElevenLabs, voice_id: str) -> list[str]:
    """Free startup check (no credits): returns info/warning lines; raises ApiFatalError on a bad key or voice.

    A key without read permissions can still be allowed to use Speech to Speech,
    so missing-permission errors here are only warnings.
    """
    lines: list[str] = []
    try:
        voice = client.voices.get(voice_id)
        lines.append(f"Voice: {getattr(voice, 'name', None) or voice_id}")
    except Exception as exc:  # noqa: BLE001 - classified below
        if _is_missing_permission(exc):
            lines.append("WARNING: can't verify the voice (the API key has no voices_read permission). "
                         "Continuing; a wrong voice ID will show up on the first phrase.")
        else:
            msg, fatal = describe_api_error(exc)
            if fatal:
                raise ApiFatalError(msg) from None
            lines.append(f"WARNING: couldn't verify the voice: {msg}")
            if isinstance(exc, (httpx.HTTPError, ConnectionError, TimeoutError)):
                return lines  # the subscription call would fail the same way

    try:
        sub = client.user.subscription.get()
        used, limit = sub.character_count, sub.character_limit
        lines.append(f"Credits: {used:,} of {limit:,} used ({max(0, limit - used):,} left, {sub.tier} tier)")
    except Exception as exc:  # noqa: BLE001
        if not _is_missing_permission(exc):
            msg, _ = describe_api_error(exc)
            lines.append(f"WARNING: couldn't read the credit balance: {msg}")
    return lines
