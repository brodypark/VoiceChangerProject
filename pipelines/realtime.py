"""Asyncio websocket clients for Pipeline A: realtime speech-to-text and streaming text-to-speech.

SttStream: ElevenLabs Scribe v2 Realtime (wss .../v1/speech-to-text/realtime)
    In:  16 kHz mono int16 LE PCM ("pcm_16000"), sent as base64 inside JSON
         {"message_type": "input_audio_chunk", "audio_base_64", "commit": false, "sample_rate"}.
         The server's VAD (commit_strategy=vad) decides when a segment is committed.
    Out: SttEvent("partial" | "committed" | "warning" | "error").
         Committed text can come in two variants: committed_transcript and
         committed_transcript_with_timestamps. Both map to "committed", and a pair that
         repeats the same text is reported once (see SttStream._committed_event).

TtsStream: ElevenLabs text-to-speech websocket "stream-input" (wss .../v1/text-to-speech/{voice_id}/stream-input)
    In:  init {"text": " ", "generation_config": {...}}, then {"text": "words "} pieces,
         {"text": " ", "flush": true} at the end of a segment (socket stays open) and
         {"text": ""} to end the input.
    Out: raw PCM chunks in `output_format` (e.g. pcm_48000: mono int16 LE at 48 kHz),
         decoded from base64. Chunks can have any length, including odd ones.

Async flow: one task sends (send_audio / send_text) while another iterates
events() / audio(); websockets allows one concurrent sender and one receiver.
Auth is the `xi-api-key` handshake header. Both clients take an injectable
`connect` callable (default: websockets.asyncio.client.connect) so tests can use a fake.

Errors: every failure is a RealtimeError with a user-facing message, `fatal`
(retrying won't help: bad key, no credits, unknown voice, rejected request) and
`error_type`. Messages and reprs never include the API key or any headers.

Closure:
    SttStream is meant to stay open. If the server closes it (any code), events()
    raises a non-fatal RealtimeError so the pipeline reconnects;
    session_time_limit_exceeded arrives that way too. events() ends quietly only after close().
    TtsStream closes itself after `inactivity_timeout` s without text. That close (and
    a normal 1000 close) just ends audio() with `is_open` False and `timed_out` set
    for the inactivity case; the pipeline reopens with connect().
"""

from __future__ import annotations

import asyncio
import base64
import json
import ssl
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, Mapping, Sequence
from urllib.parse import quote, urlencode

import certifi

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus, InvalidURI

from config import (DEFAULT_OUTPUT_FORMAT, LANGUAGE_CODE, SAMPLE_RATE, STT_MODEL_ID, STT_VAD_SILENCE_S,
                    TTS_CHUNK_LENGTH_SCHEDULE, TTS_INACTIVITY_TIMEOUT_S, TTS_MODEL_ID)
from pipelines.base import MAX_API_MESSAGE_CHARS, _parse_error_body

STT_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
TTS_BASE_URL = "wss://api.elevenlabs.io"
OPEN_TIMEOUT_S = 10.0  # TCP + TLS + websocket handshake
SESSION_START_TIMEOUT_S = 10.0  # handshake done -> session_started
MAX_MESSAGE_BYTES = 16 * 2**20  # TTS audio arrives base64-encoded; a long pcm_48000 chunk can pass the 1 MiB default

STT_KEY_HINT = "Check ELEVENLABS_API_KEY in .env and that the key has the Speech to Text permission enabled."
TTS_KEY_HINT = "Check ELEVENLABS_API_KEY in .env and that the key has the Text to Speech permission enabled."

# STT server error types that retrying won't fix. Every other error type is non-fatal.
STT_FATAL_ERRORS = frozenset({"auth_error", "quota_exceeded", "unaccepted_terms", "invalid_request"})

ConnectFn = Callable[..., Awaitable[Any]]


class RealtimeError(Exception):
    """A realtime websocket failure with a user-facing message (never contains the key or headers)."""

    def __init__(self, message: str, *, fatal: bool, error_type: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.fatal = fatal  # True: stop the pipeline; False: reconnect/retry
        self.error_type = error_type  # server error type, "closed", "timeout", "network", "http_<status>"...


@dataclass(frozen=True)
class SttEvent:
    kind: Literal["partial", "committed", "warning", "error"]
    text: str = ""  # transcript, warning text, or the user-facing error message
    error_type: str | None = None  # only for kind="error" (non-fatal server errors; fatal ones raise)


# --- shared helpers ---

def _short(text: Any) -> str:
    text = " ".join(str(text or "").split())
    return text[:MAX_API_MESSAGE_CHARS] + "..." if len(text) > MAX_API_MESSAGE_CHARS else text


def _loads(raw: str | bytes) -> dict | None:
    """Parse one JSON object message; anything else (binary noise, arrays) is ignored."""
    try:
        msg = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    return msg if isinstance(msg, dict) else None


def _classify(text: str) -> str | None:
    """Tag free-form server text (close reasons, TTS error messages): quota/auth/voice/inactivity/None."""
    t = text.lower()
    if "quota" in t or "credit" in t:
        return "quota"
    if any(w in t for w in ("api key", "api_key", "unauthori", "authenticat", "auth_error", "permission")):
        return "auth"
    if "voice" in t and any(w in t for w in ("not found", "not_found", "does not exist", "invalid voice")):
        return "voice"
    if "timeout" in t or "inactiv" in t or "not received" in t:
        return "inactivity"
    return None


def _fatal_message(tag: str, service: str, key_hint: str, said: str) -> str:
    if tag == "quota":
        return f"Out of ElevenLabs credits{said}. Check your usage at elevenlabs.io."
    if tag == "auth":
        return f"{service} rejected the API key{said}. {key_hint}"
    return f"Voice not found{said}. Check ELEVENLABS_VOICE_ID in .env."


def _close_info(exc: ConnectionClosed) -> tuple[int | None, str]:
    """(code, reason) of the close frame the server sent; (None, "") if the connection just dropped."""
    if exc.rcvd is None:
        return None, ""
    return exc.rcvd.code, exc.rcvd.reason


def _handshake_error(exc: BaseException, service: str, key_hint: str) -> RealtimeError:
    """Map a failed websocket connect to RealtimeError. Uses only the status and body, never headers."""
    if isinstance(exc, InvalidStatus):
        status = exc.response.status_code
        body: Any = bytes(exc.response.body or b"")
        try:
            body = json.loads(body)
        except ValueError:
            body = body.decode("utf-8", "replace")
        message, tags = _parse_error_body(body)
        said = f": {message}" if message else ""
        tag = _classify(f"{message} {tags}")
        if status == 402 or tag == "quota":
            return RealtimeError(_fatal_message("quota", service, key_hint, said), fatal=True, error_type="quota_exceeded")
        if status in (401, 403):
            return RealtimeError(f"{service} rejected the API key (HTTP {status}){said}. {key_hint}",
                                 fatal=True, error_type=f"http_{status}")
        if status == 404 and tag == "voice":
            return RealtimeError(_fatal_message("voice", service, key_hint, said), fatal=True, error_type="http_404")
        if status in (400, 404, 422):
            return RealtimeError(f"{service} rejected the connection request (HTTP {status}){said}.",
                                 fatal=True, error_type=f"http_{status}")
        if status == 429:
            return RealtimeError(f"{service}: rate/concurrency limit hit (HTTP 429){said}. Wait a moment.",
                                 fatal=False, error_type="http_429")
        return RealtimeError(f"{service} connection failed (HTTP {status}){said}. Try again shortly.",
                             fatal=False, error_type=f"http_{status}")
    if isinstance(exc, InvalidURI):
        return RealtimeError(f"{service}: invalid websocket URL ({_short(exc)}).", fatal=True, error_type="invalid_url")
    if isinstance(exc, TimeoutError):
        return RealtimeError(f"{service}: timed out connecting to ElevenLabs. Check your connection.",
                             fatal=False, error_type="timeout")
    return RealtimeError(f"Network problem connecting to {service} ({type(exc).__name__}: {_short(exc)}). "
                         "Check your connection.", fatal=False, error_type="network")


class _Socket:
    """Connection bookkeeping shared by both clients."""

    service = "ElevenLabs"
    key_hint = ""

    def __init__(self, api_key: str, connect: ConnectFn | None) -> None:
        self._headers = {"xi-api-key": api_key}  # only ever passed to connect(), never shown
        self._connect = connect or ws_connect
        self._ws: Any = None
        self._closing = False  # close() was called: closes from here on are expected

    @property
    def url(self) -> str:
        raise NotImplementedError

    @property
    def is_open(self) -> bool:
        return self._ws is not None and not self._closing

    def __repr__(self) -> str:
        return f"{type(self).__name__}(url={self.url!r}, is_open={self.is_open})"

    async def _open(self) -> None:
        if self.is_open:
            return
        self._closing = False
        kwargs: dict[str, Any] = {"additional_headers": dict(self._headers), "open_timeout": OPEN_TIMEOUT_S,
                                  "max_size": MAX_MESSAGE_BYTES}
        if self.url.startswith("wss://"):
            # python.org's macOS Python has no CA store until "Install Certificates" is run;
            # use certifi's bundle (the one httpx/the elevenlabs SDK already use).
            kwargs["ssl"] = ssl.create_default_context(cafile=certifi.where())
        try:
            self._ws = await self._connect(self.url, **kwargs)
        except (InvalidHandshake, InvalidURI, OSError, TimeoutError) as exc:
            self._ws = None
            raise _handshake_error(exc, self.service, self.key_hint) from None

    async def _send_json(self, msg: dict) -> None:
        if not self.is_open:
            raise RealtimeError(f"{self.service} connection is closed.", fatal=False, error_type="closed")
        try:
            await self._ws.send(json.dumps(msg))
        except ConnectionClosed as exc:
            self._drop()
            raise self._close_error(exc) from None

    def _drop(self) -> None:
        self._ws = None

    async def close(self) -> None:
        """Close the socket. Safe to call more than once."""
        ws, self._closing = self._ws, True
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:  # noqa: BLE001 - already closing; nothing useful to report
            pass
        finally:
            self._ws = None

    def _close_error(self, exc: ConnectionClosed) -> RealtimeError:
        code, reason = _close_info(exc)
        said = f": {_short(reason)}" if reason else ""
        tag = _classify(reason)
        if tag in ("quota", "auth", "voice"):
            return RealtimeError(_fatal_message(tag, self.service, self.key_hint, said), fatal=True,
                                 error_type=f"close_{code}")
        what = f"code {code}" if code is not None else "connection lost"
        return RealtimeError(f"{self.service} connection closed ({what}){said}.", fatal=False,
                             error_type="closed")


# --- Speech-to-text ---

class SttStream(_Socket):
    """Scribe v2 Realtime client: connect(), send_audio() from one task, iterate events() in another."""

    service = "Realtime speech-to-text"
    key_hint = STT_KEY_HINT

    def __init__(self, api_key: str, *, model_id: str = STT_MODEL_ID, language_code: str = LANGUAGE_CODE,
                 vad_silence_s: float = STT_VAD_SILENCE_S, no_verbatim: bool = True,
                 sample_rate: int = SAMPLE_RATE, connect: ConnectFn | None = None, url: str = STT_URL,
                 session_timeout: float = SESSION_START_TIMEOUT_S) -> None:
        super().__init__(api_key, connect)
        self.base_url = url
        self.model_id = model_id
        self.language_code = language_code
        self.vad_silence_s = vad_silence_s
        self.no_verbatim = no_verbatim
        self.sample_rate = sample_rate
        self.session_timeout = session_timeout
        self.session_id: str | None = None
        self.session_config: dict = {}  # as echoed by session_started
        self._pending: deque[SttEvent] = deque()  # events that arrived while connect() waited
        self._last_committed: tuple[str, str] | None = None  # (message_type, stripped text)
        self._last_error_type: str | None = None

    @property
    def url(self) -> str:
        query = {
            "model_id": self.model_id,
            "audio_format": f"pcm_{self.sample_rate}",
            "language_code": self.language_code,
            "commit_strategy": "vad",
            "vad_silence_threshold_secs": self.vad_silence_s,
            "no_verbatim": str(self.no_verbatim).lower(),
            "include_timestamps": "false",
        }
        return f"{self.base_url}?{urlencode(query)}"

    async def connect(self) -> None:
        """Open the socket and wait for session_started. Raises RealtimeError (and closes) on failure."""
        if self.is_open:
            return
        await self._open()
        self.session_id, self.session_config = None, {}
        self._pending.clear()
        self._last_committed = self._last_error_type = None
        try:
            async with asyncio.timeout(self.session_timeout):
                while True:
                    msg = _loads(await self._ws.recv())
                    if msg is None:
                        continue
                    if msg.get("message_type") == "session_started":
                        self.session_id = msg.get("session_id")
                        self.session_config = msg.get("config") or {}
                        return
                    event = self._parse(msg)  # raises on a fatal error
                    if event is not None and event.kind == "error":
                        raise RealtimeError(event.text, fatal=False, error_type=event.error_type)
                    if event is not None:
                        self._pending.append(event)
        except TimeoutError:
            await self.close()
            raise RealtimeError(f"{self.service} didn't start a session within {self.session_timeout:g} s. "
                                "Try again.", fatal=False, error_type="timeout") from None
        except ConnectionClosed as exc:
            self._drop()
            raise self._close_error(exc) from None
        except RealtimeError:
            await self.close()
            raise

    async def send_audio(self, pcm: bytes) -> None:
        """Send one chunk of 16-bit mono PCM at `sample_rate` (no commit; the server's VAD commits)."""
        if not pcm:
            return
        await self._send_json({
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm).decode("ascii"),
            "commit": False,
            "sample_rate": self.sample_rate,
        })

    async def events(self) -> AsyncIterator[SttEvent]:
        """Yield transcript events until close(). Raises RealtimeError on a fatal error or a server close."""
        while self._pending:
            yield self._pending.popleft()
        while self._ws is not None:
            try:
                raw = await self._ws.recv()
            except ConnectionClosed as exc:
                self._drop()
                if self._closing:
                    return
                raise self._server_close_error(exc) from None
            msg = _loads(raw)
            if msg is None:
                continue
            try:
                event = self._parse(msg)
            except RealtimeError:
                await self.close()
                raise
            if event is not None:
                yield event

    def _server_close_error(self, exc: ConnectionClosed) -> RealtimeError:
        """A close we didn't ask for. Reuses the last server error (e.g. session_time_limit_exceeded)."""
        error = self._close_error(exc)
        if not error.fatal and self._last_error_type:
            code, _ = _close_info(exc)
            error = RealtimeError(f"{self.service} session ended ({self._last_error_type}, code {code}); "
                                  "reconnecting.", fatal=False, error_type=self._last_error_type)
        return error

    def _parse(self, msg: dict) -> SttEvent | None:
        """Map one server message to an event (None = nothing to report). Raises on fatal errors."""
        kind = str(msg.get("message_type") or "")
        if kind == "partial_transcript":
            return SttEvent("partial", str(msg.get("text") or ""))
        if kind in ("committed_transcript", "committed_transcript_with_timestamps"):
            return self._committed_event(kind, str(msg.get("text") or ""))
        if kind == "warning":
            return SttEvent("warning", _short(msg.get("warning") or msg.get("message")))
        if "error" in msg or kind.endswith("error") or kind in STT_ERROR_MESSAGES:
            error_type = kind or "error"
            self._last_error_type = error_type
            message = _stt_error_message(error_type, msg.get("error") or msg.get("message"))
            if error_type in STT_FATAL_ERRORS:
                raise RealtimeError(message, fatal=True, error_type=error_type)
            return SttEvent("error", message, error_type)
        return None  # session_started again, or a message type added later

    def _committed_event(self, kind: str, text: str) -> SttEvent | None:
        """De-duplicate committed transcripts.

        The server may send the same segment as both committed_transcript and
        committed_transcript_with_timestamps, in either order. A committed message is
        dropped only when the previous committed message was the OTHER variant with
        the same text; the pair is then consumed, so a sentence the user really
        repeats is still reported (plain, plain -> both; plain, ts, plain, ts -> two).
        """
        key = text.strip()
        last = self._last_committed
        if last is not None and last[0] != kind and last[1] == key:
            self._last_committed = None
            return None
        self._last_committed = (kind, key)
        return SttEvent("committed", text)


STT_ERROR_MESSAGES = {
    "auth_error": "Realtime speech-to-text rejected the API key{said}. " + STT_KEY_HINT,
    "quota_exceeded": "Out of ElevenLabs credits{said}. Check your usage at elevenlabs.io.",
    "unaccepted_terms": "Accept the ElevenLabs terms of service for speech-to-text at elevenlabs.io{said}.",
    "invalid_request": "Realtime speech-to-text rejected the request{said}.",
    "rate_limited": "Realtime speech-to-text rate limit hit{said}. Wait a moment.",
    "queue_overflow": "Realtime speech-to-text is falling behind (queue overflow){said}.",
    "resource_exhausted": "Realtime speech-to-text is busy (resource exhausted){said}. Try again shortly.",
    "session_time_limit_exceeded": "Realtime speech-to-text session time limit reached{said}; reconnecting.",
    "commit_throttled": "Realtime speech-to-text commits are throttled{said}.",
    "chunk_size_exceeded": "Audio chunk too large for realtime speech-to-text{said}.",
    "insufficient_audio_activity": "Realtime speech-to-text heard too little audio{said}.",
}


def _stt_error_message(error_type: str, detail: Any) -> str:
    said = f": {_short(detail)}" if detail else ""
    template = STT_ERROR_MESSAGES.get(error_type, "Realtime speech-to-text error ({error_type}){said}.")
    return template.format(error_type=error_type, said=said)


# --- Text-to-speech ---

class TtsStream(_Socket):
    """stream-input client: connect(), send_text()/flush() from one task, iterate audio() in another.

    audio() can be called again after it ends on isFinal while the socket is still
    open; check `is_open` to tell "segment done" from "socket closed, reconnect".
    """

    service = "Text-to-speech stream"
    key_hint = TTS_KEY_HINT

    def __init__(self, api_key: str, voice_id: str, *, model_id: str = TTS_MODEL_ID,
                 output_format: str = DEFAULT_OUTPUT_FORMAT,
                 chunk_length_schedule: Sequence[int] = TTS_CHUNK_LENGTH_SCHEDULE,
                 inactivity_timeout: int = TTS_INACTIVITY_TIMEOUT_S,
                 voice_settings: Mapping[str, Any] | None = None, connect: ConnectFn | None = None,
                 base_url: str = TTS_BASE_URL) -> None:
        super().__init__(api_key, connect)
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.chunk_length_schedule = list(chunk_length_schedule)
        self.inactivity_timeout = inactivity_timeout
        self.voice_settings = dict(voice_settings) if voice_settings else None
        self.base_url = base_url
        self.timed_out = False  # the server closed the socket for inactivity (normal; reopen lazily)

    @property
    def url(self) -> str:
        query = urlencode({"model_id": self.model_id, "output_format": self.output_format,
                           "inactivity_timeout": self.inactivity_timeout})
        return f"{self.base_url}/v1/text-to-speech/{quote(self.voice_id, safe='')}/stream-input?{query}"

    async def connect(self) -> None:
        """Open the socket and send the init message. Raises RealtimeError on failure."""
        if self.is_open:
            return
        await self._open()
        self.timed_out = False
        init: dict[str, Any] = {"text": " ", "generation_config": {"chunk_length_schedule": self.chunk_length_schedule}}
        if self.voice_settings:
            init["voice_settings"] = self.voice_settings  # only allowed here; omitted = the voice's stored settings
        await self._send_json(init)

    async def send_text(self, text: str) -> None:
        """Send words to speak. Trailing whitespace becomes one space; blank text is skipped
        (an empty "text" would end the input)."""
        text = text.rstrip()
        if text:
            await self._send_json({"text": text + " "})

    async def flush(self) -> None:
        """Generate everything buffered now (end of a segment); the socket stays open.

        The docs don't specify the text of a flush message; " " is used because it is
        the no-op text of the init message, while "" means end of input.
        """
        await self._send_json({"text": " ", "flush": True})

    async def close(self) -> None:
        """Send end-of-input ({"text": ""}) and close the socket right away (audio not yet received is lost)."""
        if self.is_open:
            try:
                await self._ws.send(json.dumps({"text": ""}))
            except Exception:  # noqa: BLE001 - closing anyway
                pass
        await super().close()

    async def audio(self) -> AsyncIterator[bytes]:
        """Yield decoded PCM chunks until isFinal or the socket closes.

        A normal close (1000/1001, our close(), or the inactivity timeout) just ends the
        iteration. Other closes and JSON error messages raise RealtimeError.
        """
        while self._ws is not None:
            try:
                raw = await self._ws.recv()
            except ConnectionClosed as exc:
                self._drop()
                code, reason = _close_info(exc)
                if self._closing or code in (1000, 1001):
                    return
                if _classify(reason) == "inactivity":
                    self.timed_out = True
                    return
                raise self._close_error(exc) from None
            msg = _loads(raw)
            if msg is None:
                continue
            if msg.get("audio"):
                yield base64.b64decode(msg["audio"])
            elif msg.get("error") or (msg.get("message") and "audio" not in msg):
                await self._handle_error(msg)
                return
            if msg.get("isFinal"):
                return

    async def _handle_error(self, msg: dict) -> None:
        """A JSON error message: inactivity ends quietly, anything else closes and raises."""
        error, detail = msg.get("error"), msg.get("message")
        tag = _classify(f"{error or ''} {detail or ''}")
        await self.close()
        if tag == "inactivity":
            self.timed_out = True
            return
        said = f": {_short(detail or error)}"
        error_type = str(error) if error else None
        if tag in ("quota", "auth", "voice"):
            raise RealtimeError(_fatal_message(tag, self.service, self.key_hint, said), fatal=True,
                                error_type=error_type or tag)
        raise RealtimeError(f"{self.service} error{said}.", fatal=False, error_type=error_type)
