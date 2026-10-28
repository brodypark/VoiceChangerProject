"""pipelines/realtime.py tests. A scripted fake websocket stands in for the network.

FakeWebSocket replays `incoming` messages (dicts are sent as JSON text, exceptions are
raised), then either closes with `close_with` (code, reason) or, with hang=True, waits
until close() is called. FakeConnector records the URL and kwargs (headers) of each connect.
Each test drives its coroutine with asyncio.run (no pytest-asyncio).
"""

from __future__ import annotations

import asyncio
import base64
import json
import ssl
from collections import deque
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

from config import STT_MODEL_ID, TTS_CHUNK_LENGTH_SCHEDULE, TTS_INACTIVITY_TIMEOUT_S, TTS_MODEL_ID
from pipelines.realtime import (STT_FATAL_ERRORS, STT_KEY_HINT, TTS_KEY_HINT, RealtimeError, SttEvent,
                                SttStream, TtsStream)

FAKE_KEY = "FAKE-SECRET-KEY-0000"
FAKE_VOICE = "fake/voice id"
SESSION = {"message_type": "session_started", "session_id": "sess-1", "config": {"model_id": STT_MODEL_ID}}

STT_ERROR_TYPES = (
    "error", "auth_error", "quota_exceeded", "commit_throttled", "unaccepted_terms", "rate_limited",
    "queue_overflow", "resource_exhausted", "session_time_limit_exceeded", "input_error", "invalid_request",
    "chunk_size_exceeded", "insufficient_audio_activity", "transcriber_error",
)


def closed_exc(code: int | None, reason: str = ""):
    if code is None:
        return ConnectionClosedError(None, None)
    cls = ConnectionClosedOK if code in (1000, 1001) else ConnectionClosedError
    return cls(Close(code, reason), None)


class FakeWebSocket:
    def __init__(self, incoming=(), *, close_with: tuple[int | None, str] = (1000, ""), hang: bool = False) -> None:
        self.incoming = deque(incoming)
        self.close_with = close_with
        self.hang = hang
        self.sent: list[str] = []
        self.closed = False
        self._closed_event = asyncio.Event()

    @property
    def sent_json(self) -> list[dict]:
        return [json.loads(m) for m in self.sent]

    async def send(self, data: str) -> None:
        if self.closed:
            raise closed_exc(1000)
        self.sent.append(data)

    async def recv(self):
        await asyncio.sleep(0)
        if self.incoming:
            item = self.incoming.popleft()
            if isinstance(item, BaseException):
                raise item
            return json.dumps(item) if isinstance(item, dict) else item
        if self.hang and not self.closed:
            await self._closed_event.wait()
        if self.closed:
            raise closed_exc(1000)
        raise closed_exc(*self.close_with)

    async def close(self) -> None:
        self.closed = True
        self._closed_event.set()

    async def __aiter__(self):
        while True:
            yield await self.recv()


class FakeConnector:
    def __init__(self, *sockets: FakeWebSocket, error: BaseException | None = None) -> None:
        self.sockets = list(sockets)
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if self.error is not None:
            raise self.error
        return self.sockets.pop(0)


def invalid_status(status: int, body: bytes = b"") -> InvalidStatus:
    headers = Headers({"xi-api-key": FAKE_KEY, "set-cookie": "session=FAKE-COOKIE"})
    return InvalidStatus(Response(status, "Rejected", headers, body))


def stt(*sockets: FakeWebSocket, **kwargs) -> tuple[SttStream, FakeConnector]:
    connector = FakeConnector(*sockets, error=kwargs.pop("error", None))
    return SttStream(FAKE_KEY, connect=connector, **kwargs), connector


def tts(*sockets: FakeWebSocket, **kwargs) -> tuple[TtsStream, FakeConnector]:
    connector = FakeConnector(*sockets, error=kwargs.pop("error", None))
    return TtsStream(FAKE_KEY, FAKE_VOICE, connect=connector, **kwargs), connector


async def collect(aiter) -> tuple[list, RealtimeError | None]:
    items = []
    try:
        async for item in aiter:
            items.append(item)
    except RealtimeError as exc:
        return items, exc
    return items, None


def assert_no_secrets(*things) -> None:
    for thing in things:
        for text in (str(thing), repr(thing)):
            assert FAKE_KEY not in text and "FAKE-COOKIE" not in text


# --- URLs, auth, secrets ---

def test_stt_url_has_all_query_params() -> None:
    stream = SttStream(FAKE_KEY, language_code="en", vad_silence_s=0.5)
    parts = urlsplit(stream.url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    assert {k: v[0] for k, v in parse_qs(parts.query).items()} == {
        "model_id": "scribe_v2_realtime", "audio_format": "pcm_16000", "language_code": "en",
        "commit_strategy": "vad", "vad_silence_threshold_secs": "0.5", "no_verbatim": "true",
        "include_timestamps": "false",
    }
    off = parse_qs(urlsplit(SttStream(FAKE_KEY, no_verbatim=False).url).query)
    assert off["no_verbatim"] == ["false"]
    assert FAKE_KEY not in stream.url


def test_tts_url_has_voice_in_path_and_query() -> None:
    stream = TtsStream(FAKE_KEY, "voice123", output_format="pcm_24000")
    parts = urlsplit(stream.url)
    assert parts.path == "/v1/text-to-speech/voice123/stream-input"
    assert parts.netloc == "api.elevenlabs.io"
    assert {k: v[0] for k, v in parse_qs(parts.query).items()} == {
        "model_id": TTS_MODEL_ID, "output_format": "pcm_24000", "inactivity_timeout": str(TTS_INACTIVITY_TIMEOUT_S),
    }
    assert "/v1/text-to-speech/fake%2Fvoice%20id/stream-input" in TtsStream(FAKE_KEY, FAKE_VOICE).url
    assert FAKE_KEY not in stream.url


def test_auth_header_sent_but_key_never_shown() -> None:
    async def run():
        s, s_conn = stt(FakeWebSocket([SESSION], hang=True))
        await s.connect()
        t, t_conn = tts(FakeWebSocket(hang=True))
        await t.connect()
        return s, s_conn, t, t_conn

    s, s_conn, t, t_conn = asyncio.run(run())
    for stream, connector in ((s, s_conn), (t, t_conn)):
        url, kwargs = connector.calls[0]
        assert url == stream.url
        assert kwargs["additional_headers"] == {"xi-api-key": FAKE_KEY}
        assert isinstance(kwargs["ssl"], ssl.SSLContext)  # certifi CA bundle for wss://
        assert_no_secrets(stream)
        assert stream.is_open


# --- STT: messages sent ---

def test_stt_send_audio_base64_round_trip() -> None:
    pcm = bytes(range(256)) * 10
    ws = FakeWebSocket([SESSION], hang=True)

    async def run():
        s, _ = stt(ws)
        await s.connect()
        await s.send_audio(pcm)
        await s.send_audio(b"")  # skipped

    asyncio.run(run())
    (msg,) = ws.sent_json
    assert msg == {"message_type": "input_audio_chunk", "audio_base_64": msg["audio_base_64"],
                   "commit": False, "sample_rate": 16000}
    assert base64.b64decode(msg["audio_base_64"]) == pcm


def test_stt_send_after_close_raises_non_fatal() -> None:
    async def run():
        s, _ = stt(FakeWebSocket([SESSION], hang=True))
        await s.connect()
        await s.close()
        await s.close()  # idempotent
        assert not s.is_open
        with pytest.raises(RealtimeError) as info:
            await s.send_audio(b"\x00\x00")
        return info.value

    err = asyncio.run(run())
    assert not err.fatal and err.error_type == "closed"


# --- STT: connect ---

def test_stt_connect_waits_for_session_started() -> None:
    warning = {"message_type": "warning", "warning": "audio is quiet"}
    ws = FakeWebSocket([warning, SESSION, {"message_type": "partial_transcript", "text": "hi"}])

    async def run():
        s, _ = stt(ws)
        await s.connect()
        assert s.session_id == "sess-1" and s.session_config == {"model_id": STT_MODEL_ID}
        return await collect(s.events())

    events, err = asyncio.run(run())
    # The warning that arrived before session_started is delivered first, not lost.
    assert events == [SttEvent("warning", "audio is quiet"), SttEvent("partial", "hi")]
    assert err is not None and not err.fatal  # the server closed the socket on its own


def test_stt_connect_times_out() -> None:
    ws = FakeWebSocket(hang=True)

    async def run():
        s, _ = stt(ws, session_timeout=0.05)
        with pytest.raises(RealtimeError) as info:
            await s.connect()
        assert not s.is_open
        return info.value

    err = asyncio.run(run())
    assert not err.fatal and err.error_type == "timeout" and ws.closed


@pytest.mark.parametrize("error_type", ["auth_error", "rate_limited"])
def test_stt_connect_error_before_session(error_type: str) -> None:
    ws = FakeWebSocket([{"message_type": error_type, "error": "nope"}, SESSION])

    async def run():
        s, _ = stt(ws)
        with pytest.raises(RealtimeError) as info:
            await s.connect()
        assert not s.is_open
        return info.value

    err = asyncio.run(run())
    assert err.fatal is (error_type in STT_FATAL_ERRORS)
    assert err.error_type == error_type and "nope" in err.message and ws.closed


def test_stt_connect_server_closes_before_session() -> None:
    async def run():
        s, _ = stt(FakeWebSocket(close_with=(1008, "Quota exceeded")))
        with pytest.raises(RealtimeError) as info:
            await s.connect()
        return info.value

    err = asyncio.run(run())
    assert err.fatal and "credits" in err.message


# --- STT: events ---

def test_stt_parses_every_message_type() -> None:
    ws = FakeWebSocket([
        SESSION,
        {"message_type": "partial_transcript", "text": "hel"},
        {"message_type": "committed_transcript", "text": "Hello."},
        {"message_type": "committed_transcript_with_timestamps", "text": "Bye.", "words": []},
        {"message_type": "warning", "warning": "slow down"},
        {"message_type": "transcriber_error", "error": "hiccup"},
        {"message_type": "session_started", "session_id": "again"},  # ignored
        {"message_type": "some_future_type"},  # ignored
        "not json",
        b"\xff\xfe",
        "[1, 2]",
    ], hang=True)

    async def run():
        s, _ = stt(ws)
        await s.connect()
        events = []
        async for event in s.events():
            events.append(event)
            if len(events) == 5:
                await s.close()
        return events

    events = asyncio.run(run())
    assert events[:4] == [SttEvent("partial", "hel"), SttEvent("committed", "Hello."),
                          SttEvent("committed", "Bye."), SttEvent("warning", "slow down")]
    assert events[4].kind == "error" and events[4].error_type == "transcriber_error"
    assert "hiccup" in events[4].text


@pytest.mark.parametrize("sequence, expected", [
    # (message_type suffix, text) -> committed texts reported
    ([("", "Hi."), ("_ts", "Hi.")], ["Hi."]),
    ([("_ts", "Hi."), ("", "Hi.")], ["Hi."]),
    ([("", "Hi."), ("_ts", "Hi."), ("", "Hi."), ("_ts", "Hi.")], ["Hi.", "Hi."]),  # real repeat
    ([("", "Yes."), ("", "Yes.")], ["Yes.", "Yes."]),  # one variant only: never dropped
    ([("_ts", "Yes."), ("_ts", "Yes.")], ["Yes.", "Yes."]),
    ([("", "One."), ("_ts", "Two.")], ["One.", "Two."]),  # different text
    ([("", "Hi. "), ("_ts", " Hi.")], ["Hi. "]),  # whitespace differences still pair
])
def test_stt_committed_dedup(sequence, expected) -> None:
    types = {"": "committed_transcript", "_ts": "committed_transcript_with_timestamps"}
    msgs = [{"message_type": types[v], "text": text, "words": []} for v, text in sequence]

    async def run():
        s, _ = stt(FakeWebSocket([SESSION, *msgs]))
        await s.connect()
        return await collect(s.events())

    events, _ = asyncio.run(run())
    assert [e.text for e in events if e.kind == "committed"] == expected


@pytest.mark.parametrize("error_type", STT_ERROR_TYPES)
def test_stt_error_mapping(error_type: str) -> None:
    ws = FakeWebSocket([SESSION, {"message_type": error_type, "error": f"detail for {error_type}"}],
                       close_with=(1000, ""))

    async def run():
        s, _ = stt(ws)
        await s.connect()
        events, err = await collect(s.events())
        return s, events, err

    s, events, err = asyncio.run(run())
    fatal = error_type in ("auth_error", "quota_exceeded", "unaccepted_terms", "invalid_request")
    assert err is not None and err.fatal is fatal and not s.is_open
    if fatal:
        assert events == [] and err.error_type == error_type and ws.closed
        assert f"detail for {error_type}" in err.message
    else:
        # Non-fatal errors are reported as events; the following server close then raises a
        # non-fatal error that carries the same type, so the pipeline knows to reconnect.
        (event,) = events
        assert event.kind == "error" and event.error_type == error_type
        assert f"detail for {error_type}" in event.text
        assert err.error_type == error_type
    assert_no_secrets(err, *events)


def test_stt_auth_error_message_has_permission_hint() -> None:
    async def run():
        s, _ = stt(FakeWebSocket([{"message_type": "auth_error", "error": "Invalid API key"}]))
        with pytest.raises(RealtimeError) as info:
            await s.connect()
        return info.value

    assert STT_KEY_HINT in asyncio.run(run()).message


def test_stt_session_time_limit_means_reconnect() -> None:
    async def run():
        s, conn = stt(
            FakeWebSocket([SESSION, {"message_type": "session_time_limit_exceeded", "error": "limit"}],
                          close_with=(1000, "")),
            FakeWebSocket([SESSION], hang=True),
        )
        await s.connect()
        _, err = await collect(s.events())
        await s.connect()  # reconnect on the same object
        return s, conn, err

    s, conn, err = asyncio.run(run())
    assert not err.fatal and err.error_type == "session_time_limit_exceeded"
    assert s.is_open and len(conn.calls) == 2


def test_stt_events_end_quietly_after_close() -> None:
    async def run():
        s, _ = stt(FakeWebSocket([SESSION], hang=True))
        await s.connect()
        task = asyncio.create_task(collect(s.events()))
        await asyncio.sleep(0.01)
        await s.close()
        return await asyncio.wait_for(task, 1)

    assert asyncio.run(run()) == ([], None)


def test_stt_dropped_connection_is_non_fatal() -> None:
    async def run():
        s, _ = stt(FakeWebSocket([SESSION], close_with=(None, "")))
        await s.connect()
        return await collect(s.events())

    _, err = asyncio.run(run())
    assert err is not None and not err.fatal and "connection lost" in err.message


# --- TTS: messages sent ---

def test_tts_connect_sends_init_without_voice_settings() -> None:
    ws = FakeWebSocket(hang=True)

    async def run():
        t, _ = tts(ws)
        await t.connect()
        await t.connect()  # already open: no second init

    asyncio.run(run())
    assert ws.sent_json == [{"text": " ", "generation_config": {"chunk_length_schedule": list(TTS_CHUNK_LENGTH_SCHEDULE)}}]


def test_tts_init_includes_voice_settings_when_given() -> None:
    ws = FakeWebSocket(hang=True)
    settings = {"stability": 0.5, "similarity_boost": 0.8, "speed": 1.0}

    async def run():
        t, _ = tts(ws, voice_settings=settings, chunk_length_schedule=[60, 120])
        await t.connect()

    asyncio.run(run())
    assert ws.sent_json == [{"text": " ", "generation_config": {"chunk_length_schedule": [60, 120]},
                             "voice_settings": settings}]


def test_tts_text_flush_close_messages() -> None:
    ws = FakeWebSocket(hang=True)

    async def run():
        t, _ = tts(ws)
        await t.connect()
        await t.send_text("Hello there.")
        await t.send_text("How are you?  \n")
        await t.send_text("   ")  # blank: skipped, since "" would end the input
        await t.send_text("")
        await t.flush()
        await t.close()
        assert not t.is_open
        await t.close()

    asyncio.run(run())
    assert ws.sent_json[1:] == [{"text": "Hello there. "}, {"text": "How are you? "},
                                {"text": " ", "flush": True}, {"text": ""}]
    assert ws.closed


# --- TTS: audio ---

def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_tts_audio_decodes_and_stops_on_is_final() -> None:
    c1, c2, c3 = b"\x01\x02\x03", b"\x04\x05", b"\x06\x07\x08\x09"
    ws = FakeWebSocket([
        {"audio": b64(c1), "isFinal": None, "normalizedAlignment": {}, "alignment": {}},
        {"audio": None, "isFinal": None},
        {"audio": "", "isFinal": False},
        {"audio": b64(c2), "isFinal": True},
        {"audio": b64(c3), "isFinal": None},
    ], close_with=(1000, ""))

    async def run():
        t, _ = tts(ws)
        await t.connect()
        first = await collect(t.audio())
        still_open = t.is_open
        second = await collect(t.audio())
        return first, still_open, second, t

    first, still_open, second, t = asyncio.run(run())
    assert first == ([c1, c2], None) and still_open
    assert second == ([c3], None)  # then the normal 1000 close ends it quietly
    assert not t.is_open and not t.timed_out


@pytest.mark.parametrize("msg, fatal", [
    ({"error": "quota_exceeded", "message": "This request exceeds your quota.", "code": 1008}, True),
    ({"error": "voice_not_found", "message": "Voice abc not found", "code": 1008}, True),
    ({"error": "invalid_api_key", "message": "Invalid API key", "code": 1008}, True),
    ({"message": "Something went wrong on our side"}, False),
    ({"error": "internal_error"}, False),
])
def test_tts_json_error(msg: dict, fatal: bool) -> None:
    ws = FakeWebSocket([{"audio": b64(b"\x00\x01"), "isFinal": None}, msg], hang=True)

    async def run():
        t, _ = tts(ws)
        await t.connect()
        chunks, err = await collect(t.audio())
        return t, chunks, err

    t, chunks, err = asyncio.run(run())
    assert chunks == [b"\x00\x01"]
    assert err is not None and err.fatal is fatal and not t.is_open and ws.closed
    if fatal and "key" in msg["message"].lower():
        assert TTS_KEY_HINT in err.message
    assert_no_secrets(err)


@pytest.mark.parametrize("code, reason, fatal", [
    (1008, "Quota exceeded. You have 0 credits remaining.", True),
    (1008, "Invalid API key", True),
    (1008, "A voice with voice_id xyz does not exist.", True),
    (1011, "Internal server error", False),
    (None, "", False),
])
def test_tts_close_code_errors(code: int | None, reason: str, fatal: bool) -> None:
    async def run():
        t, _ = tts(FakeWebSocket(close_with=(code, reason)))
        await t.connect()
        chunks, err = await collect(t.audio())
        return t, err

    t, err = asyncio.run(run())
    assert err is not None and err.fatal is fatal and not t.is_open


@pytest.mark.parametrize("incoming, close_with", [
    ([], (1008, "Have not received new text for 180 seconds. Closing the connection.")),
    ([{"message": "Input timeout exceeded", "error": "input_timeout_exceeded", "code": 1008}], (1008, "")),
])
def test_tts_inactivity_close_does_not_raise(incoming, close_with) -> None:
    async def run():
        t, conn = tts(FakeWebSocket(incoming, close_with=close_with), FakeWebSocket(hang=True))
        await t.connect()
        result = await collect(t.audio())
        state = (t.is_open, t.timed_out)
        await t.connect()  # lazily reopened
        return result, state, t, conn

    result, state, t, conn = asyncio.run(run())
    assert result == ([], None)
    assert state == (False, True)
    assert t.is_open and not t.timed_out and len(conn.calls) == 2


def test_tts_audio_ends_quietly_after_close() -> None:
    async def run():
        t, _ = tts(FakeWebSocket(hang=True))
        await t.connect()
        task = asyncio.create_task(collect(t.audio()))
        await asyncio.sleep(0.01)
        await t.close()
        return await asyncio.wait_for(task, 1)

    assert asyncio.run(run()) == ([], None)


def test_tts_send_after_server_close_raises_closed() -> None:
    async def run():
        t, _ = tts(FakeWebSocket(close_with=(1000, "")))
        await t.connect()
        await collect(t.audio())
        with pytest.raises(RealtimeError) as info:
            await t.send_text("Hi")
        return info.value

    err = asyncio.run(run())
    assert not err.fatal and err.error_type == "closed"


# --- handshake failures ---

@pytest.mark.parametrize("factory", [stt, tts])
@pytest.mark.parametrize("status", [401, 403])
def test_handshake_auth_failure_is_fatal(factory, status: int) -> None:
    body = json.dumps({"detail": {"status": "invalid_api_key", "message": "Invalid API key"}}).encode()

    async def run():
        stream, _ = factory(error=invalid_status(status, body))
        with pytest.raises(RealtimeError) as info:
            await stream.connect()
        return stream, info.value

    stream, err = asyncio.run(run())
    assert err.fatal and err.error_type == f"http_{status}" and not stream.is_open
    assert (STT_KEY_HINT if factory is stt else TTS_KEY_HINT) in err.message
    assert "Invalid API key" in err.message
    assert err.__cause__ is None and err.__suppress_context__  # the original exception (with headers) is dropped
    assert_no_secrets(err, stream)


@pytest.mark.parametrize("error, fatal, error_type", [
    (invalid_status(402), True, "quota_exceeded"),
    (invalid_status(404, b'{"detail": {"message": "Voice not found"}}'), True, "http_404"),
    (invalid_status(429), False, "http_429"),
    (invalid_status(503, b"<html>down</html>"), False, "http_503"),
    (OSError("nodename nor servname provided"), False, "network"),
    (ConnectionRefusedError("refused"), False, "network"),
    (TimeoutError(), False, "timeout"),
])
def test_handshake_other_failures(error, fatal: bool, error_type: str) -> None:
    async def run():
        t, _ = tts(error=error)
        with pytest.raises(RealtimeError) as info:
            await t.connect()
        return info.value

    err = asyncio.run(run())
    assert err.fatal is fatal and err.error_type == error_type
    assert_no_secrets(err)


def test_default_connect_is_websockets() -> None:
    from websockets.asyncio.client import connect

    assert SttStream(FAKE_KEY)._connect is connect
    assert TtsStream(FAKE_KEY, "v")._connect is connect
