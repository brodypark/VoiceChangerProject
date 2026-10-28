"""pipelines/stt_tts.py tests. Fake STT/TTS streams stand in for the websockets and a fake client for the
HTTP text-to-speech stream; no network, no audio hardware.

Most tests run the websocket path (--timing balanced, make_config's default here); the "smooth timing"
section covers the default timing, where FakeTtsClient stands in for client.text_to_speech.stream.

FakeStt / FakeTts mimic SttStream / TtsStream. Each connection is a new fake made by a
Factory (like the pipeline's real factories), so tests can count reconnects. Tests drive a
fake from the main thread with push(), which hands items to the fake's asyncio queue on the
pipeline's loop thread: SttEvents (or exceptions) for events(), PCM chunks (or exceptions)
for audio(). CLOSED ends the iteration as a server close; END ends one audio() call while
the socket stays open (like an isFinal).

Synthetic speech for the local VAD is seeded white noise (std 0.3 full scale) and silence is
zeros, with the RMS detector (use_webrtc=False) so frame counts are exact.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import httpx
import numpy as np
import pytest

import pipelines.stt_tts as stt_tts
from elevenlabs import VoiceSettings
from elevenlabs.core.api_error import ApiError

from config import (FRAME_BYTES, PARTIAL_HOLDBACK_WORDS, SAMPLE_RATE, SMOOTH_TTS_MODEL, STREAMING_TIMINGS,
                    STT_MODEL_ID, TTS_CHUNK_LENGTH_SCHEDULE, TTS_INACTIVITY_TIMEOUT_S, TTS_MODEL_ID,
                    TTS_VOICE_SETTINGS, Config, ConfigError)
from latency import LatencyTracker
from pipelines.base import ApiFatalError
from pipelines.realtime import RealtimeError, SttEvent, SttStream, TtsStream
from pipelines.stt_tts import SEND_CHUNK_BYTES, SttTtsPipeline
from pipelines.text_stream import TextStabilizer
from vad import VoiceActivityDetector, iter_frames

BYTES_PER_MS = SAMPLE_RATE // 1000 * 2
FAKE_KEY = "FAKE-KEY-not-a-real-key"
FAKE_VOICE = "fake-voice-id"
WAIT_S = 1.0
FLUSH = "<flush>"
CLOSED = object()
END = object()


def silence(ms: int) -> bytes:
    return bytes(ms * BYTES_PER_MS)


def speech(ms: int, seed: int = 0) -> bytes:
    noise = np.random.default_rng(seed).normal(0, 0.3, ms * SAMPLE_RATE // 1000)
    return (np.clip(noise, -1, 1) * 32767).astype("<i2").tobytes()


def feed(pipeline: SttTtsPipeline, pcm: bytes) -> None:
    frames, rest = iter_frames(pcm, FRAME_BYTES)
    assert rest == b""
    for frame in frames:
        pipeline.feed_audio(frame)


def wait_for(predicate, timeout: float = WAIT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def make_config(**overrides) -> Config:
    values = dict(api_key=FAKE_KEY, voice_id=FAKE_VOICE, mode="stt-tts", input_device=None,
                  output_device="BlackHole 2ch", monitor_device=None, vad_aggressiveness=2,
                  dry_run=False, list_devices=False, timing="balanced", delivery="")  # tag tests set it explicitly
    return Config(**{**values, **overrides})


def partial(text: str) -> SttEvent:
    return SttEvent("partial", text)


def committed(text: str) -> SttEvent:
    return SttEvent("committed", text)


def closed_error(msg: str = "Text-to-speech stream connection is closed.") -> RealtimeError:
    return RealtimeError(msg, fatal=False, error_type="closed")


class FakeStream:
    """Shared connect/close/push plumbing for the fakes."""

    def __init__(self, connect_error: BaseException | None = None) -> None:
        self.connect_error = connect_error
        self.connected = threading.Event()
        self.closed = threading.Event()  # close() was called by the pipeline
        self.server_closed = False  # the "server" closed the socket
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None

    @property
    def is_open(self) -> bool:
        return self.connected.is_set() and not self.closed.is_set() and not self.server_closed

    async def connect(self) -> None:
        await asyncio.sleep(0)
        if self.connect_error is not None:
            raise self.connect_error
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self.connected.set()

    async def close(self) -> None:
        self.closed.set()
        if self._queue is not None:
            self._queue.put_nowait(CLOSED)

    def push(self, *items) -> None:
        """Main thread: queue items for events()/audio() on the pipeline's loop."""
        assert self.connected.wait(WAIT_S)
        for item in items:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def _items(self):
        while True:
            item = await self._queue.get()
            if item is CLOSED:
                self.server_closed = True
                return
            if item is END:
                return
            if isinstance(item, BaseException):
                self.server_closed = True
                raise item
            yield item


class FakeStt(FakeStream):
    def __init__(self, connect_error: BaseException | None = None) -> None:
        super().__init__(connect_error)
        self.sent: list[bytes] = []

    async def send_audio(self, pcm: bytes) -> None:
        if not self.is_open:
            raise closed_error("Realtime speech-to-text connection is closed.")
        self.sent.append(pcm)

    def events(self):
        return self._items()


class FakeTts(FakeStream):
    def __init__(self, connect_error: BaseException | None = None) -> None:
        super().__init__(connect_error)
        self.sent: list[str] = []  # texts as sent, FLUSH for each flush
        self.timed_out = False
        self.fail_sends = False  # simulate the socket closing just before the next send

    async def send_text(self, text: str) -> None:
        if self.fail_sends:
            self.server_closed = True
        if not self.is_open:
            raise closed_error()
        text = text.rstrip()
        if text:
            self.sent.append(text + " ")

    async def flush(self) -> None:
        if not self.is_open:
            raise closed_error()
        self.sent.append(FLUSH)

    def audio(self):
        return self._items()

    def server_close(self, timed_out: bool = True) -> None:
        self.timed_out = timed_out
        self.push(CLOSED)


class Factory:
    """Makes one fake per connection attempt; `connect_errors` are used by the first attempts, in order."""

    def __init__(self, cls, *connect_errors: BaseException) -> None:
        self.cls = cls
        self.errors = list(connect_errors)
        self.made: list = []

    def __call__(self):
        stream = self.cls(self.errors.pop(0) if self.errors else None)
        self.made.append(stream)
        return stream

    def wait(self, n: int = 1):
        """The n-th fake (1-based), once it is connected."""
        assert wait_for(lambda: len(self.made) >= n and self.made[n - 1].connected.is_set())
        return self.made[n - 1]


class Sink:
    """on_audio_out recorder that can be waited on."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self._lock = threading.Lock()

    def __call__(self, pcm: bytes) -> None:
        with self._lock:
            self.chunks.append(pcm)

    def wait(self, n: int) -> bool:
        return wait_for(lambda: len(self.chunks) >= n)


class FakeTtsClient:
    """Stands in for ElevenLabs(): `.text_to_speech.stream(voice_id, **kwargs)` returns a lazy generator.

    `script` holds one entry per call: a list of chunks to yield, or an exception to raise on the
    first next() (like the SDK, where the request goes out lazily). A threading.Event in a chunk
    list blocks the generator until it is set. Unscripted calls yield DEFAULT_CHUNKS. Every chunk
    is preceded by `delay` seconds. `active`/`max_active` count generators being iterated at once.
    """

    DEFAULT_CHUNKS = [b"\x01\x02"]

    def __init__(self, *script: Any, delay: float = 0.0) -> None:
        self.text_to_speech = self
        self.script = list(script)
        self.delay = delay
        self.calls: list[dict] = []  # {"voice_id": ..., **kwargs} per stream() call
        self.closed: list[int] = []  # indexes of calls whose generator finished or was closed
        self.active = self.max_active = 0
        self._lock = threading.Lock()

    def stream(self, voice_id: str, **kwargs):
        with self._lock:
            index = len(self.calls)
            self.calls.append({"voice_id": voice_id, **kwargs})
        item = self.script.pop(0) if self.script else self.DEFAULT_CHUNKS
        return self._generate(index, item)

    def _generate(self, index: int, item: Any):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if isinstance(item, BaseException):
                raise item
            for chunk in item:
                if isinstance(chunk, threading.Event):
                    assert chunk.wait(5 * WAIT_S)
                    continue
                if self.delay:
                    time.sleep(self.delay)
                yield chunk
        finally:
            with self._lock:
                self.active -= 1
                self.closed.append(index)

    def texts(self) -> list[str]:
        return [call["text"] for call in self.calls]


class Harness:
    def __init__(self, pipeline: SttTtsPipeline, stt: Factory, tts: Factory, sink: Sink,
                 client: FakeTtsClient) -> None:
        self.p, self.stt, self.tts, self.sink, self.client = pipeline, stt, tts, sink, client


@pytest.fixture
def make():
    """Factory for started pipelines with fake streams and the RMS detector; stops each one at teardown."""
    made: list[SttTtsPipeline] = []

    def build(config: Config | None = None, *, start: bool = True, stt: Factory | None = None,
              tts: Factory | None = None, latency: LatencyTracker | None = None, sink=None,
              client: FakeTtsClient | None = None) -> Harness:
        stt = stt or Factory(FakeStt)
        tts = tts or Factory(FakeTts)
        sink = sink or Sink()
        client = client or FakeTtsClient()
        p = SttTtsPipeline(config or make_config(), sink, latency=latency, stt_factory=stt, tts_factory=tts,
                           tts_client=client, vad=VoiceActivityDetector(use_webrtc=False))
        made.append(p)
        if start:
            p.start()
        return Harness(p, stt, tts, sink, client)

    yield build
    for p in made:
        p.stop()


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(stt_tts, "STT_RECONNECT_DELAYS_S", (0.01,))


def events_until(pipeline: SttTtsPipeline, needle: str, timeout: float = WAIT_S) -> list[str]:
    """Drain events until one contains `needle` (or time out); returns everything drained."""
    seen: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        seen += pipeline.drain_events()
        if any(needle in line for line in seen):
            return seen
        time.sleep(0.001)
    return seen


def wait_sent(tts: FakeTts, n_flushes: int = 1) -> bool:
    return wait_for(lambda: tts.sent.count(FLUSH) >= n_flushes)


# --- mic audio -> STT ---

def test_frames_batched_into_100ms_chunks(make):
    h = make()
    stt = h.stt.wait()
    frames = [bytes([i]) * FRAME_BYTES for i in range(1, 14)]
    for frame in frames:
        h.p.feed_audio(frame)
    assert wait_for(lambda: len(stt.sent) == 2)
    time.sleep(0.02)
    assert SEND_CHUNK_BYTES == 5 * FRAME_BYTES
    assert stt.sent == [b"".join(frames[:5]), b"".join(frames[5:10])]  # the last 3 frames wait for a full batch


def test_frames_before_start_are_dropped(make):
    h = make(start=False)
    feed(h.p, silence(200))
    h.p.start()
    stt = h.stt.wait()
    feed(h.p, silence(100))
    assert wait_for(lambda: len(stt.sent) == 1)
    assert stt.sent == [silence(100)]


# --- transcripts -> TTS ---

def test_partials_and_commit_send_stabilizer_text_and_one_flush(make):
    h = make()
    partials = ["Hello", "Hello there", "Hello there how", "Hello there how are", "Hello there how are you"]
    final = "Hello there, how are you?"
    stab = TextStabilizer()
    expected = [t for t in (stab.on_partial(x) for x in partials) if t] + [stab.on_committed(final)]
    assert expected == ["Hello ", "there ", "how ", "are you? "]

    h.stt.wait().push(*map(partial, partials), committed(final))
    tts = h.tts.wait()
    assert wait_sent(tts)
    time.sleep(0.02)
    assert tts.sent == expected + [FLUSH]
    assert h.p.tts_chars_sent == 25
    assert events_until(h.p, "heard") == ['heard: "Hello there, how are you?" (25 chars spoken)']


def test_each_commit_gets_its_own_flush(make):
    h = make()
    h.stt.wait().push(committed("One sentence."), committed("And another one."))
    tts = h.tts.wait()
    assert wait_sent(tts, 2)
    assert tts.sent == ["One sentence. ", FLUSH, "And another one. ", FLUSH]
    assert len(h.tts.made) == 1  # same socket for both
    assert h.p.tts_chars_sent == len("One sentence. And another one. ")
    lines = events_until(h.p, "another")
    assert lines == ['heard: "One sentence." (14 chars spoken)', 'heard: "And another one." (17 chars spoken)']


def test_long_transcript_is_truncated_in_heard_line(make):
    h = make()
    text = "word " * 30
    h.stt.wait().push(committed(text))
    (line,) = events_until(h.p, "heard")
    assert line == 'heard: "' + "word " * 11 + 'word..." (150 chars spoken)'


@pytest.mark.parametrize("text, lines", [
    ("Um.", ['ignored sound: no words ("Um.")']),
    ("(sighs)", ['ignored sound: no words ("(sighs)")']),
    ("", []),
])
def test_filler_only_segment_speaks_nothing(make, text, lines):
    h = make()
    h.stt.wait().push(partial(text), committed(text), SttEvent("warning", "sentinel"))
    assert events_until(h.p, "sentinel") == lines + ["WARNING: sentinel"]
    assert h.tts.made == []  # never even connected
    assert h.p.tts_chars_sent == 0


def test_stt_warning_and_non_fatal_error_events_are_warnings(make):
    h = make()
    h.stt.wait().push(SttEvent("warning", "Low audio."),
                      SttEvent("error", "Realtime speech-to-text commits are throttled.", "commit_throttled"))
    lines = events_until(h.p, "throttled")
    assert lines == ["WARNING: Low audio.", "WARNING: Realtime speech-to-text commits are throttled."]
    assert h.p.error is None and not h.p.finished.is_set()


# --- TTS audio out ---

def test_tts_audio_reaches_output_in_order(make):
    h = make()
    h.stt.wait().push(committed("Hello there."))
    tts = h.tts.wait()
    chunks = [b"\x01\x02", b"\x03", b"", b"\x04\x05\x06"]  # odd lengths pass through; empty ones are skipped
    tts.push(*chunks)
    assert h.sink.wait(3)
    time.sleep(0.02)
    assert h.sink.chunks == [b"\x01\x02", b"\x03", b"\x04\x05\x06"]


def test_reader_survives_an_audio_end_while_socket_stays_open(make):
    h = make()
    h.stt.wait().push(committed("Hello there."))
    tts = h.tts.wait()
    tts.push(b"\x01\x02", END, b"\x03\x04")  # END = isFinal with the socket still open
    assert h.sink.wait(2)
    assert h.sink.chunks == [b"\x01\x02", b"\x03\x04"]
    assert len(h.tts.made) == 1


def test_latency_from_speech_start_to_first_audio(make):
    now = [10.0]
    tracker = LatencyTracker(clock=lambda: now[0])
    h = make(latency=tracker)
    stt = h.stt.wait()
    feed(h.p, silence(300) + speech(200))  # speech confirmed on the 4th speech frame: started 80 ms "ago"
    tts = h.tts.wait()  # the local VAD's speech start warmed up the TTS socket before any text
    assert tts.sent == []

    now[0] = 10.5
    tts.push(b"old")  # audio before this utterance's text was sent (e.g. the previous sentence's tail)
    assert h.sink.wait(1)
    assert tracker.timings == []

    stt.push(committed("Hello there."))
    assert wait_sent(tts)
    now[0] = 11.0
    tts.push(b"new", b"more")
    assert h.sink.wait(3)
    assert wait_for(lambda: len(tracker.timings) == 1)
    time.sleep(0.02)
    (timing,) = tracker.timings
    assert timing.speech_end == pytest.approx(9.92)  # = speech START in this mode
    assert timing.latency_ms == pytest.approx(1080.0)


# --- TTS connection lifecycle ---

def test_tts_connects_lazily_and_reconnects_after_timeout(make):
    h = make()
    stt = h.stt.wait()
    time.sleep(0.02)
    assert h.tts.made == []  # nothing to say yet

    stt.push(committed("First sentence."))
    tts1 = h.tts.wait(1)
    assert wait_sent(tts1)
    tts1.server_close(timed_out=True)  # the 180 s inactivity timeout
    assert wait_for(lambda: tts1.server_closed and h.p._tts is None)

    stt.push(committed("Second sentence."))
    tts2 = h.tts.wait(2)
    assert wait_sent(tts2)
    assert tts1.sent == ["First sentence. ", FLUSH]
    assert tts2.sent == ["Second sentence. ", FLUSH]
    assert [line for line in h.p.drain_events() if "WARNING" in line] == []  # a timeout is normal


def test_text_is_resent_when_socket_closes_just_before_send(make):
    h = make()
    stt = h.stt.wait()
    stt.push(committed("First sentence."))
    tts1 = h.tts.wait(1)
    assert wait_sent(tts1)
    tts1.fail_sends = True  # still looks open, but the next send finds it closed

    stt.push(committed("Not lost."))
    tts2 = h.tts.wait(2)
    assert wait_sent(tts2)
    assert tts1.sent == ["First sentence. ", FLUSH]
    assert tts2.sent == ["Not lost. ", FLUSH]
    assert tts1.closed.is_set()
    assert h.p.tts_chars_sent == len("First sentence. Not lost. ")
    assert [line for line in h.p.drain_events() if "WARNING" in line] == []


def test_tts_connect_failure_skips_sentence_then_next_reconnects(make):
    h = make(tts=Factory(FakeTts, RealtimeError("Network problem connecting to Text-to-speech stream.",
                                                fatal=False, error_type="network")))
    stt = h.stt.wait()
    stt.push(partial("Lost words here now"), partial("Lost words here now ok"), committed("Lost words here now ok."))
    lines = events_until(h.p, "WARNING")  # the "heard" line may come before or after it
    assert ("WARNING: Network problem connecting to Text-to-speech stream. "
            "Skipping the rest of this sentence.") in lines

    stt.push(committed("Back again."))
    tts = h.tts.wait(2)
    assert wait_sent(tts)
    assert tts.sent == ["Back again. ", FLUSH]
    assert sum("WARNING" in line for line in lines + h.p.drain_events()) == 1
    assert h.p.error is None


def test_tts_non_fatal_audio_error_warns_and_next_text_reconnects(make):
    h = make()
    stt = h.stt.wait()
    stt.push(committed("First."))
    tts1 = h.tts.wait(1)
    assert wait_sent(tts1)
    tts1.push(RealtimeError("Text-to-speech stream error: boom.", fatal=False))
    assert events_until(h.p, "WARNING")[-1] == ("WARNING: Text-to-speech stream error: boom. "
                                               "The next words will reconnect.")
    assert wait_for(lambda: tts1.closed.is_set())

    stt.push(committed("Second."))
    tts2 = h.tts.wait(2)
    assert wait_sent(tts2)
    assert tts2.sent == ["Second. ", FLUSH]
    assert h.p.error is None and not h.p.finished.is_set()


# --- fatal errors ---

def assert_fatal(h: Harness, message: str) -> None:
    assert h.p.finished.wait(WAIT_S)
    assert isinstance(h.p.error, ApiFatalError)
    assert str(h.p.error) == message
    lines = events_until(h.p, "ERROR")
    assert lines.count(f"ERROR: {message}") == 1
    assert sum(line.startswith("ERROR") for line in lines + h.p.drain_events()) == 1
    assert wait_for(lambda: not h.p._thread.is_alive())  # tasks shut down, loop thread ended


def test_fatal_stt_error_in_events(make):
    h = make()
    stt = h.stt.wait()
    stt.push(RealtimeError("Out of ElevenLabs credits. Check your usage at elevenlabs.io.", fatal=True,
                           error_type="quota_exceeded"))
    assert_fatal(h, "Out of ElevenLabs credits. Check your usage at elevenlabs.io.")
    assert len(h.stt.made) == 1  # no reconnect after a fatal error
    feed(h.p, speech(200))  # ignored once finished
    assert stt.sent == []


def test_fatal_stt_error_on_connect(make):
    h = make(stt=Factory(FakeStt, RealtimeError("Realtime speech-to-text rejected the API key (HTTP 401).",
                                                fatal=True, error_type="http_401")))
    assert_fatal(h, "Realtime speech-to-text rejected the API key (HTTP 401).")
    assert len(h.stt.made) == 1


def test_fatal_tts_error_on_connect(make):
    h = make(tts=Factory(FakeTts, RealtimeError("Voice not found. Check ELEVENLABS_VOICE_ID in .env.",
                                                fatal=True, error_type="http_404")))
    stt = h.stt.wait()
    stt.push(committed("Hello there."))
    assert_fatal(h, "Voice not found. Check ELEVENLABS_VOICE_ID in .env.")
    assert stt.closed.is_set()


def test_fatal_tts_error_in_audio(make):
    h = make()
    stt = h.stt.wait()
    stt.push(committed("Hello there."))
    tts = h.tts.wait()
    tts.push(RealtimeError("Out of ElevenLabs credits.", fatal=True))
    assert_fatal(h, "Out of ElevenLabs credits.")
    assert stt.closed.is_set()


def test_unexpected_exception_in_a_task_stops_pipeline(make):
    def broken_output(pcm: bytes) -> None:
        raise ValueError("device gone")

    h = make(sink=broken_output)
    h.stt.wait().push(committed("Hello there."))
    h.tts.wait().push(b"\x01\x02")
    assert_fatal(h, "Unexpected error in text-to-speech audio (ValueError: device gone).")


# --- STT reconnect ---

def test_non_fatal_stt_disconnect_reconnects_and_resets_segment(make):
    h = make()
    stt1 = h.stt.wait(1)
    stt1.push(partial("One two three"), partial("One two three four"))  # stabilizer speaks "One two "
    tts = h.tts.wait()
    assert wait_for(lambda: tts.sent == ["One two "])
    stt1.push(RealtimeError("Realtime speech-to-text connection closed (code 1011).", fatal=False,
                            error_type="closed"))
    stt2 = h.stt.wait(2)
    lines = events_until(h.p, "reconnected")
    assert lines == ["WARNING: Realtime speech-to-text connection closed (code 1011). Reconnecting in 0.01 s.",
                     "speech-to-text reconnected"]
    assert wait_sent(tts)  # what was already sent gets spoken

    stt2.push(committed("Fresh start."))  # not diffed against the lost segment's "One two"
    assert wait_sent(tts, 2)
    assert tts.sent == ["One two ", FLUSH, "Fresh start. ", FLUSH]
    assert stt1.closed.is_set()
    feed(h.p, silence(100))
    assert wait_for(lambda: stt2.sent == [silence(100)])
    assert h.p.error is None


def test_stt_connect_failures_retry_with_backoff(make, monkeypatch):
    monkeypatch.setattr(stt_tts, "STT_RECONNECT_DELAYS_S", (0.01, 0.02))
    err = RealtimeError("Realtime speech-to-text: timed out connecting to ElevenLabs.", fatal=False,
                        error_type="timeout")
    h = make(stt=Factory(FakeStt, err, err, err))
    h.stt.wait(4)
    lines = events_until(h.p, "reconnected")
    assert lines == [f"WARNING: {err.message} Reconnecting in 0.01 s.",
                     f"WARNING: {err.message} Reconnecting in 0.02 s.",
                     f"WARNING: {err.message} Reconnecting in 0.02 s.",
                     "speech-to-text reconnected"]


# --- credit cap ---

def test_cap_counts_only_speech(make):
    h = make()
    h.stt.wait()
    feed(h.p, silence(1000))
    assert h.p.speech_sent_s == 0.0
    # Start run (4 frames) + 46 more speech frames + 25 silent frames until the phrase ends = 75 frames.
    feed(h.p, silence(300) + speech(1000) + silence(600))
    assert h.p.speech_sent_s == pytest.approx(1.5)
    feed(h.p, silence(1000))
    assert h.p.speech_sent_s == pytest.approx(1.5)


def test_cap_stops_mic_audio_flushes_last_segment_then_finishes(make, monkeypatch):
    monkeypatch.setattr(stt_tts, "CAP_AUDIO_IDLE_S", 0.05)
    h = make(make_config(max_speech_s=1.0))
    stt = h.stt.wait()
    stt.push(partial("Hello there my"), partial("Hello there my friend"))  # segment in progress
    tts = h.tts.wait()
    assert wait_for(lambda: tts.sent == ["Hello there "])

    voiced = speech(2000)
    feed(h.p, silence(300) + voiced)
    assert h.p.speech_sent_s == pytest.approx(1.0)
    assert events_until(h.p, "cap") == ["Speech cap reached (1 s). Use --max-speech to raise it."]

    # Mic audio up to the cap, then silence so the server's VAD commits the last segment.
    total = 300 + 1000 + CAP_SILENCE_MS
    assert wait_for(lambda: sum(map(len, stt.sent)) == total * BYTES_PER_MS)
    assert b"".join(stt.sent) == silence(300) + voiced[:1000 * BYTES_PER_MS] + silence(CAP_SILENCE_MS)
    feed(h.p, speech(500, seed=1))  # no longer forwarded
    time.sleep(0.03)
    assert sum(map(len, stt.sent)) == total * BYTES_PER_MS
    assert not h.p.finished.is_set()  # still waiting for the commit

    stt.push(committed("Hello there, my friend."))
    assert wait_sent(tts)
    tts.push(b"\x01\x02")
    assert h.sink.wait(1)
    assert h.p.finished.wait(WAIT_S)
    assert tts.sent == ["Hello there ", "my friend. ", FLUSH]
    assert h.p.error is None
    assert h.p.speech_sent_s == pytest.approx(1.0)


def test_cap_without_open_segment_finishes_promptly(make):
    h = make(make_config(max_speech_s=0.5))
    stt = h.stt.wait()
    feed(h.p, silence(300) + speech(1000))
    assert h.p.finished.wait(WAIT_S)
    assert events_until(h.p, "cap") == ["Speech cap reached (0.5 s). Use --max-speech to raise it."]
    assert h.p.tts_chars_sent == 0
    # Mic audio up to the cap only: no silence padding, as there was no segment to commit.
    assert wait_for(lambda: sum(map(len, stt.sent)) == (300 + 500) * BYTES_PER_MS)
    time.sleep(0.03)
    assert sum(map(len, stt.sent)) == (300 + 500) * BYTES_PER_MS


def test_zero_cap_means_unlimited(make):
    h = make(make_config(max_speech_s=0))
    h.stt.wait()
    feed(h.p, silence(300) + speech(3000))
    assert h.p.speech_sent_s == pytest.approx(3.0)
    assert not h.p.finished.is_set()
    assert h.p.drain_events() == []


def test_cap_before_start_finishes_immediately(make):
    h = make(make_config(max_speech_s=0.5), start=False)
    feed(h.p, silence(300) + speech(1000))
    assert h.p.finished.is_set()
    assert h.p.drain_events() == ["Speech cap reached (0.5 s). Use --max-speech to raise it."]


# --- start / stop ---

def test_stop_before_start_is_fine(make):
    h = make(start=False)
    h.p.stop()
    h.p.stop()
    h.p.start()  # not restartable: start after stop does nothing
    assert h.p._thread is None
    feed(h.p, speech(200))


def test_stop_is_prompt_idempotent_and_closes_streams(make):
    h = make()
    stt = h.stt.wait()
    stt.push(committed("Hello there."))
    tts = h.tts.wait()
    assert wait_sent(tts)
    t0 = time.monotonic()
    h.p.stop()
    assert time.monotonic() - t0 < 0.5
    assert not h.p._thread.is_alive()
    assert stt.closed.is_set() and tts.closed.is_set()
    h.p.stop()
    feed(h.p, speech(200))  # ignored after stop, no error
    assert h.p.error is None


def test_start_is_idempotent(make):
    h = make()
    thread = h.p._thread
    h.p.start()
    assert h.p._thread is thread
    h.stt.wait()
    time.sleep(0.02)
    assert len(h.stt.made) == 1


# --- construction ---

@pytest.mark.parametrize("key, voice", [(None, FAKE_VOICE), (FAKE_KEY, None)])
def test_missing_credentials_raise_config_error(key, voice):
    with pytest.raises(ConfigError, match="Missing ELEVENLABS_") as info:
        SttTtsPipeline(make_config(api_key=key, voice_id=voice), lambda pcm: None)
    assert FAKE_KEY not in str(info.value)


def test_default_factories_build_streams_from_config():
    p = SttTtsPipeline(make_config(output_format="pcm_24000", vad_aggressiveness=1), lambda pcm: None)
    assert (p.input_sample_rate, p.output_sample_rate) == (16_000, 24_000)
    assert p._segmenter.vad.aggressiveness == 1

    stt = p._stt_factory()
    assert isinstance(stt, SttStream) and not stt.is_open
    assert (stt.model_id, stt.language_code, stt.vad_silence_s, stt.no_verbatim, stt.sample_rate) == (
        STT_MODEL_ID, "en", 0.5, True, 16_000)
    assert p._stt_factory() is not stt  # one stream per connection

    tts = p._tts_factory()
    assert isinstance(tts, TtsStream) and not tts.is_open
    assert (tts.voice_id, tts.model_id, tts.output_format) == (FAKE_VOICE, TTS_MODEL_ID, "pcm_24000")
    assert tts.chunk_length_schedule == list(TTS_CHUNK_LENGTH_SCHEDULE)
    assert tts.inactivity_timeout == TTS_INACTIVITY_TIMEOUT_S
    assert tts.voice_settings == TTS_VOICE_SETTINGS and tts.voice_settings["similarity_boost"] == 0.9
    assert p._stabilizer.holdback_words == PARTIAL_HOLDBACK_WORDS
    assert p._tts_client is None  # the websocket path never builds the HTTP client
    assert FAKE_KEY not in repr(stt) + repr(tts)
    p.stop()


def test_fast_timing_uses_its_holdback_and_schedule():
    p = SttTtsPipeline(make_config(timing="fast"), lambda pcm: None)
    assert STREAMING_TIMINGS["fast"] == (1, (50, 70, 100, 140))
    assert p._stabilizer.holdback_words == 1
    tts = p._tts_factory()
    assert tts.chunk_length_schedule == [50, 70, 100, 140]
    assert tts.model_id == TTS_MODEL_ID
    assert tts.voice_settings == TTS_VOICE_SETTINGS
    p.stop()


def test_smooth_default_client_is_built_from_the_key(monkeypatch):
    made = []
    monkeypatch.setattr(stt_tts, "make_client", lambda api_key: made.append(api_key) or "client")
    p = SttTtsPipeline(make_config(timing="smooth"), lambda pcm: None)
    assert made == [FAKE_KEY] and p._tts_client == "client"
    p.stop()


def test_smooth_missing_credentials_raise_config_error_without_a_fake_client():
    with pytest.raises(ConfigError, match="Missing ELEVENLABS_VOICE_ID"):
        SttTtsPipeline(make_config(timing="smooth", voice_id=None), lambda pcm: None,
                       stt_factory=Factory(FakeStt))


# --- smooth timing (the default): one HTTP stream per committed phrase, no websocket TTS ---

CAP_SILENCE_MS = 500 + stt_tts.CAP_SILENCE_EXTRA_MS  # default pause + extra


def smooth_config(**overrides) -> Config:
    return make_config(timing="smooth", **overrides)


def api_error(status: int, message: str = "") -> ApiError:
    body = {"detail": {"message": message}} if message else None
    return ApiError(status_code=status, headers={"xi-api-key": FAKE_KEY}, body=body)


def test_smooth_is_the_default_timing():
    config = Config(api_key=FAKE_KEY, voice_id=FAKE_VOICE, mode="stt-tts", input_device=None,
                    output_device="BlackHole 2ch", monitor_device=None, vad_aggressiveness=2,
                    dry_run=False, list_devices=False)
    assert (config.timing, config.tts_model) == ("smooth", SMOOTH_TTS_MODEL)


def test_smooth_partials_never_speak(make):
    h = make(smooth_config())
    h.stt.wait().push(partial("Hello"), partial("Hello there"), partial("Hello there how are"),
                      partial("Hello there how are you"), SttEvent("warning", "sentinel"))
    assert events_until(h.p, "sentinel") == ["WARNING: sentinel"]
    time.sleep(0.02)
    assert h.client.calls == []
    assert h.p._segment_open  # still tracked for the credit cap
    assert h.p.tts_chars_sent == 0


def test_smooth_commit_makes_one_stream_call_with_model_format_and_voice_settings(make):
    h = make(smooth_config(output_format="pcm_24000"))
    h.stt.wait().push(partial("Um hello there"), partial("Um hello there how are you"),
                      committed("Um, hello there, (sighs) how are you?"))
    assert h.sink.wait(1)
    (call,) = h.client.calls
    assert call["voice_id"] == FAKE_VOICE
    assert call["text"] == "hello there, how are you?"  # whole phrase, fillers/events stripped, no trailing space
    assert call["model_id"] == SMOOTH_TTS_MODEL == "eleven_v4_turbo"
    assert call["output_format"] == "pcm_24000"
    settings = call["voice_settings"]
    assert isinstance(settings, VoiceSettings)
    assert settings.similarity_boost == 0.9
    assert settings.model_dump(exclude_none=True) == TTS_VOICE_SETTINGS
    assert h.p.tts_chars_sent == len("hello there, how are you?")
    assert events_until(h.p, "heard") == ['heard: "Um, hello there, (sighs) how are you?" (25 chars spoken)']


def test_smooth_audio_chunks_reach_output_in_order(make):
    h = make(smooth_config(), client=FakeTtsClient([b"\x01\x02", b"\x03", b"", b"\x04\x05\x06"]))
    h.stt.wait().push(committed("Hello there."))
    assert h.sink.wait(3)
    assert wait_for(lambda: h.client.closed == [0])
    assert h.sink.chunks == [b"\x01\x02", b"\x03", b"\x04\x05\x06"]  # odd lengths pass through, empty skipped


def test_smooth_phrases_are_spoken_one_at_a_time_in_order(make):
    first = [b"a1", b"a2", b"a3", b"a4"]
    client = FakeTtsClient(first, [b"b1", b"b2"], delay=0.02)
    h = make(smooth_config(), client=client)
    stt = h.stt.wait()
    stt.push(committed("First phrase."))
    assert h.sink.wait(1)
    stt.push(committed("Second phrase."))  # arrives while the first is still streaming
    assert h.sink.wait(6)
    assert h.sink.chunks == [b"a1", b"a2", b"a3", b"a4", b"b1", b"b2"]
    assert client.texts() == ["First phrase.", "Second phrase."]
    assert client.max_active == 1  # never overlapping
    assert events_until(h.p, "Second") == ['heard: "First phrase." (13 chars spoken)',
                                           'heard: "Second phrase." (14 chars spoken)']


@pytest.mark.parametrize("text, lines", [
    ("Um.", ['ignored sound: no words ("Um.")']),
    ("(sighs)", ['ignored sound: no words ("(sighs)")']),
    ("", []),
])
def test_smooth_filler_only_commit_makes_no_call(make, text, lines):
    h = make(smooth_config())
    h.stt.wait().push(partial(text), committed(text), SttEvent("warning", "sentinel"))
    assert events_until(h.p, "sentinel") == lines + ["WARNING: sentinel"]
    time.sleep(0.02)
    assert h.client.calls == []
    assert h.p._phrases_pending == 0


def test_smooth_latency_from_commit_to_first_audio(make):
    now = [9.0]
    tracker = LatencyTracker(clock=lambda: now[0])
    gate = threading.Event()
    h = make(smooth_config(), latency=tracker, client=FakeTtsClient([gate, b"first", b"second"]))
    stt = h.stt.wait()
    feed(h.p, silence(300) + speech(200))  # speech start is NOT timed in smooth mode
    time.sleep(0.02)
    assert tracker._pending is None

    now[0] = 10.0
    stt.push(committed("Hello there."))
    assert wait_for(lambda: len(h.client.calls) == 1)
    now[0] = 10.3
    gate.set()
    assert h.sink.wait(2)
    assert wait_for(lambda: len(tracker.timings) == 1)
    time.sleep(0.02)
    (timing,) = tracker.timings
    assert timing.speech_end == pytest.approx(9.5)  # the commit (t=10.0) minus the 0.5 s pause
    assert timing.latency_ms == pytest.approx(800.0)
    assert [(name, round(ms)) for name, ms in timing.stages()] == [("pause", 500), ("queue", 0), ("voice", 300)]


def test_smooth_queued_phrase_keeps_its_own_speech_end(make):
    now = [10.0]
    tracker = LatencyTracker(clock=lambda: now[0])
    gate1, gate2 = threading.Event(), threading.Event()
    h = make(smooth_config(), latency=tracker, client=FakeTtsClient([gate1, b"a"], [gate2, b"b"]))
    stt = h.stt.wait()
    stt.push(committed("First."))
    assert wait_for(lambda: len(h.client.calls) == 1)
    now[0] = 11.0
    stt.push(committed("Second."))  # queued behind the first, which hasn't produced audio yet
    assert wait_for(lambda: h.p._phrases_pending == 2)
    now[0] = 11.5
    gate1.set()
    assert wait_for(lambda: len(tracker.timings) == 1)
    now[0] = 12.0
    gate2.set()
    assert wait_for(lambda: len(tracker.timings) == 2)
    assert [(t.speech_end, round(t.latency_ms)) for t in tracker.timings] == [(9.5, 2000), (10.5, 1500)]
    assert [round(ms) for _, ms in tracker.timings[1].stages()] == [500, 500, 500]  # pause, queue, voice


@pytest.mark.parametrize("error, message", [
    (api_error(401, "Invalid API key"),
     "ElevenLabs rejected the API key: Invalid API key. Check ELEVENLABS_API_KEY in .env and that the key has "
     "the Text to Speech permission enabled."),
    (api_error(400, "Model eleven_v4 is not available for this voice"),
     "ElevenLabs rejected the request (400): Model eleven_v4 is not available for this voice."),
])
def test_smooth_fatal_api_error_stops_pipeline(make, error, message):
    h = make(smooth_config(), client=FakeTtsClient(error))
    stt = h.stt.wait()
    stt.push(committed("Hello there."))
    assert_fatal(h, message)
    assert FAKE_KEY not in message
    assert stt.closed.is_set()
    assert wait_for(lambda: not h.p._speaker.is_alive())
    assert h.sink.chunks == []


def test_smooth_unexpected_output_error_stops_pipeline(make):
    def broken_output(pcm: bytes) -> None:
        raise ValueError("device gone")

    h = make(smooth_config(), sink=broken_output)
    h.stt.wait().push(committed("Hello there."))
    assert_fatal(h, "Unexpected error in text-to-speech audio (ValueError: device gone).")


@pytest.fixture
def fast_retry(monkeypatch):
    monkeypatch.setattr(stt_tts, "RETRY_DELAY_S", 0.01)


def test_smooth_temporary_error_is_retried_once(make, fast_retry):
    h = make(smooth_config(), client=FakeTtsClient(api_error(503), [b"\x07\x08"]))
    h.stt.wait().push(committed("Retry me."))
    assert h.sink.wait(1)
    assert h.sink.chunks == [b"\x07\x08"]
    assert "WARNING: ElevenLabs server error (503). Try again shortly. Retrying the phrase." in h.p.drain_events()
    assert h.client.texts() == ["Retry me.", "Retry me."]
    assert h.p.error is None


def test_smooth_second_failure_skips_phrase_and_next_one_plays(make, fast_retry):
    h = make(smooth_config(), client=FakeTtsClient(api_error(503), api_error(503), [b"\x07\x08"]))
    stt = h.stt.wait()
    stt.push(committed("Lost phrase."))
    lines = events_until(h.p, "skipped")
    assert ('WARNING: ElevenLabs server error (503). Try again shortly. - skipped "Lost phrase."') in lines
    stt.push(committed("Back again."))
    assert h.sink.wait(1)
    assert h.sink.chunks == [b"\x07\x08"]
    assert h.client.texts() == ["Lost phrase.", "Lost phrase.", "Back again."]
    assert h.p.tts_chars_sent == 2 * len("Lost phrase.") + len("Back again.")  # failed requests were sent too
    assert h.p.error is None and not h.p.finished.is_set()


def test_smooth_no_retry_once_audio_has_played(make, fast_retry):
    class HalfwayClient(FakeTtsClient):
        def stream(self, voice_id: str, **kwargs):
            self.calls.append({"voice_id": voice_id, **kwargs})

            def gen():
                yield b"\x01\x02"
                raise httpx.ReadError("dropped")
            return gen()

    h = make(smooth_config(), client=HalfwayClient())
    h.stt.wait().push(committed("Half."))
    lines = events_until(h.p, "skipped")
    assert any(line.startswith("WARNING: Network problem") and 'skipped "Half."' in line for line in lines)
    assert len(h.client.calls) == 1 and h.sink.chunks == [b"\x01\x02"]


def test_smooth_set_voice_applies_to_the_next_phrase(make):
    h = make(smooth_config())
    stt = h.stt.wait()
    stt.push(committed("First."))
    assert h.sink.wait(1)
    h.p.set_voice("other-voice")
    stt.push(committed("Second."))
    assert h.sink.wait(2)
    assert [c["voice_id"] for c in h.client.calls] == [FAKE_VOICE, "other-voice"]


def voiced_tts_factory(holder: list) -> Factory:
    """FakeTts that records the voice the pipeline would connect with (like the real TtsStream URL)."""

    def make_tts(error):
        tts = FakeTts(error)
        tts.voice_id = holder[0].config.voice_id
        return tts

    return Factory(make_tts)


def test_websocket_set_voice_reconnects_at_next_sentence(make, monkeypatch):
    monkeypatch.setattr(stt_tts, "VOICE_SWITCH_IDLE_S", 0.0)
    holder: list = []
    tts = voiced_tts_factory(holder)
    h = make(make_config(), tts=tts, start=False)
    holder.append(h.p)
    h.p.start()
    stt = h.stt.wait()
    stt.push(committed("Hello there."))
    first = tts.wait(1)
    assert wait_sent(first)
    h.p.set_voice("other-voice")
    stt.push(committed("Second one."))
    second = tts.wait(2)
    assert wait_sent(second)
    assert (first.voice_id, second.voice_id) == (FAKE_VOICE, "other-voice")
    assert first.closed.is_set()
    assert second.sent == ["Second one. ", FLUSH]


def test_websocket_voice_switch_waits_for_the_sentence_to_finish(make, monkeypatch):
    monkeypatch.setattr(stt_tts, "VOICE_SWITCH_IDLE_S", 0.0)
    holder: list = []
    tts = voiced_tts_factory(holder)
    h = make(make_config(), tts=tts, start=False)
    holder.append(h.p)
    h.p.start()
    stt = h.stt.wait()
    stt.push(partial("one two three four"), partial("one two three four five"))  # settled words go early
    first = tts.wait(1)
    assert wait_for(lambda: first.sent == ["one two three "])
    h.p.set_voice("other-voice")  # mid-sentence: text is waiting for a flush
    stt.push(committed("one two three four five six seven."))
    assert wait_sent(first)
    assert len(tts.made) == 1  # the sentence finished on the old voice


def test_smooth_stop_mid_phrase_is_prompt_and_delivers_nothing_more(make):
    client = FakeTtsClient([bytes([i]) * 2 for i in range(1, 100)], delay=0.03)
    h = make(smooth_config(), client=client)
    h.stt.wait().push(committed("A long sentence that takes a while."))
    assert h.sink.wait(2)
    t0 = time.monotonic()
    h.p.stop()
    assert time.monotonic() - t0 < 0.5
    assert not h.p._speaker.is_alive() and not h.p._thread.is_alive()
    delivered = len(h.sink.chunks)
    time.sleep(0.1)
    assert len(h.sink.chunks) == delivered < 99
    assert client.closed == [0]  # the generator (HTTP response) was closed
    h.p.stop()  # idempotent
    assert h.p.error is None


def test_smooth_cap_drain_waits_for_the_current_phrase(make):
    gate = threading.Event()
    h = make(smooth_config(max_speech_s=1.0), client=FakeTtsClient([b"\x01\x02", gate, b"\x03\x04"]))
    stt = h.stt.wait()
    stt.push(partial("Hello there my"), partial("Hello there my friend"))
    assert wait_for(lambda: h.p._segment_open)

    voiced = speech(2000)
    feed(h.p, silence(300) + voiced)
    assert events_until(h.p, "cap") == ["Speech cap reached (1 s). Use --max-speech to raise it."]
    total = 300 + 1000 + CAP_SILENCE_MS  # mic audio to the cap, then silence so the server commits
    assert wait_for(lambda: sum(map(len, stt.sent)) == total * BYTES_PER_MS)
    stt.push(committed("Hello there, my friend."))
    assert h.sink.wait(1)  # the phrase is streaming...
    time.sleep(0.05)
    assert not h.p.finished.is_set()  # ...and the drain waits for it
    gate.set()
    assert h.p.finished.wait(WAIT_S)
    assert h.sink.chunks == [b"\x01\x02", b"\x03\x04"]
    assert h.client.texts() == ["Hello there, my friend."]
    assert h.p.error is None


def test_smooth_cap_drain_has_a_hard_timeout(make, monkeypatch):
    monkeypatch.setattr(stt_tts, "CAP_DRAIN_MAX_S", 0.1)
    gate = threading.Event()
    h = make(smooth_config(max_speech_s=0.5), client=FakeTtsClient([gate, b"late"]))
    stt = h.stt.wait()
    stt.push(committed("Stuck phrase."))
    assert wait_for(lambda: len(h.client.calls) == 1)
    feed(h.p, silence(300) + speech(1000))
    assert h.p.finished.wait(WAIT_S)  # no open segment; the stuck phrase only delays it by CAP_DRAIN_MAX_S
    gate.set()


def test_smooth_never_opens_a_tts_websocket(make):
    h = make(smooth_config())
    stt = h.stt.wait()
    feed(h.p, silence(300) + speech(400) + silence(600))  # local VAD speech start would warm the socket
    stt.push(partial("Hello there"), committed("Hello there."), committed("Second one."))
    assert h.sink.wait(2)
    time.sleep(0.02)
    assert h.tts.made == []
    assert h.p._tts_queue.empty()


def test_smooth_tts_model_override_passes_through(make):
    h = make(smooth_config(tts_model_override="eleven_v4"))
    h.stt.wait().push(committed("Hello there."))
    assert wait_for(lambda: len(h.client.calls) == 1)
    assert h.client.calls[0]["model_id"] == "eleven_v4"


def test_smooth_stt_drop_mid_segment_reconnects_and_speaks_nothing_stale(make):
    h = make(smooth_config())
    stt1 = h.stt.wait(1)
    stt1.push(partial("One two three"), RealtimeError("Realtime speech-to-text connection closed (code 1011).",
                                                      fatal=False, error_type="closed"))
    stt2 = h.stt.wait(2)
    assert not h.p._segment_open
    stt2.push(committed("Fresh start."))
    assert h.sink.wait(1)
    assert h.client.texts() == ["Fresh start."]


# --- delivery tag, calm punctuation, speed ---

@pytest.mark.parametrize("text, expected", [
    ("Wow!", "Wow."),
    ("Really?!", "Really?"),
    ("No way!!! Seriously!", "No way... Seriously."),
    ("Plain sentence.", "Plain sentence."),
])
def test_calm_punctuation(text, expected):
    assert stt_tts.calm_punctuation(text) == expected


@pytest.mark.parametrize("model, delivery, expected", [
    ("eleven_v4_turbo", "calm, relaxed", "[calm, relaxed] Hi."),
    ("eleven_v4", "calm", "[calm] Hi."),
    ("eleven_v3", "calm", "[calm] Hi."),
    ("eleven_v4_turbo", "", "Hi."),  # --delivery "" turns it off
])
def test_delivery_tag_for_v3_v4_models(model, delivery, expected):
    assert stt_tts.with_delivery_tag("Hi.", smooth_config(tts_model_override=model, delivery=delivery)) == expected


def test_no_delivery_tag_for_flash_which_would_read_it_aloud():
    config = smooth_config(tts_model_override="eleven_flash_v2_5", delivery="calm")
    assert stt_tts.with_delivery_tag("Hi.", config) == "Hi."


def test_smooth_sends_tagged_calm_text_and_counts_its_chars(make):
    h = make(smooth_config(delivery="calm, relaxed, conversational", speed=0.95))
    h.stt.wait().push(committed("That is amazing!"))
    assert h.sink.wait(1)
    (call,) = h.client.calls
    assert call["text"] == "[calm, relaxed, conversational] That is amazing."
    assert call["voice_settings"].speed == 0.95
    assert h.p.tts_chars_sent == len(call["text"])  # the tag is billed too


def test_websocket_text_gets_calm_punctuation_but_no_tag(make):
    h = make(make_config(delivery="calm"))  # balanced: Flash over the websocket
    h.stt.wait().push(committed("Hello there, that is great!"))
    tts = h.tts.wait()
    assert wait_sent(tts)
    time.sleep(0.02)
    assert tts.sent == ["Hello there, that is great. ", FLUSH]


def test_smooth_uses_stability_and_similarity_overrides(make):
    h = make(smooth_config(stability=0.35, similarity=0.8))
    h.stt.wait().push(committed("Hello there."))
    assert h.sink.wait(1)
    (call,) = h.client.calls
    assert (call["voice_settings"].stability, call["voice_settings"].similarity_boost) == (0.35, 0.8)
