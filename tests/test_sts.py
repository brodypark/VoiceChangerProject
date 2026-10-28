"""pipelines/sts.py tests. A fake ElevenLabs client stands in for the network; no audio hardware.

The fake client's speech_to_text answers "Hello there." by default, so the word check
(on by default) lets every phrase through unless a test scripts something else.

Synthetic speech is seeded white noise (std 0.3 full scale) wrapped in silence and fed in
20 ms frames. The RMS detector (use_webrtc=False) keeps phrase lengths exact:
emitted phrase = 200 ms pre-roll + speech + 200 ms tail.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from elevenlabs.core.api_error import ApiError

import pipelines.sts as sts
from config import FRAME_BYTES, SAMPLE_RATE, STS_LATENCY_OPTIMIZATION, STS_MODEL_ID, Config, ConfigError
from latency import LatencyTracker
from pipelines.base import ApiFatalError
from pipelines.sts import INPUT_FILE_FORMAT, MAX_QUEUED_PHRASES, StsPipeline, phrase_seconds
from vad import PhraseSegmenter, VoiceActivityDetector, iter_frames

BYTES_PER_MS = SAMPLE_RATE // 1000 * 2
PAD_MS = 200  # segmenter default pre-roll == tail
FAKE_KEY = "FAKE-KEY-not-a-real-key"
FAKE_VOICE = "fake-voice-id"
WAIT_S = 1.0


def silence(ms: int) -> bytes:
    return bytes(ms * BYTES_PER_MS)


def speech(ms: int, seed: int = 0) -> bytes:
    noise = np.random.default_rng(seed).normal(0, 0.3, ms * SAMPLE_RATE // 1000)
    return (np.clip(noise, -1, 1) * 32767).astype("<i2").tobytes()


def expected_phrase(voiced: bytes) -> bytes:
    return silence(PAD_MS) + voiced + silence(PAD_MS)


def utterance(ms: int = 1000, seed: int = 0) -> tuple[bytes, bytes]:
    """(audio to feed, phrase the segmenter will emit): speech wrapped in enough silence to end it."""
    voiced = speech(ms, seed)
    return silence(300) + voiced + silence(600), expected_phrase(voiced)


def feed(pipeline: StsPipeline, pcm: bytes) -> None:
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
    values = dict(api_key=FAKE_KEY, voice_id=FAKE_VOICE, mode="sts", input_device=None,
                  output_device="BlackHole 2ch", monitor_device=None, vad_aggressiveness=2,
                  dry_run=False, list_devices=False)
    return Config(**{**values, **overrides})


CHUNKS = [b"\x01\x02" * 50, b"", b"\x03" * 101, b"", b"\x04\x05" * 7]  # odd-length and empty chunks


class FakeSts:
    """client.speech_to_speech: stream() records its arguments and returns a lazy generator, like the SDK.

    `script` is consumed one entry per call: a list of chunks, or an exception to raise
    on the first next() (when the real request would go out).
    """

    def __init__(self, script=None, delay_s: float = 0.0) -> None:
        self.script = list(script) if script is not None else []
        self.delay_s = delay_s
        self.calls: list[dict] = []
        self.closed = threading.Event()

    def stream(self, voice_id: str, **kwargs):
        self.calls.append({"voice_id": voice_id, **kwargs})
        step = self.script.pop(0) if self.script else list(CHUNKS)

        def gen():
            try:
                if isinstance(step, BaseException):
                    raise step
                for chunk in step:
                    if self.delay_s:
                        time.sleep(self.delay_s)
                    yield chunk
            finally:
                self.closed.set()

        return gen()


def stt_result(*words: tuple[str, str]) -> SimpleNamespace:
    """A Scribe response: words are (text, type) pairs; text is the words joined."""
    return SimpleNamespace(text="".join(t for t, _ in words),
                           words=[SimpleNamespace(text=t, type=k) for t, k in words])


REAL_WORDS = stt_result(("Hello", "word"), (" ", "spacing"), ("there.", "word"))
FILLER = stt_result(("Hmm", "word"))
SIGH = stt_result(("(sighs)", "audio_event"))


class FakeStt:
    """client.speech_to_text: convert() records its arguments.

    `script` is consumed one entry per call: a result, or an exception to raise. Default: REAL_WORDS.
    `delay_s` sleeps before answering (or raising).
    """

    def __init__(self, script=None, delay_s: float = 0.0) -> None:
        self.script = list(script) if script is not None else []
        self.delay_s = delay_s
        self.calls: list[dict] = []

    def convert(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0) if self.script else REAL_WORDS
        if self.delay_s:
            time.sleep(self.delay_s)
        if isinstance(step, BaseException):
            raise step
        return step


def fake_client(script=None, delay_s: float = 0.0, stt_script=None, stt_delay_s: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(speech_to_speech=FakeSts(script, delay_s),
                           speech_to_text=FakeStt(stt_script, stt_delay_s))


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


@pytest.fixture
def make_pipeline():
    """Factory for StsPipelines with the RMS detector; stops every one at teardown."""
    made: list[StsPipeline] = []

    def make(config: Config | None = None, client=None, start: bool = True, **kwargs) -> StsPipeline:
        config = config or make_config()
        kwargs.setdefault("vad", VoiceActivityDetector(use_webrtc=False))
        if client is None and not config.dry_run:
            client = fake_client()
        p = StsPipeline(config, kwargs.pop("sink", Sink()), client=client, **kwargs)
        made.append(p)
        if start:
            p.start()
        return p

    yield make
    for p in made:
        p.stop()


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch):
    """Fail loudly if anything tries to build a real ElevenLabs client."""

    def forbidden(api_key: str):
        raise AssertionError("tests must not create a real ElevenLabs client")

    monkeypatch.setattr(sts, "make_client", forbidden)


def events_until(pipeline: StsPipeline, needle: str, timeout: float = WAIT_S) -> list[str]:
    """Drain events until one contains `needle` (or time out); returns everything drained."""
    seen: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        seen += pipeline.drain_events()
        if any(needle in line for line in seen):
            return seen
        time.sleep(0.001)
    return seen + pipeline.drain_events()


# --- helpers ---

def test_phrase_seconds():
    assert phrase_seconds(b"") == 0.0
    assert phrase_seconds(silence(1500)) == pytest.approx(1.5)


def test_utterance_helper_matches_segmenter():
    """Sanity check on the synthetic input: the RMS segmenter emits exactly pre-roll + speech + tail."""
    seg = PhraseSegmenter(VoiceActivityDetector(use_webrtc=False))
    audio, phrase = utterance()
    frames, _ = iter_frames(audio)
    assert [p for p in map(seg.feed, frames) if p is not None] == [phrase]


# --- request arguments and delivery ---

@pytest.mark.parametrize("denoise", [False, True])
def test_stream_kwargs(make_pipeline, denoise):
    client = fake_client()
    sink = Sink()
    cfg = make_config(output_format="pcm_24000", denoise=denoise)
    p = make_pipeline(cfg, client=client, sink=sink)
    audio, phrase = utterance()
    feed(p, audio)
    assert sink.wait(3)

    (call,) = client.speech_to_speech.calls
    assert call["voice_id"] == FAKE_VOICE
    assert call["model_id"] == STS_MODEL_ID
    assert call["output_format"] == "pcm_24000"
    assert call["optimize_streaming_latency"] == STS_LATENCY_OPTIMIZATION
    assert call["file_format"] == INPUT_FILE_FORMAT == "pcm_s16le_16"
    assert call["remove_background_noise"] is denoise
    name, data, mime = call["audio"]
    assert data == phrase
    assert (name, mime) == ("phrase.pcm", "application/octet-stream")
    assert FAKE_KEY not in repr(call)
    assert p.output_sample_rate == 24_000 and p.input_sample_rate == SAMPLE_RATE


def test_chunks_delivered_in_order_without_empty_ones(make_pipeline):
    sink = Sink()
    p = make_pipeline(sink=sink)
    audio, phrase = utterance()
    feed(p, audio)
    lines = events_until(p, "sent")
    assert sink.chunks == [c for c in CHUNKS if c]  # odd lengths passed through untouched
    seconds = len(phrase) / BYTES_PER_MS / 1000
    assert lines == [f"phrase {seconds:.1f} s sent (total {seconds:.1f}/60 s)"]
    assert p.speech_sent_s == pytest.approx(seconds)
    assert p.speech_total_s == pytest.approx(seconds)


def test_phrases_processed_in_order(make_pipeline):
    client = fake_client(script=[[b"A"], [b"B"], [b"C"]])
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    phrases = []
    for seed in (1, 2, 3):
        audio, phrase = utterance(500, seed)
        phrases.append(phrase)
        feed(p, audio)
    assert sink.wait(3)
    assert sink.chunks == [b"A", b"B", b"C"]
    assert [c["audio"][1] for c in client.speech_to_speech.calls] == phrases


def test_generator_is_closed_after_each_phrase(make_pipeline):
    client = fake_client()
    p = make_pipeline(client=client)
    feed(p, utterance()[0])
    assert client.speech_to_speech.closed.wait(WAIT_S)
    events_until(p, "sent")


def test_latency_recorded_from_real_end_of_speech_with_stages(make_pipeline):
    timings = []
    tracker = LatencyTracker(clock=lambda: 10.0, on_utterance=timings.append)  # frozen clock: only the pause counts
    sink = Sink()
    p = make_pipeline(make_config(pause_s=0.7), sink=sink, latency=tracker,
                      segmenter=PhraseSegmenter(VoiceActivityDetector(use_webrtc=False), end_silence_ms=700))
    feed(p, silence(300) + speech(1000) + silence(1100))
    assert sink.wait(1)
    assert wait_for(lambda: len(timings) == 1)
    assert timings[0].latency_ms == pytest.approx(700.0)  # the speaker stopped pause_s before the phrase ended
    assert [(name, round(ms)) for name, ms in timings[0].stages()] == \
        [("pause", 700), ("queue", 0), ("voice", 0), ("word check", 0)]
    assert tracker.summary().count == 1


def test_pause_sets_segmenter_end_silence(make_pipeline):
    p = make_pipeline(make_config(pause_s=0.8), start=False, vad=VoiceActivityDetector(use_webrtc=False))
    assert p._segmenter.end_silence_frames == 40  # 800 ms of 20 ms frames


def test_latency_marked_on_first_non_empty_chunk(make_pipeline):
    calls = []

    class Spy(LatencyTracker):
        def mark_speech_end(self, at: float | None = None) -> int:
            calls.append("end")
            return super().mark_speech_end(at)

        def mark_first_audio_out(self):
            calls.append("first")
            return super().mark_first_audio_out()

    client = fake_client(script=[[b"", b"", b"\x01\x02", b"\x03\x04"]])
    sink = Sink()
    p = make_pipeline(client=client, sink=sink, latency=Spy())
    feed(p, utterance()[0])
    assert sink.wait(2)
    assert calls == ["end", "first"]  # only once per phrase, and not for empty chunks


def test_latency_correct_when_next_phrase_ends_while_previous_in_flight(make_pipeline):
    """Phrase 2 ending before phrase 1's first chunk must not overwrite phrase 1's speech-end time."""
    now = [0.0]
    gates = [threading.Event(), threading.Event()]

    class GatedSts(FakeSts):
        def stream(self, voice_id: str, **kwargs):
            gate = gates[len(self.calls)]
            self.calls.append(kwargs)

            def gen():
                gate.wait(WAIT_S)
                yield b"\x01\x02"
            return gen()

    sts_client = GatedSts()
    tracker = LatencyTracker(clock=lambda: now[0])
    sink = Sink()
    p = make_pipeline(client=SimpleNamespace(speech_to_speech=sts_client, speech_to_text=FakeStt()),
                      sink=sink, latency=tracker)

    feed(p, utterance(seed=1)[0])  # phrase 1 ends at t=0
    assert wait_for(lambda: len(sts_client.calls) == 1)  # phrase 1 in flight, waiting on its gate
    now[0] = 2.0
    feed(p, utterance(seed=2)[0])  # phrase 2 ends at t=2.0, while phrase 1 is still waiting
    now[0] = 2.5
    gates[0].set()  # phrase 1's first chunk at t=2.5
    assert sink.wait(1)
    now[0] = 3.0
    gates[1].set()  # phrase 2's first chunk at t=3.0
    assert sink.wait(2)
    assert wait_for(lambda: len(tracker.timings) == 2)
    assert [round(t.latency_ms) for t in tracker.timings] == [3000, 1500]  # each includes the 0.5 s pause


def test_cap_stops_pipeline_and_emits_message(make_pipeline):
    client = fake_client()
    p = make_pipeline(make_config(max_speech_s=2.0), client=client)
    for seed in (1, 2):  # 1.4 s each: the second would exceed the 2 s cap
        feed(p, utterance(1000, seed)[0])
    assert p.finished.wait(WAIT_S)
    lines = events_until(p, "cap")
    assert lines == ["phrase 1.4 s sent (total 1.4/2 s)", "Speech cap reached (2 s). Use --max-speech to raise it."]
    assert len(client.speech_to_speech.calls) == 1  # the capped phrase was never sent
    assert p.error is None
    assert p.speech_sent_s == pytest.approx(1.4)


def test_cap_blocks_first_phrase_if_too_long(make_pipeline):
    client = fake_client()
    p = make_pipeline(make_config(max_speech_s=1.0), client=client)
    feed(p, utterance(1000)[0])
    assert p.finished.wait(WAIT_S)
    assert events_until(p, "cap") == ["Speech cap reached (1 s). Use --max-speech to raise it."]
    assert client.speech_to_speech.calls == []
    assert p.speech_sent_s == 0.0


def test_zero_cap_means_unlimited(make_pipeline):
    client = fake_client()
    p = make_pipeline(make_config(max_speech_s=0.0), client=client)
    for seed in range(3):
        feed(p, utterance(1000, seed)[0])
    assert wait_for(lambda: len(client.speech_to_speech.calls) == 3)
    lines = events_until(p, "total 4.2")
    assert lines[-1] == "phrase 1.4 s sent (total 4.2 s)"  # no "/cap" when uncapped
    assert not p.finished.is_set()


# --- word check (speech-to-text in parallel with speech-to-speech) ---

def missing_permission_error(status: int = 401) -> ApiError:
    return ApiError(status_code=status, headers={"xi-api-key": FAKE_KEY},
                    body={"detail": {"status": "missing_permissions",
                                     "message": "The API key you used is missing the permission speech_to_text"}})


def test_word_check_runs_alongside_sts_and_plays_real_words(make_pipeline):
    client = fake_client()
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    audio, phrase = utterance()
    feed(p, audio)
    lines = events_until(p, "sent")
    assert sink.chunks == [c for c in CHUNKS if c]
    assert lines == ["phrase 1.4 s sent (total 1.4/60 s)"]
    (call,) = client.speech_to_text.calls
    assert call["file"] == ("phrase.pcm", phrase, "application/octet-stream")
    assert call["file_format"] == "pcm_s16le_16"
    assert call["model_id"] == "scribe_v2" and call["language_code"] == "en"
    assert call["tag_audio_events"] is True
    assert FAKE_KEY not in repr(call)


def test_word_check_overlaps_sts_request(make_pipeline):
    """The STT check is submitted before the STS stream is read, so the two run at the same time."""
    stt_started = threading.Event()
    seen_by_sts = []

    class WatchingSts(FakeSts):
        def stream(self, voice_id, **kwargs):
            inner = super().stream(voice_id, **kwargs)

            def gen():
                seen_by_sts.append(stt_started.wait(WAIT_S))  # first next(): the STS request goes out
                yield from inner
            return gen()

    class SignallingStt(FakeStt):
        def convert(self, **kwargs):
            stt_started.set()
            return super().convert(**kwargs)

    client = SimpleNamespace(speech_to_speech=WatchingSts(), speech_to_text=SignallingStt())
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance()[0])
    assert sink.wait(3)
    assert seen_by_sts == [True]


@pytest.mark.parametrize("result, shown", [
    (FILLER, '("Hmm")'),
    (stt_result(("Um,", "word"), (" ", "spacing"), ("uh-huh", "word")), '("Um, uh-huh")'),
    (SIGH, '("(sighs)")'),
    (stt_result(), "(silence)"),
])
def test_phrase_without_words_is_dropped(make_pipeline, result, shown):
    client = fake_client(stt_script=[result])
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    audio, phrase = utterance()
    feed(p, audio)
    lines = events_until(p, "ignored")
    assert lines == [f"ignored sound: no words {shown}"]
    assert client.speech_to_speech.closed.wait(WAIT_S)  # STS stream closed early
    time.sleep(0.02)
    assert sink.chunks == []
    assert p.speech_sent_s == pytest.approx(1.4)  # the STS upload already happened
    assert p.drain_events() == []  # no "sent" line for an ignored phrase


def test_long_transcript_is_truncated_in_message(make_pipeline):
    many = " ".join(["hmm"] * 30)
    client = fake_client(stt_script=[stt_result((many, "word"))])
    p = make_pipeline(client=client)
    feed(p, utterance()[0])
    (line,) = events_until(p, "ignored")
    assert line == f'ignored sound: no words ("{many[:sts.MAX_SHOWN_TRANSCRIPT_CHARS]}...")'


def test_dropped_then_real_phrase(make_pipeline):
    client = fake_client(script=[[b"A"], [b"B"]], stt_script=[FILLER, REAL_WORDS])
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance(1000, 1)[0])
    feed(p, utterance(1000, 2)[0])
    lines = events_until(p, "sent")
    assert lines == ['ignored sound: no words ("Hmm")', "phrase 1.4 s sent (total 2.8/60 s)"]
    assert sink.chunks == [b"B"]


def test_slow_word_check_is_awaited(make_pipeline):
    """STT answering after the first STS chunk still decides: chunks are held back, then dropped or played."""
    client = fake_client(script=[[b"A1", b"A2"], [b"B1", b"B2"]], stt_script=[FILLER, REAL_WORDS],
                         stt_delay_s=0.15)
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance(1000, 1)[0])
    feed(p, utterance(1000, 2)[0])
    lines = events_until(p, "sent", timeout=2.0)
    assert lines == ['ignored sound: no words ("Hmm")', "phrase 1.4 s sent (total 2.8/60 s)"]
    assert sink.chunks == [b"B1", b"B2"]


def test_word_check_with_no_sts_chunks_does_not_hang(make_pipeline):
    client = fake_client(script=[[], [b"", b""]], stt_script=[FILLER, REAL_WORDS])
    p = make_pipeline(client=client)
    feed(p, utterance(1000, 1)[0])
    feed(p, utterance(1000, 2)[0])
    lines = events_until(p, "sent")
    assert lines == ['ignored sound: no words ("Hmm")', "phrase 1.4 s sent (total 2.8/60 s)"]


def test_word_check_timeout_fails_open(make_pipeline, monkeypatch):
    monkeypatch.setattr(sts, "WORD_CHECK_TIMEOUT_S", 0.05)
    client = fake_client(stt_script=[FILLER], stt_delay_s=0.5)  # would drop it, but answers too late
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance()[0])
    lines = events_until(p, "sent")
    assert lines == ["WARNING: word check failed (no answer after 0.05 s); playing phrase anyway",
                     "phrase 1.4 s sent (total 1.4/60 s)"]
    assert sink.chunks == [c for c in CHUNKS if c]


@pytest.mark.parametrize("status", [401, 403])
def test_missing_stt_permission_disables_check_once(make_pipeline, status):
    client = fake_client(script=[[b"A"], [b"B"]], stt_script=[missing_permission_error(status), FILLER])
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance(1000, 1)[0])
    feed(p, utterance(1000, 2)[0])
    lines = events_until(p, "total 2.8")
    assert len(lines) == 3
    assert lines[0].startswith("WARNING: word check disabled: ElevenLabs API key is missing a permission: "
                               "The API key you used is missing the permission speech_to_text.")
    assert "Speech to Text permission" in lines[0] and "--no-word-check" in lines[0]
    assert "Speech to Speech" not in lines[0]  # the STS key hint would be misleading here
    assert FAKE_KEY not in lines[0]
    assert lines[1:] == ["phrase 1.4 s sent (total 1.4/60 s)", "phrase 1.4 s sent (total 2.8/60 s)"]
    assert sink.chunks == [b"A", b"B"]  # failed open, then no check at all
    assert len(client.speech_to_text.calls) == 1
    assert p.error is None and not p.finished.is_set()


@pytest.mark.parametrize("exc, short", [
    (ApiError(status_code=503, headers={"xi-api-key": FAKE_KEY}, body=None), "ElevenLabs server error (503)"),
    (ApiError(status_code=429, headers={}, body={"detail": {"message": "Too many concurrent requests"}}),
     "ElevenLabs rate/concurrency limit hit: Too many concurrent requests"),
    (ApiError(status_code=401, headers={}, body={"detail": {"status": "quota_exceeded", "message": "No quota"}}),
     "Out of ElevenLabs credits: No quota"),
    (ConnectionError("network down"),
     "Network problem talking to ElevenLabs (ConnectionError: network down)"),
])
def test_word_check_error_fails_open_per_phrase(make_pipeline, exc, short):
    client = fake_client(script=[[b"A"], [b"B"]], stt_script=[exc, FILLER])
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance(1000, 1)[0])
    feed(p, utterance(1000, 2)[0])
    lines = events_until(p, "ignored")
    assert lines == [f"WARNING: word check failed ({short}); playing phrase anyway",
                     "phrase 1.4 s sent (total 1.4/60 s)",
                     'ignored sound: no words ("Hmm")']  # still checking after a transient error
    assert sink.chunks == [b"A"]
    assert len(client.speech_to_text.calls) == 2
    assert p.error is None and not p.finished.is_set()


def test_word_check_off_makes_no_stt_calls(make_pipeline):
    client = fake_client(stt_script=[FILLER])
    sink = Sink()
    p = make_pipeline(make_config(word_check=False), client=client, sink=sink)
    feed(p, utterance()[0])
    assert events_until(p, "sent") == ["phrase 1.4 s sent (total 1.4/60 s)"]
    assert sink.chunks == [c for c in CHUNKS if c]
    assert client.speech_to_text.calls == []


def test_dry_run_has_no_word_check(make_pipeline):
    sink = Sink()
    p = make_pipeline(make_config(dry_run=True, word_check=True), sink=sink)
    assert p._word_checker is None
    feed(p, utterance()[0])
    assert sink.wait(1)


def test_latency_recorded_only_for_played_phrases(make_pipeline):
    now = [0.0]
    tracker = LatencyTracker(clock=lambda: now[0])
    client = fake_client(script=[[b"A"], [b"B"], [b"C"]], stt_script=[FILLER, SIGH, REAL_WORDS])
    sink = Sink()
    p = make_pipeline(client=client, sink=sink, latency=tracker)
    for seed in (1, 2):
        feed(p, utterance(1000, seed)[0])
    assert len(events_until(p, "(sighs)")) == 2
    assert tracker.timings == []
    now[0] = 5.0
    feed(p, utterance(1000, 3)[0])
    assert sink.wait(1)
    assert sink.chunks == [b"C"]
    assert wait_for(lambda: len(tracker.timings) == 1)
    assert tracker.timings[0].speech_end == 4.5  # replaced the dropped phrases' timings (5.0 minus the pause)
    assert tracker.timings[0].utterance_id == 3


def test_stop_while_waiting_for_word_check_is_prompt(make_pipeline):
    client = fake_client(stt_delay_s=1.0)
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance()[0])
    assert wait_for(lambda: len(client.speech_to_text.calls) == 1)
    time.sleep(0.05)  # worker is now holding the first chunk, waiting for the verdict
    t0 = time.monotonic()
    p.stop()
    assert time.monotonic() - t0 < 0.3
    assert not p._thread.is_alive()
    assert sink.chunks == []
    assert client.speech_to_speech.closed.is_set()


# --- dry run ---

def test_dry_run_echoes_without_client(make_pipeline):
    sink = Sink()
    p = make_pipeline(make_config(dry_run=True, api_key=None, voice_id=None), sink=sink)
    assert p._client is None
    assert p.output_sample_rate == 16_000
    audio, phrase = utterance()
    feed(p, audio)
    assert sink.wait(1)
    assert sink.chunks == [phrase]
    assert events_until(p, "echoed") == ["phrase 1.4 s echoed (dry run, no credits; total 1.4 s)"]
    assert p.speech_sent_s == 0.0
    assert p.speech_total_s == pytest.approx(1.4)


def test_dry_run_ignores_cap(make_pipeline):
    sink = Sink()
    p = make_pipeline(make_config(dry_run=True, max_speech_s=1.0), sink=sink)
    for seed in (1, 2):
        feed(p, utterance(1000, seed)[0])
    assert sink.wait(2)
    assert not p.finished.is_set()


# --- errors ---

def test_fatal_error_sets_error_and_finished(make_pipeline):
    err = ApiError(status_code=401, headers={"xi-api-key": FAKE_KEY},
                   body={"detail": {"status": "invalid_api_key", "message": "Invalid API key"}})
    client = fake_client(script=[err])
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance(1000, 1)[0])
    assert p.finished.wait(WAIT_S)
    assert isinstance(p.error, ApiFatalError)
    lines = events_until(p, "ERROR:")
    assert len(lines) == 1
    assert lines[0].startswith("ERROR: ElevenLabs rejected the API key: Invalid API key.")
    assert FAKE_KEY not in lines[0] and FAKE_KEY not in str(p.error)
    assert sink.chunks == []

    feed(p, utterance(1000, 2)[0])  # ignored: the pipeline has finished
    time.sleep(0.02)
    assert len(client.speech_to_speech.calls) == 1


@pytest.fixture
def fast_retry(monkeypatch):
    monkeypatch.setattr(sts, "RETRY_DELAY_S", 0.01)


def server_error() -> ApiError:
    return ApiError(status_code=503, headers={"xi-api-key": FAKE_KEY}, body=None)


def test_temporary_error_is_retried_once(make_pipeline, fast_retry):
    client = fake_client(script=[server_error(), [b"ok"]])
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance(1000, 1)[0])
    assert sink.wait(1)
    lines = events_until(p, "total 2.8")
    assert lines == ["WARNING: ElevenLabs server error (503). Try again shortly. Retrying the phrase.",
                     "phrase 1.4 s sent (total 2.8/60 s)"]  # the failed upload still counts as sent
    assert sink.chunks == [b"ok"]
    assert len(client.speech_to_speech.calls) == 2
    assert p.error is None and not p.finished.is_set()


def test_second_failure_skips_phrase_and_continues(make_pipeline, fast_retry):
    client = fake_client(script=[server_error(), server_error(), [b"ok"]])
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance(1000, 1)[0])
    feed(p, utterance(1000, 2)[0])
    assert sink.wait(1)
    lines = events_until(p, "total 4.2")
    assert lines[:2] == ["WARNING: ElevenLabs server error (503). Try again shortly. Retrying the phrase.",
                         "skipped phrase: ElevenLabs server error (503). Try again shortly."]
    assert sink.chunks == [b"ok"]


def test_no_retry_once_audio_has_played(make_pipeline, fast_retry):
    class HalfwayFail(FakeSts):
        def stream(self, voice_id, **kwargs):
            self.calls.append(kwargs)

            def gen():
                yield b"half"
                raise ConnectionError("dropped")
            return gen()

    sts_client = HalfwayFail()
    sink = Sink()
    p = make_pipeline(client=SimpleNamespace(speech_to_speech=sts_client, speech_to_text=FakeStt()), sink=sink)
    feed(p, utterance()[0])
    lines = events_until(p, "skipped")
    assert lines[-1].startswith("skipped phrase: Network problem")
    assert sink.chunks == [b"half"] and len(sts_client.calls) == 1  # a retry would repeat what was heard


def test_error_raised_by_stream_call_itself_is_handled(make_pipeline, fast_retry):
    class Boom:
        def stream(self, voice_id, **kwargs):
            raise ConnectionError("network down")

    p = make_pipeline(client=SimpleNamespace(speech_to_speech=Boom(), speech_to_text=FakeStt()))
    feed(p, utterance()[0])
    lines = events_until(p, "skipped")
    msg = "Network problem talking to ElevenLabs (ConnectionError: network down). Check your connection."
    assert lines == [f"WARNING: {msg} Retrying the phrase.", f"skipped phrase: {msg}"]
    assert p.speech_sent_s == 0.0  # nothing was uploaded


# --- backlog ---

def test_backlog_drops_oldest_phrases(make_pipeline):
    sink = Sink()
    p = make_pipeline(make_config(dry_run=True), sink=sink, start=False)
    phrases = []
    for seed in range(MAX_QUEUED_PHRASES + 2):
        audio, phrase = utterance(600, seed)  # above vad.py's minimum phrase length
        phrases.append(phrase)
        feed(p, audio)
    assert p.drain_events() == ["WARNING: falling behind; dropped 1 queued phrase(s)."] * 2

    p.start()
    assert sink.wait(MAX_QUEUED_PHRASES)
    assert sink.chunks == phrases[-MAX_QUEUED_PHRASES:]  # newest kept, in order


# --- stop / feed after stop ---

def test_stop_is_prompt_and_idempotent_when_idle(make_pipeline):
    p = make_pipeline()
    t0 = time.monotonic()
    p.stop()
    p.stop()
    assert time.monotonic() - t0 < 0.2
    assert not p._thread.is_alive()


def test_stop_unstarted_pipeline(make_pipeline):
    p = make_pipeline(start=False)
    p.stop()
    p.stop()


def test_stop_interrupts_a_long_stream(make_pipeline):
    client = fake_client(script=[[b"\x00\x01"] * 10_000], delay_s=0.002)  # ~20 s if never interrupted
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance()[0])
    assert sink.wait(2)
    t0 = time.monotonic()
    p.stop()
    assert time.monotonic() - t0 < 0.3
    assert client.speech_to_speech.closed.is_set()  # the HTTP response would be closed
    assert len(sink.chunks) < 1000
    p.stop()


def test_stop_discards_half_phrase(make_pipeline):
    client = fake_client()
    p = make_pipeline(client=client)
    feed(p, silence(300) + speech(500))  # still in speech, no ending silence
    p.stop()
    time.sleep(0.02)
    assert client.speech_to_speech.calls == []


def test_feed_audio_ignored_after_stop(make_pipeline):
    seg = PhraseSegmenter(VoiceActivityDetector(use_webrtc=False))
    client = fake_client()
    p = make_pipeline(client=client, segmenter=seg)
    p.stop()
    feed(p, utterance()[0])
    assert not seg.in_speech
    assert client.speech_to_speech.calls == []


def test_feed_audio_ignored_after_finished(make_pipeline):
    seg = PhraseSegmenter(VoiceActivityDetector(use_webrtc=False))
    p = make_pipeline(segmenter=seg, start=False)
    p.finished.set()
    feed(p, silence(100) + speech(200))
    assert not seg.in_speech


def test_start_is_idempotent(make_pipeline):
    p = make_pipeline()
    thread = p._thread
    p.start()
    assert p._thread is thread


# --- construction ---

@pytest.mark.parametrize("key, voice", [(None, FAKE_VOICE), (FAKE_KEY, None), (None, None)])
def test_missing_credentials_raise_config_error(key, voice):
    with pytest.raises(ConfigError, match="Missing ELEVENLABS_") as info:
        StsPipeline(make_config(api_key=key, voice_id=voice), lambda pcm: None)
    assert FAKE_KEY not in str(info.value)


def test_builds_client_from_key_when_none_given(monkeypatch):
    made = []
    monkeypatch.setattr(sts, "make_client", lambda api_key: made.append(api_key) or fake_client())
    p = StsPipeline(make_config(), lambda pcm: None)
    assert made == [FAKE_KEY]
    assert p.output_sample_rate == 48_000  # default pcm_48000
    p.stop()


def test_default_vad_uses_config_aggressiveness():
    p = StsPipeline(make_config(dry_run=True, vad_aggressiveness=1), lambda pcm: None)
    assert p._segmenter.vad.aggressiveness == 1
    p.stop()


def test_vad_dropped_sound_is_reported_and_not_sent(make_pipeline):
    client = fake_client()
    p = make_pipeline(client=client)
    feed(p, silence(300) + speech(300) + silence(600))  # 300 ms voiced: under the 450 ms minimum
    assert events_until(p, "ignored sound") == ["ignored sound: too short (0.30 s)"]
    assert client.speech_to_speech.calls == []


def test_voice_settings_not_sent_by_default(make_pipeline):
    client = fake_client()
    sink = Sink()
    p = make_pipeline(client=client, sink=sink)
    feed(p, utterance()[0])
    assert sink.wait(3)
    assert "voice_settings" not in client.speech_to_speech.calls[0]  # the voice's stored settings apply


def test_voice_settings_sent_as_json_when_overridden(make_pipeline):
    client = fake_client()
    sink = Sink()
    p = make_pipeline(make_config(stability=0.3), client=client, sink=sink)
    feed(p, utterance()[0])
    assert sink.wait(3)
    sent = json.loads(client.speech_to_speech.calls[0]["voice_settings"])
    assert sent == {"stability": 0.3, "similarity_boost": 0.9, "style": 0.0, "use_speaker_boost": True}
