"""Pipeline B: VAD phrases -> ElevenLabs speech-to-speech (Voice Changer) stream -> audio out.

Formats:
    In:  16 kHz mono int16 LE PCM, 20 ms frames (config.py). Whole phrases are
         uploaded as raw PCM with file_format="pcm_s16le_16", which skips
         server-side decoding and lowers latency.
    Out: raw mono int16 LE PCM at config.output_rate (e.g. "pcm_48000"),
         streamed in chunks of arbitrary, possibly odd, length. AudioOutput
         copes with odd lengths, so chunks are passed on untouched.
    Dry run: no API call; each phrase is echoed back unchanged at 16 kHz
         (no word check either: there is no client).

Threads:
    audio callback -> feed_audio(): VAD/segmenter only, then a non-blocking
        put of each finished phrase on a small queue.
    worker (daemon) -> one phrase at a time, in order: credit cap check, the
        streaming request, on_audio_out() per chunk, a status line on `events`.
    word check (one daemon thread per phrase, only when config.word_check) -> batch
        speech-to-text of the same phrase, run IN PARALLEL with the STS request.

Word check flow per phrase (pipelines/word_filter.py decides what a real word is):
    1. Worker submits the STT check, then starts reading the STS stream.
    2. STS chunks are held back until the first non-empty one arrives (~700 ms;
       STT usually answers sooner), then the worker waits up to
       WORD_CHECK_TIMEOUT_S for the verdict.
    3. Real words: the held chunk is delivered and the rest streams straight through.
       No words ("hmm", "uh", a sigh): nothing is delivered, the STS stream is
       closed, and an "ignored sound" line is emitted.
    STT errors and timeouts never stop the pipeline: they fail open (the phrase
    plays). A key without the Speech to Text permission turns the check off for
    the rest of the session.

Errors: fatal ones (bad key, quota, unknown voice) stop the pipeline. A temporary one (5xx, 429,
network) is retried once after RETRY_DELAY_S if none of the phrase had played yet; otherwise the
phrase is skipped. A retry uploads (and counts towards the cap) again.

Latency stages (LatencyTracker marks; "end of speech" = phrase emitted minus config.pause_s):
    pause       the silence the segmenter waits for before it ends the phrase (= config.pause_s)
    queue       waiting behind earlier phrases
    voice       upload + conversion until the first audio chunk arrives
    word check  extra wait for the word check's verdict (only with the word check on)
"""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future
from typing import Callable

from elevenlabs import ElevenLabs

from config import (SAMPLE_RATE, SAMPLE_WIDTH, STS_LATENCY_OPTIMIZATION, STS_MODEL_ID, Config,
                    require_credentials, voice_settings)
from latency import LatencyTracker
from pipelines.base import (KEY_HINT, RETRY_DELAY_S, ApiFatalError, Pipeline, _is_missing_permission,
                            describe_api_error, make_client)
from pipelines.word_filter import WordChecker, WordVerdict
from vad import PhraseSegmenter, VoiceActivityDetector

MAX_QUEUED_PHRASES = 3  # beyond this we're falling behind; drop the oldest rather than build a backlog
STOP_JOIN_TIMEOUT_S = 1.0  # a blocked network read can't be interrupted; the worker is a daemon
INPUT_FILE_FORMAT = "pcm_s16le_16"  # raw 16-bit 16 kHz mono LE PCM, exactly what the segmenter emits
WORD_CHECK_TIMEOUT_S = 3.0  # max wait for the STT verdict after the first STS chunk; then play anyway
WORD_CHECK_POLL_S = 0.05  # verdict wait granularity, so stop() isn't held up by a slow check
MAX_SHOWN_TRANSCRIPT_CHARS = 40
MAX_SHORT_ERROR_CHARS = 120


def phrase_seconds(phrase: bytes) -> float:
    return len(phrase) / SAMPLE_WIDTH / SAMPLE_RATE


class StsPipeline(Pipeline):
    """Speech-to-speech pipeline. `client`, `vad` and `segmenter` are injectable for tests.

    Not restartable: create a new instance after stop().
    """

    def __init__(
        self,
        config: Config,
        on_audio_out: Callable[[bytes], None],
        client: ElevenLabs | None = None,
        latency: LatencyTracker | None = None,
        vad: VoiceActivityDetector | None = None,
        segmenter: PhraseSegmenter | None = None,
    ) -> None:
        super().__init__(on_audio_out)
        self.config = config
        self.input_sample_rate = SAMPLE_RATE
        self.output_sample_rate = SAMPLE_RATE if config.dry_run else config.output_rate

        if client is None and not config.dry_run:
            require_credentials(config)  # ConfigError with a clear message if key/voice is missing
            client = make_client(config.api_key)
        self._client = client
        self._latency = latency
        if segmenter is None:
            segmenter = PhraseSegmenter(vad or VoiceActivityDetector(aggressiveness=config.vad_aggressiveness),
                                        end_silence_ms=round(config.pause_s * 1000))
        self._segmenter = segmenter
        self._segmenter_lock = threading.Lock()  # feed_audio (audio thread) vs flush in stop() (main thread)

        # Items are (phrase, time the segmenter emitted it on the latency clock); None = sentinel.
        self._queue: queue.Queue[tuple[bytes, float | None] | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sent_s = 0.0  # billed seconds (written only by the worker)
        self._total_s = 0.0  # all processed seconds, including dry-run echoes
        self._chunks_out = 0  # chunks delivered so far (a retry is only safe if a phrase delivered none)

        # Voice settings are only sent when overridden (--stability/--similarity, GUI sliders); otherwise the
        # voice's stored settings apply. The STS endpoint takes them as a JSON string; speed doesn't apply here.
        self._voice_settings_json: str | None = None
        if config.stability is not None or config.similarity is not None:
            settings = voice_settings(config)
            settings.pop("speed")
            self._voice_settings_json = json.dumps(settings)

        # Word check: only with a real client. Set back to None (by the worker) if the key lacks permission.
        self._word_checker: WordChecker | None = None
        if config.word_check and not config.dry_run:
            self._word_checker = WordChecker(client)

    @property
    def speech_sent_s(self) -> float:
        """Seconds of speech sent to ElevenLabs speech-to-speech (billed) so far.

        Includes phrases the word check ignored (they were uploaded before the verdict),
        excludes the word check's own speech-to-text uploads.
        """
        return self._sent_s

    @property
    def speech_total_s(self) -> float:
        """Seconds of speech processed so far, including dry-run phrases."""
        return self._total_s

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="sts-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._segmenter_lock:
            self._segmenter.flush()  # discard any half phrase
        self._queue.put_nowait(None)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=STOP_JOIN_TIMEOUT_S)

    def feed_audio(self, frame: bytes) -> None:
        if self._stop.is_set() or self.finished.is_set():
            return
        with self._segmenter_lock:
            was_in_speech = self._segmenter.in_speech
            phrase = self._segmenter.feed(frame)
            dropped_reason = (self._segmenter.last_drop_reason
                              if phrase is None and was_in_speech and not self._segmenter.in_speech else None)
        if dropped_reason is not None:
            self._emit(f"ignored sound: {dropped_reason}")  # VAD's free first-pass filter (too short/quiet)
        if phrase is None:
            return
        # Timing starts in the worker (which handles one phrase at a time), so a phrase ending
        # while the previous one is still in flight can't overwrite that one's speech-end time.
        emitted = self._latency.now() if self._latency is not None else None
        self._queue.put_nowait((phrase, emitted))
        dropped = 0
        while self._queue.qsize() > MAX_QUEUED_PHRASES:
            try:
                self._queue.get_nowait()
            except queue.Empty:  # the worker took it first
                break
            dropped += 1
        if dropped:
            self._emit(f"WARNING: falling behind; dropped {dropped} queued phrase(s).")

    # --- worker thread ---

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None or self._stop.is_set():
                break
            phrase, emitted = item
            if self._latency is not None and emitted is not None:
                # The speaker actually stopped `pause_s` before the segmenter emitted the phrase.
                self._latency.mark_speech_end(at=emitted - self.config.pause_s)
                self._latency.mark("pause", at=emitted)
                self._latency.mark("queue")
            if not self._process(phrase):
                break

    def _process(self, phrase: bytes) -> bool:
        """Handle one phrase. Returns False if the pipeline must stop (cap reached or fatal error)."""
        seconds = phrase_seconds(phrase)
        if self.config.dry_run:
            self._deliver(phrase, first=True)
            self._total_s += seconds
            self._emit(f"phrase {seconds:.1f} s echoed (dry run, no credits; total {self._total_s:.1f} s)")
            return True

        cap = self.config.max_speech_s
        if cap > 0 and self._sent_s + seconds > cap:
            self._emit(f"Speech cap reached ({cap:g} s). Use --max-speech to raise it.")
            self.finished.set()
            return False

        for attempt in (1, 2):
            chunks_before = self._chunks_out
            try:
                played = self._stream_phrase(phrase, seconds)
                break
            except Exception as exc:  # noqa: BLE001 - classified by describe_api_error
                msg, fatal = describe_api_error(exc)
                if fatal:
                    self.error = ApiFatalError(msg)
                    self._emit(f"ERROR: {msg}")
                    self.finished.set()
                    return False
                if attempt == 1 and self._chunks_out == chunks_before and not self._stop.is_set():
                    self._emit(f"WARNING: {msg} Retrying the phrase.")
                    if self._stop.wait(RETRY_DELAY_S):
                        return True
                    continue
                self._emit(f"skipped phrase: {msg}")
                return True

        if not played:
            return True  # the word check dropped it and already said so
        cap_text = f"/{cap:g}" if cap > 0 else ""
        self._emit(f"phrase {seconds:.1f} s sent (total {self._sent_s:.1f}{cap_text} s)")
        return True

    def _stream_phrase(self, phrase: bytes, seconds: float) -> bool:
        """Send one phrase and play the result. Returns False if the word check dropped it."""
        # The SDK call is a lazy generator: the HTTP request goes out on the first next().
        extra = {}
        if self._voice_settings_json is not None:
            extra["voice_settings"] = self._voice_settings_json
        chunks = self._client.speech_to_speech.stream(
            self.config.voice_id,
            audio=("phrase.pcm", phrase, "application/octet-stream"),
            model_id=STS_MODEL_ID,
            output_format=self.config.output_format,
            optimize_streaming_latency=STS_LATENCY_OPTIMIZATION,
            file_format=INPUT_FILE_FORMAT,
            remove_background_noise=self.config.denoise,
            **extra,
        )
        self._sent_s += seconds  # counted up front: once the upload starts it may be billed
        self._total_s += seconds
        verdict = self._submit_word_check(phrase)  # before the first next(), so both requests overlap
        first = True
        try:
            if verdict is not None:
                held = b""
                for chunk in chunks:  # hold back the first non-empty chunk until the verdict is in
                    if self._stop.is_set():
                        return True
                    if chunk:
                        held = chunk
                        self._mark("voice")
                        break
                if not self._await_word_verdict(verdict):
                    return False
                if held and not self._stop.is_set():
                    self._mark("word check")
                    self._deliver(held, first)
                    first = False
            for chunk in chunks:
                if self._stop.is_set():
                    break
                if chunk:
                    if first:
                        self._mark("voice")
                    self._deliver(chunk, first)
                    first = False
        finally:
            if verdict is not None:
                verdict.cancel()  # no-op unless the check never started (e.g. the STS request failed)
            close = getattr(chunks, "close", None)
            if close is not None:
                close()  # closes the HTTP response if we broke out early

        return True

    def _submit_word_check(self, phrase: bytes) -> Future[WordVerdict] | None:
        """Start the speech-to-text check on its own daemon thread (one per phrase), so a hung
        check can neither delay the next phrase's check nor keep the process alive at exit."""
        checker = self._word_checker
        if checker is None or self._stop.is_set():
            return None
        future: Future[WordVerdict] = Future()

        def run() -> None:
            if not future.set_running_or_notify_cancel():
                return  # cancelled before it started
            try:
                future.set_result(checker.check(phrase))
            except BaseException as exc:  # noqa: BLE001 - handed to the worker via the future
                future.set_exception(exc)

        threading.Thread(target=run, name="sts-word-check", daemon=True).start()
        return future

    def _await_word_verdict(self, future: Future[WordVerdict]) -> bool:
        """Wait (bounded) for the word check. Returns False only if the phrase has no real words.

        Fails open: on a timeout, an error, or stop(), the phrase is played.
        """
        deadline = time.monotonic() + WORD_CHECK_TIMEOUT_S
        while not future.done():
            if self._stop.is_set():
                return True  # the caller sees the stop flag and delivers nothing
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._emit(f"WARNING: word check failed (no answer after {WORD_CHECK_TIMEOUT_S:g} s); "
                           "playing phrase anyway")
                return True
            time.sleep(min(WORD_CHECK_POLL_S, remaining))

        if future.cancelled():
            return True
        exc = future.exception()
        if exc is not None:
            self._word_check_failed(exc)
            return True

        result = future.result()
        if result.has_words:
            return True
        shown = result.text
        if len(shown) > MAX_SHOWN_TRANSCRIPT_CHARS:
            shown = shown[:MAX_SHOWN_TRANSCRIPT_CHARS] + "..."
        self._emit(f'ignored sound: no words ("{shown}")' if shown else "ignored sound: no words (silence)")
        return False

    def _word_check_failed(self, exc: BaseException) -> None:
        msg, _ = describe_api_error(exc)  # type: ignore[arg-type] - futures only hold Exceptions here
        if _is_missing_permission(exc):  # type: ignore[arg-type]
            # Won't fix itself: stop paying for checks that can't succeed. KEY_HINT is about STS, so swap it.
            self._word_checker = None
            msg = msg.replace(KEY_HINT, "").strip()
            self._emit(f"WARNING: word check disabled: {msg} Enable the Speech to Text permission on the key, "
                       "or use --no-word-check. Non-word sounds will be played.")
            return
        # Quota errors land here too and fail open: the STS request reports quota itself.
        short = msg.split(". ")[0].rstrip(".")
        if len(short) > MAX_SHORT_ERROR_CHARS:
            short = short[:MAX_SHORT_ERROR_CHARS] + "..."
        self._emit(f"WARNING: word check failed ({short}); playing phrase anyway")

    def _mark(self, stage: str) -> None:
        if self._latency is not None:
            self._latency.mark(stage)

    def _deliver(self, pcm: bytes, first: bool) -> None:
        if first and self._latency is not None:
            self._latency.mark_first_audio_out()
        self._chunks_out += 1
        self.on_audio_out(pcm)
