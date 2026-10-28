"""Pipeline A: realtime speech-to-text -> text-to-speech -> audio out.

Two timings (config.timing, --timing):

    smooth (default): each committed phrase is spoken whole, after the user pauses.
        mic frames -> SttStream (Scribe v2 Realtime; the server's VAD commits a segment after
        config.pause_s of silence; no_verbatim drops most fillers) -> on commit, TextStabilizer
        strips fillers/audio events -> phrase speaker: one streaming HTTP text-to-speech request
        per phrase (client.text_to_speech.stream, config.tts_model, default eleven_v4_turbo,
        voice_settings(config)) -> on_audio_out. Partials only track whether a segment is open
        (for the credit cap). No TTS websocket is ever opened. Any TTS model works, and the
        voice sees the whole sentence, so pace and rhythm sound natural.

    balanced / fast: speech starts mid-sentence over the TTS websocket (Flash v2.5 only).
        mic frames -> SttStream -> TextStabilizer (speaks settled words of partial transcripts
        early, the rest on commit, strips fillers; holdback from STREAMING_TIMINGS) -> TtsStream
        (chunk_length_schedule from STREAMING_TIMINGS, voice_settings(config) in the init message,
        flushed at every commit) -> on_audio_out.

Formats:
    In:  16 kHz mono int16 LE PCM, 20 ms frames (config.py), batched into
         STT_SEND_CHUNK_MS (100 ms) websocket messages.
    Out: raw mono int16 LE PCM at config.output_rate (e.g. "pcm_48000"), in chunks of
         arbitrary, possibly odd, length, from either TTS path. AudioOutput copes with that,
         so chunks are passed on untouched.

Threads / async flow:
    audio callback -> feed_audio(): local VAD (credit cap; latency start in balanced/fast) only,
        then a non-blocking loop.call_soon_threadsafe(put frame on an asyncio queue).
    loop thread (daemon, started by start()) runs one asyncio event loop with these tasks:
        STT loop      connects SttStream (reconnecting with backoff after non-fatal drops)
                      and iterates its events. smooth: committed -> phrase queue, "heard" line.
                      balanced/fast: partial -> stabilizer -> TTS queue; committed -> rest of
                      the segment + a flush -> TTS queue, "heard" line.
        STT sender    batches queued mic frames into 100 ms send_audio() calls. Frames that
                      arrive while STT is disconnected/reconnecting are dropped.
      balanced/fast only:
        TTS sender    the ONLY task that writes to TTS, so text keeps its order and two sends
                      can never both reconnect. Connects lazily (first text, or when the
                      local VAD hears speech start, to warm the socket) and again whenever
                      the previous socket closed (inactivity timeout, error). A text whose
                      send hits a just-closed socket is resent on a new one.
        TTS reader    one long-lived task per TTS connection: `async for chunk in audio()`
                      -> on_audio_out(). After a flush the socket stays open and no isFinal
                      arrives, so the reader only ends when the socket closes.
    speaker thread (smooth only; daemon, started by start()): takes phrases from a queue.Queue
        one at a time, in order, and runs the blocking SDK stream for each (the request goes
        out on the first next()), calling on_audio_out() per chunk and checking the halt flag
        between chunks. A dedicated daemon thread rather than asyncio.to_thread: the default
        executor's threads are not daemons, so a blocked HTTP read there would hang exit;
        stop() can't interrupt such a read, so it joins this thread with a timeout instead.
    on_audio_out runs on the loop thread (balanced/fast) or the speaker thread (smooth);
    AudioOutput.write is thread-safe.
    Each TTS connection is a fresh TtsStream (and each STT connection a fresh SttStream)
    from the factory, so a dying socket's reader can never race with its replacement.

Credit cap (config.max_speech_s, 0 = none): seconds of the user's speech as detected by the
local PhraseSegmenter (frames counted while it is in speech, including the short pause that
ends a phrase), so it means the same as in sts mode. When exceeded: a message is emitted,
mic audio is no longer forwarded, ~0.7 s of silence is sent so the server commits the last
segment, and then `finished` is set: smooth = once the phrase queue is empty and the current
phrase has finished streaming; balanced/fast = once the flushed text's audio has gone quiet.

Latency (LatencyTracker):
    smooth: "end of speech -> first audio out", as in sts. The commit time is taken when the
        commit arrives (on the loop thread) and handed to the speaker with the phrase, which
        calls mark_speech_end(at=commit - config.pause_s) when it starts that phrase (so a phrase
        queued behind another can't overwrite its timing) and mark_first_audio_out() on its
        first chunk. Stages: pause (the server's silence wait, ~config.pause_s, plus transcription
        until the commit arrives), queue (behind earlier phrases), voice (TTS until first audio).
    balanced/fast: "start of speech -> first audio out". mark_speech_end() is called when
        the local segmenter ENTERS speech (at = when the first frame of its start run began),
        so the tracker's "speech end" means speech START. The first audio chunk received
        after text of that utterance was sent to TTS completes it.

Errors: a fatal RealtimeError (bad key, no credits, unknown voice) from either stream sets
`error` (ApiFatalError) and `finished`, emits "ERROR: ...", and shuts the tasks down; so does
any unexpected exception in a task or the speaker. Non-fatal STT drops reconnect with backoff
(the open segment is lost; in balanced/fast what was already sent is still spoken).
smooth: speaker errors are classified by describe_api_error; fatal ones (bad key, quota,
unknown voice, 400/422 e.g. an unsupported model) stop the pipeline. A non-fatal one (5xx,
429, network) is retried once after RETRY_DELAY_S if none of the phrase had played yet;
otherwise (or if the retry fails too) a WARNING is emitted and the phrase is skipped.

Voice switching (set_voice, from any thread): smooth and sts read the voice per phrase. The
websocket timings reconnect TTS with the new voice at the next sentence boundary: only when no
text is waiting for a flush and no audio has arrived for VOICE_SWITCH_IDLE_S, so nothing is cut off.
balanced/fast: non-fatal TTS failures emit a WARNING, skip the rest of that sentence, and the
next text reconnects.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
from typing import Any, Callable, Coroutine

import httpx
from elevenlabs import VoiceSettings
from elevenlabs.core.api_error import ApiError
from elevenlabs.core.parse_error import ParsingError

from config import (FRAME_MS, LANGUAGE_CODE, SAMPLE_RATE, SAMPLE_WIDTH, STREAMING_TIMINGS, STT_MODEL_ID,
                    STT_SEND_CHUNK_MS, STT_VAD_SILENCE_S, TTS_INACTIVITY_TIMEOUT_S, DELIVERY_TAG_MODELS, Config,
                    require_credentials, voice_settings)
from latency import LatencyTracker
from pipelines.base import KEY_HINT, RETRY_DELAY_S, ApiFatalError, Pipeline, describe_api_error, make_client
from pipelines.realtime import TTS_KEY_HINT, RealtimeError, SttEvent, SttStream, TtsStream
from pipelines.text_stream import TextStabilizer
from vad import PhraseSegmenter, VoiceActivityDetector

logger = logging.getLogger(__name__)

SEND_CHUNK_BYTES = SAMPLE_RATE * STT_SEND_CHUNK_MS // 1000 * SAMPLE_WIDTH  # 3200 bytes = 100 ms
STOP_JOIN_TIMEOUT_S = 2.0  # the loop thread is a daemon; don't hang Ctrl+C on a stuck socket
SPEAKER_JOIN_TIMEOUT_S = 1.0  # smooth: a blocked HTTP read can't be interrupted; the speaker is a daemon
CLOSE_TIMEOUT_S = 0.5  # per step of shutdown: cancelling tasks, closing each socket
STT_RECONNECT_DELAYS_S = (0.5, 1.0, 2.0, 4.0)  # backoff after consecutive failures; the last one repeats
STT_RECONNECT_RESET_S = 10.0  # a connection that lasted this long resets the backoff
CAP_SILENCE_EXTRA_MS = 200  # silence sent after the cap = the pause + this, so the server commits
VOICE_SWITCH_IDLE_S = 1.0  # websocket timings: switch voices only after this long without TTS audio
CAP_COMMIT_TIMEOUT_S = 3.0  # max wait for that last commit
CAP_AUDIO_IDLE_S = 1.5  # TTS sends no isFinal after a flush: "done" = no audio for this long
CAP_DRAIN_MAX_S = 15.0  # hard limit on waiting for the last TTS audio (or queued phrases) after the cap
POLL_S = 0.02
MAX_SHOWN_TRANSCRIPT_CHARS = 60
SILENCE_FRAME = bytes(SAMPLE_RATE * FRAME_MS // 1000 * SAMPLE_WIDTH)
# Exceptions from the SDK's HTTP stream that describe_api_error classifies; anything else is a bug (fatal).
API_ERRORS = (ApiError, ParsingError, httpx.HTTPError, ConnectionError, TimeoutError)

SttFactory = Callable[[], Any]  # returns an unconnected SttStream (or a test fake with the same methods)
TtsFactory = Callable[[], Any]  # returns an unconnected TtsStream (or a test fake)
PhraseItem = tuple[str, float | None]  # (text to speak, commit time on the latency clock)


def calm_punctuation(text: str) -> str:
    """Exclamation marks make the voice sound excited: "Wow!" -> "Wow.", "Really?!" -> "Really?"."""
    return re.sub(r"\?!+", "?", text).replace("!", ".")


def with_delivery_tag(text: str, config: Config) -> str:
    """Prefix a delivery tag ("[calm, relaxed] ...") for v3/v4 models, which follow it without reading it aloud."""
    if config.delivery and config.tts_model in DELIVERY_TAG_MODELS:
        return f"[{config.delivery}] {text}"
    return text


def _shown(text: str) -> str:
    text = " ".join(text.split())
    return text[:MAX_SHOWN_TRANSCRIPT_CHARS].rstrip() + "..." if len(text) > MAX_SHOWN_TRANSCRIPT_CHARS else text


class SttTtsPipeline(Pipeline):
    """Realtime STT -> TTS pipeline.

    Injectable for tests: `stt_factory`, `vad`, and per timing either `tts_client` (smooth: an object
    with `.text_to_speech.stream`, default an ElevenLabs client) or `tts_factory` (balanced/fast).

    Not restartable: create a new instance after stop().
    """

    def __init__(
        self,
        config: Config,
        on_audio_out: Callable[[bytes], None],
        latency: LatencyTracker | None = None,
        *,
        stt_factory: SttFactory | None = None,
        tts_factory: TtsFactory | None = None,
        tts_client: Any = None,
        vad: VoiceActivityDetector | None = None,
    ) -> None:
        super().__init__(on_audio_out)
        self.config = config
        self.input_sample_rate = SAMPLE_RATE
        self.output_sample_rate = config.output_rate
        self._smooth = config.timing == "smooth"
        needs_default_tts = tts_client is None if self._smooth else tts_factory is None
        if stt_factory is None or needs_default_tts:
            require_credentials(config)  # ConfigError with a clear message if key/voice is missing
        self._stt_factory = stt_factory or self._default_stt
        self._tts_factory = tts_factory or self._default_tts
        if self._smooth and tts_client is None:
            tts_client = make_client(config.api_key)
        self._tts_client = tts_client  # smooth only
        self._voice_settings_dict = voice_settings(config)
        self._voice_settings = VoiceSettings(**self._voice_settings_dict)
        self._latency = latency

        # Audio-thread state (feed_audio only).
        self._segmenter = PhraseSegmenter(vad or VoiceActivityDetector(aggressiveness=config.vad_aggressiveness),
                                          end_silence_ms=round(config.pause_s * 1000))
        self._speech_frames = 0  # frames counted towards the cap (read by speech_sent_s from other threads)
        self._capped = False

        # Cross-thread state.
        self._stopped = False
        self._halt = threading.Event()  # stop() or a fatal error: the speaker stops between chunks
        self._fail_lock = threading.Lock()  # _fail can run on the loop thread and the speaker thread
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # Smooth: phrase queue (loop thread -> speaker thread). None = sentinel to unblock the speaker.
        self._phrases: queue.Queue[PhraseItem | None] = queue.Queue()
        self._phrases_lock = threading.Lock()
        self._phrases_pending = 0  # queued + currently streaming phrases (the cap drain waits for 0)
        self._phrase_audio_started = False  # the current phrase delivered audio (then it isn't retried)
        self._speaker: threading.Thread | None = None

        # Loop-thread state. asyncio primitives bind to the loop on first use, so creating them here is fine.
        self._shutdown = asyncio.Event()
        self._frames: asyncio.Queue[bytes] = asyncio.Queue()
        self._tts_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()  # ("connect"|"text"|"flush", text)
        self._tasks: set[asyncio.Task] = set()
        self._stt: Any = None  # current open SttStream, None while (re)connecting
        self._tts: Any = None  # current TtsStream, None until the next text reconnects
        holdback = TextStabilizer().holdback_words if self._smooth else STREAMING_TIMINGS[config.timing][0]
        self._stabilizer = TextStabilizer(holdback_words=holdback)  # smooth only uses on_committed
        self._segment_open = False  # a partial with text arrived and its commit hasn't yet
        self._segment_chars = 0  # chars queued for TTS in the current segment
        self._unflushed = False  # text sent on the current TTS socket since the last flush
        self._skip_segment = False  # TTS failed mid-sentence: drop text until the next flush
        self._tts_chars = 0
        self._last_tts_activity = 0.0  # loop time of the last text/flush sent or audio received
        self._latency_armed = False  # text of the current utterance was sent; next audio chunk completes timing

    # --- defaults ---

    def _default_stt(self) -> SttStream:
        return SttStream(self.config.api_key, model_id=STT_MODEL_ID, language_code=LANGUAGE_CODE,
                         vad_silence_s=self.config.pause_s, no_verbatim=True, sample_rate=SAMPLE_RATE)

    def _default_tts(self) -> TtsStream:
        """balanced/fast only (config validation keeps their model on Flash, the only websocket model)."""
        _, schedule = STREAMING_TIMINGS[self.config.timing]
        return TtsStream(self.config.api_key, self.config.voice_id, model_id=self.config.tts_model,
                         output_format=self.config.output_format, chunk_length_schedule=schedule,
                         inactivity_timeout=TTS_INACTIVITY_TIMEOUT_S, voice_settings=self._voice_settings_dict)

    # --- public API ---

    @property
    def speech_sent_s(self) -> float:
        """Seconds of the user's speech (local VAD) streamed so far: what the credit cap counts."""
        return self._speech_frames * FRAME_MS / 1000

    @property
    def tts_chars_sent(self) -> int:
        """Characters sent to text-to-speech so far (what TTS bills)."""
        return self._tts_chars

    def start(self) -> None:
        if self._thread is not None or self._stopped:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="stt-tts-loop", daemon=True)
        if self._smooth:
            self._speaker = threading.Thread(target=self._speaker_run, name="stt-tts-speaker", daemon=True)
            self._speaker.start()
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True
        self._halt.set()
        self._phrases.put_nowait(None)  # wakes an idle speaker
        self._call_soon(self._shutdown.set)
        current = threading.current_thread()
        if self._thread is not None and self._thread is not current:
            self._thread.join(timeout=STOP_JOIN_TIMEOUT_S)
        if self._speaker is not None and self._speaker is not current:
            self._speaker.join(timeout=SPEAKER_JOIN_TIMEOUT_S)

    def feed_audio(self, frame: bytes) -> None:
        if self._stopped or self._capped or self.finished.is_set():
            return
        was_in_speech = self._segmenter.in_speech
        self._segmenter.feed(frame)  # emitted phrases are not needed: only in_speech matters here
        started = not was_in_speech and self._segmenter.in_speech
        counted = 1 if was_in_speech else self._segmenter.start_frames if started else 0

        cap = self.config.max_speech_s
        if counted and cap > 0 and (self._speech_frames + counted) * FRAME_MS / 1000 > cap:
            self._capped = True  # this frame is not counted or forwarded
            self._emit(f"Speech cap reached ({cap:g} s). Use --max-speech to raise it.")
            if not self._call_soon(self._start_cap_drain):
                self.finished.set()  # loop not running: nothing in flight to wait for
            return
        self._speech_frames += counted

        if started and not self._smooth:  # smooth times from the commit instead and has no socket to warm
            # The start run's first frame began start_frames x 20 ms ago (frames arrive in real time).
            at = None
            if self._latency is not None:
                at = self._latency.now() - self._segmenter.start_frames * FRAME_MS / 1000
            self._call_soon(self._on_speech_start, at)
        self._call_soon(self._frames.put_nowait, frame)

    # --- thread plumbing ---

    def _call_soon(self, callback: Callable[..., Any], *args: Any) -> bool:
        """Run `callback` on the loop thread. Returns False (dropped) if the loop isn't running."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return False
        try:
            loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:  # closed between the check and the call
            return False
        return True

    def _run_loop(self) -> None:
        loop = self._loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except Exception as exc:  # noqa: BLE001 - never let the thread die silently
            logger.exception("stt-tts event loop failed")
            if self.error is None:
                self.error = ApiFatalError(f"Unexpected error ({type(exc).__name__}: {exc}).")
                self._emit(f"ERROR: {self.error}")
            self.finished.set()
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    async def _main(self) -> None:
        self._spawn(self._stt_loop, "speech-to-text")
        self._spawn(self._stt_sender, "speech-to-text audio sender")
        if not self._smooth:
            self._spawn(self._tts_sender, "text-to-speech sender")
        await self._shutdown.wait()
        await self._teardown()

    async def _teardown(self) -> None:
        self._halt.set()
        self._phrases.put_nowait(None)  # the speaker exits after (or instead of) its current phrase
        streams = [s for s in (self._stt, self._tts) if s is not None]
        tasks = [t for t in self._tasks if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=CLOSE_TIMEOUT_S)
        for stream in streams:
            try:
                await asyncio.wait_for(stream.close(), CLOSE_TIMEOUT_S)
            except Exception:  # noqa: BLE001 - shutting down; nothing useful to report
                pass

    def _spawn(self, fn: Callable[..., Coroutine[Any, Any, None]], what: str, *args: Any) -> None:
        task = asyncio.get_running_loop().create_task(self._guard(fn, what, *args))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _guard(self, fn: Callable[..., Coroutine[Any, Any, None]], what: str, *args: Any) -> None:
        """Run a task; anything escaping it stops the pipeline (tasks handle non-fatal errors themselves).

        The coroutine is created here, so a task cancelled before it starts leaves no unawaited coroutine.
        """
        try:
            await fn(*args)
        except asyncio.CancelledError:
            raise
        except RealtimeError as exc:
            self._fail(exc.message)
        except Exception as exc:  # noqa: BLE001 - a bug must not hang the pipeline
            logger.exception("%s task failed", what)
            self._fail(f"Unexpected error in {what} ({type(exc).__name__}: {exc}).")

    def _fail(self, msg: str) -> None:
        """Stop the pipeline with a fatal error. Runs on the loop thread or the speaker thread."""
        with self._fail_lock:
            first = self.error is None
            if first:
                self.error = ApiFatalError(msg)
        if first:
            self._emit(f"ERROR: {msg}")
        self._halt.set()
        self.finished.set()
        if threading.current_thread() is self._thread:
            self._shutdown.set()
        else:
            self._call_soon(self._shutdown.set)  # asyncio.Event isn't thread-safe

    # --- speech-to-text ---

    async def _stt_loop(self) -> None:
        loop = asyncio.get_running_loop()
        failures = 0
        while not self._shutdown.is_set():
            stt = self._stt_factory()
            try:
                await stt.connect()
            except RealtimeError as exc:
                if exc.fatal:
                    raise
                failures = await self._stt_backoff(exc.message, failures)
                continue
            self._stt = stt
            if failures:
                self._emit("speech-to-text reconnected")
            connected_at = loop.time()
            try:
                async for event in stt.events():
                    self._on_stt_event(event)
                if self._shutdown.is_set():
                    return
                reason = "Realtime speech-to-text stream ended."
            except RealtimeError as exc:
                if exc.fatal:
                    raise
                reason = exc.message
            finally:
                if self._stt is stt:
                    self._stt = None
            await stt.close()
            self._stt_lost()
            if loop.time() - connected_at >= STT_RECONNECT_RESET_S:
                failures = 0
            failures = await self._stt_backoff(reason, failures)

    async def _stt_backoff(self, reason: str, failures: int) -> int:
        delay = STT_RECONNECT_DELAYS_S[min(failures, len(STT_RECONNECT_DELAYS_S) - 1)]
        self._emit(f"WARNING: {reason} Reconnecting in {delay:g} s.")
        await asyncio.sleep(delay)
        return failures + 1

    def _stt_lost(self) -> None:
        """The STT socket dropped mid-stream: the segment is gone, but speak what was already sent
        (balanced/fast; smooth sends nothing before the commit)."""
        self._stabilizer.reset()
        self._segment_open = False
        self._segment_chars = 0
        if not self._smooth:
            self._say_flush()

    async def _stt_sender(self) -> None:
        buf = bytearray()
        while True:
            frame = await self._frames.get()
            stt = self._stt
            if stt is None or not stt.is_open:
                buf.clear()  # disconnected: drop audio rather than replay stale speech later
                continue
            buf += frame
            if len(buf) < SEND_CHUNK_BYTES:
                continue
            chunk = bytes(buf)
            buf.clear()
            try:
                await stt.send_audio(chunk)
            except RealtimeError as exc:
                if exc.fatal:
                    raise
                # The socket closed; the STT loop sees the same close and reconnects. This chunk is lost.

    def _on_stt_event(self, event: SttEvent) -> None:
        if event.kind == "partial":
            if event.text.strip():
                self._segment_open = True
            if not self._smooth:  # smooth never speaks partials, so on_committed returns the whole phrase
                self._say(self._stabilizer.on_partial(event.text))
        elif event.kind == "committed":
            self._segment_open = False
            if self._smooth:
                self._queue_phrase(event.text)
                return
            self._say(self._stabilizer.on_committed(event.text))
            self._say_flush()
            chars, self._segment_chars = self._segment_chars, 0
            if chars:
                self._emit(f'heard: "{_shown(event.text)}" ({chars} chars spoken)')
            elif event.text.strip():
                self._emit(f'ignored sound: no words ("{_shown(event.text)}")')
        else:  # "warning", or a non-fatal server "error" (fatal ones raise from events())
            self._emit(f"WARNING: {event.text}")

    # --- text-to-speech, smooth: one HTTP stream per phrase on the speaker thread ---

    def _queue_phrase(self, transcript: str) -> None:
        """Loop thread: a segment committed. Queue its words (fillers/events stripped) for the speaker."""
        text = self._stabilizer.on_committed(transcript).strip()
        if not text:
            if transcript.strip():
                self._emit(f'ignored sound: no words ("{_shown(transcript)}")')
            return
        # The speaker starts the timing with the commit time when it gets to the phrase,
        # so a phrase queued behind another one can't overwrite that one's pending timing.
        committed = self._latency.now() if self._latency is not None else None
        with self._phrases_lock:
            self._phrases_pending += 1
        self._phrases.put_nowait((text, committed))
        self._emit(f'heard: "{_shown(transcript)}" ({len(text)} chars spoken)')

    def _speaker_run(self) -> None:
        """Speaker thread: speak queued phrases one at a time, in order, until halted or a fatal error."""
        while not self._halt.is_set():
            item = self._phrases.get()
            if item is None:
                break
            text, committed = item
            try:
                self._speak_with_retry(text, committed)
            except Exception as exc:  # noqa: BLE001 - classified below; the thread must never die silently
                if not self._phrase_failed(text, exc):
                    break
            finally:
                with self._phrases_lock:
                    self._phrases_pending -= 1

    def _speak_with_retry(self, text: str, committed: float | None) -> None:
        """Speak one phrase; retry once after a temporary error if none of it had played yet."""
        for attempt in (1, 2):
            if self._halt.is_set():
                return
            try:
                self._speak_phrase(text, committed)
                return
            except API_ERRORS as exc:
                msg, fatal = describe_api_error(exc)
                if fatal or attempt == 2 or self._phrase_audio_started or self._halt.is_set():
                    raise
                self._emit(f"WARNING: {msg.replace(KEY_HINT, TTS_KEY_HINT)} Retrying the phrase.")
                if self._halt.wait(RETRY_DELAY_S):
                    return

    def _speak_phrase(self, text: str, committed: float | None) -> None:
        self._phrase_audio_started = False
        if self._latency is not None and committed is not None:
            # The user stopped about pause_s before the server committed the segment.
            self._latency.mark_speech_end(at=committed - self.config.pause_s)
            self._latency.mark("pause", at=committed)
            self._latency.mark("queue")
        # The SDK call is a lazy generator: the HTTP request goes out on the first next().
        text = with_delivery_tag(calm_punctuation(text), self.config)
        chunks = self._tts_client.text_to_speech.stream(
            self.config.voice_id,
            text=text,
            model_id=self.config.tts_model,
            output_format=self.config.output_format,
            voice_settings=self._voice_settings,
        )
        self._tts_chars += len(text)  # counted up front: once the request is sent it may be billed
        first = True
        try:
            for chunk in chunks:
                if self._halt.is_set():
                    break
                if not chunk:
                    continue
                if first and self._latency is not None:
                    self._latency.mark("voice")
                    self._latency.mark_first_audio_out()
                first = False
                self._phrase_audio_started = True
                self.on_audio_out(chunk)
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()  # closes the HTTP response if we broke out early

    def _phrase_failed(self, text: str, exc: Exception) -> bool:
        """Handle a failed phrase (called inside the except block). Returns False if the speaker must stop."""
        if self._halt.is_set():
            return False  # stopping anyway; nothing useful to report
        if isinstance(exc, API_ERRORS):
            msg, fatal = describe_api_error(exc)
            msg = msg.replace(KEY_HINT, TTS_KEY_HINT)  # KEY_HINT is about the STS permission
            if not fatal:
                self._emit(f'WARNING: {msg} - skipped "{_shown(text)}"')
                return True
        else:
            logger.exception("text-to-speech phrase failed")
            msg = f"Unexpected error in text-to-speech audio ({type(exc).__name__}: {exc})."
        self._fail(msg)
        return False

    # --- text-to-speech, balanced/fast: websocket ---

    def _say(self, text: str) -> None:
        text = calm_punctuation(text)  # Flash would read a delivery tag aloud, so websocket text gets no tag
        if text:
            self._segment_chars += len(text)
            self._tts_queue.put_nowait(("text", text))

    def _say_flush(self) -> None:
        self._tts_queue.put_nowait(("flush", ""))

    def _on_speech_start(self, at: float | None) -> None:
        """Loop thread: the local VAD heard speech begin. Start timing and warm up the TTS socket."""
        if self._latency is not None:
            self._latency.mark_speech_end(at=at)  # this tracker's "speech end" = speech START here
        self._latency_armed = False
        self._tts_queue.put_nowait(("connect", ""))

    async def _tts_sender(self) -> None:
        while True:
            kind, text = await self._tts_queue.get()
            try:
                await self._tts_handle(kind, text)
            finally:
                self._tts_queue.task_done()

    async def _tts_handle(self, kind: str, text: str) -> None:
        loop = asyncio.get_running_loop()
        if kind == "flush":
            if self._skip_segment:
                self._skip_segment = False
                return
            tts = self._tts
            if not self._unflushed or tts is None or not tts.is_open:
                return  # nothing buffered (or it died with its socket)
            self._unflushed = False
            try:
                await tts.flush()
            except RealtimeError as exc:
                if exc.fatal:
                    raise
                await self._drop_tts(tts)
                self._emit(f"WARNING: {exc.message} The last words may be missing.")
                return
            self._last_tts_activity = loop.time()
            return

        if kind == "connect":
            await self._ensure_tts(quiet=True)  # warm-up only; a failure is reported by the next text
            return
        if self._skip_segment:
            return
        for attempt in (1, 2):
            tts = await self._ensure_tts()
            if tts is None:
                return
            try:
                await tts.send_text(text)
            except RealtimeError as exc:
                if exc.fatal:
                    raise
                # Usually the socket closed (e.g. inactivity timeout) just before this send: the text
                # never left, so resend it on a new connection once.
                await self._drop_tts(tts)
                if attempt == 2:
                    self._emit(f"WARNING: {exc.message} Skipping the rest of this sentence.")
                    self._skip_segment = True
                continue
            self._tts_chars += len(text)
            self._unflushed = True
            self._latency_armed = True
            self._last_tts_activity = loop.time()
            return

    async def _ensure_tts(self, quiet: bool = False) -> Any:
        """The open TTS socket, connecting a new one if needed. None after a non-fatal connect failure."""
        tts = self._tts
        if tts is not None and tts.is_open:
            voice = getattr(tts, "voice_id", self.config.voice_id)
            idle = asyncio.get_running_loop().time() - self._last_tts_activity >= VOICE_SWITCH_IDLE_S
            if voice == self.config.voice_id or self._unflushed or not idle:
                return tts  # same voice, or mid-sentence: a switch waits for the next sentence
            await self._drop_tts(tts)  # set_voice() was called: reconnect with the new voice
        self._tts = None
        self._unflushed = False
        tts = self._tts_factory()
        try:
            await tts.connect()
        except RealtimeError as exc:
            if exc.fatal:
                raise
            if not quiet:
                self._emit(f"WARNING: {exc.message} Skipping the rest of this sentence.")
                self._skip_segment = True
            return None
        self._tts = tts
        self._spawn(self._tts_reader, "text-to-speech audio", tts)
        return tts

    async def _drop_tts(self, tts: Any) -> None:
        if self._tts is tts:
            self._tts = None
            self._unflushed = False
        try:
            await asyncio.wait_for(tts.close(), CLOSE_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - already broken
            pass

    async def _tts_reader(self, tts: Any) -> None:
        """Deliver audio for one TTS connection until its socket closes."""
        loop = asyncio.get_running_loop()
        try:
            while tts.is_open:  # audio() also ends on an isFinal while the socket stays open
                async for chunk in tts.audio():
                    if not chunk:
                        continue
                    self._last_tts_activity = loop.time()
                    if self._latency_armed and self._latency is not None:
                        self._latency.mark_first_audio_out()
                    self._latency_armed = False
                    self.on_audio_out(chunk)
        except RealtimeError as exc:
            if exc.fatal:
                raise
            self._emit(f"WARNING: {exc.message} The next words will reconnect.")
            await self._drop_tts(tts)
        finally:
            if self._tts is tts:
                self._tts = None
                self._unflushed = False

    # --- credit cap ---

    def _start_cap_drain(self) -> None:
        if not self._shutdown.is_set():
            self._spawn(self._drain_after_cap, "credit cap")

    async def _drain_after_cap(self) -> None:
        """Mic audio has stopped: let the last segment commit, speak it, then set `finished`."""
        loop = asyncio.get_running_loop()
        if self._segment_open and self._stt is not None:
            silence_ms = round(self.config.pause_s * 1000) + CAP_SILENCE_EXTRA_MS
            for _ in range(silence_ms // FRAME_MS):
                self._frames.put_nowait(SILENCE_FRAME)  # the server's VAD commits after this much silence
            deadline = loop.time() + CAP_COMMIT_TIMEOUT_S
            while self._segment_open and loop.time() < deadline:
                await asyncio.sleep(POLL_S)
        if self._smooth:
            # The commit (if it came) is already queued: wait for the speaker to finish everything.
            deadline = loop.time() + CAP_DRAIN_MAX_S
            while self._phrases_pending and not self._halt.is_set() and loop.time() < deadline:
                await asyncio.sleep(POLL_S)
            self.finished.set()
            return
        self._say_flush()  # speaks already-sent partial words if the commit never came
        await self._tts_queue.join()
        if self._tts_chars:
            deadline = loop.time() + CAP_DRAIN_MAX_S
            while (self._tts is not None and loop.time() < deadline
                   and loop.time() - self._last_tts_activity < CAP_AUDIO_IDLE_S):
                await asyncio.sleep(POLL_S)
        self.finished.set()
