"""One voice-changer run: devices + audio streams + (optional) AI pipeline, without a UI loop.

Shared by the CLI (main.py) and the GUI (gui/), so neither duplicates setup or pipeline logic.

Lifecycle (all calls from one controlling thread; the GUI uses a worker thread for the slow ones):
    session = VoiceSession(config)
    lines = session.prepare()   # credentials + free account check; no audio opened. May raise.
    info = session.start()      # resolve devices, open outputs + mic, start the pipeline. May raise;
                                # on failure everything it opened is closed again.
    while ...:
        session.poll()          # pipeline event lines + a one-time "mic is silent" hint
        session.meter.level     # live input level (0-1 RMS)
        session.done            # the pipeline stopped itself (cap/fatal error) and its audio finished
        session.set_voice(id)   # switch voices live; the next phrase uses it
    lines = session.stop()      # close everything; returns the final summary lines

Latency: each utterance becomes an event line with its stage breakdown, e.g.
"latency: 1240 ms (end of speech -> first audio out: pause 500 + voice 740)", and, with
config.latency_log, a CSV row. Rows are written by poll()/stop() on the controlling thread,
never by the pipeline threads that complete the timings.

Errors raised by prepare()/start() are ConfigError, DeviceError, AudioIOError or ApiFatalError,
all with user-facing messages.
"""

from __future__ import annotations

import csv
import dataclasses
import queue
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from audio_io import AudioOutput, MicCapture, OutputGroup, looks_like_speakers, output_channels_for
from config import DEFAULT_OUTPUT_DEVICE, PASSTHROUGH_SAMPLE_RATE, SAMPLE_RATE, Config, ConfigError, require_credentials
from devices import DeviceError, find_blackhole, find_device
from devices import get_devices as default_get_devices
from latency import LatencyTracker, UtteranceTiming
from pipelines.base import Pipeline
from pipelines.base import check_account as default_check_account
from pipelines.base import make_client as default_make_client
from vad import rms

METER_WIDTH = 30
SILENCE_HINT_AFTER_S = 3.0  # all-zero input this long usually means macOS blocked the mic
PASSTHROUGH_BUFFER_MS = 200  # live mic audio: drop the oldest rather than drift behind
AI_OUTPUT_BUFFER_MS = 60_000  # AI audio streams in faster than real time, so never drop it
MIN_SHOWN_STAGE_MS = 10  # stages shorter than this (e.g. an empty queue) are left out of the latency line
LOG_STAGES = ("pause", "queue", "voice", "word check", "other")  # CSV columns, in pipeline order
LOG_FIELDS = ("time", "mode", "voice_id", "latency_ms", *(f"{s.replace(' ', '_')}_ms" for s in LOG_STAGES))
SILENCE_HINT = ("No sound from the mic at all. macOS may be blocking it: System Settings > "
                "Privacy & Security > Microphone, enable your terminal app, then restart it.")

PipelineFactory = Callable[[Callable[[bytes], None], LatencyTracker], Pipeline]


class LevelMeter:
    """Latest input level, written by the audio callback thread and read by the main thread."""

    def __init__(self) -> None:
        self.level = 0.0  # float assignment is atomic in CPython, so no lock needed
        self.peak = 0.0  # max level seen so far

    def update(self, frame: bytes) -> None:
        self.level = rms(frame)
        self.peak = max(self.peak, self.level)

    def bar(self) -> str:
        # RMS of normal speech is ~0.01-0.1, so scale x5 (capped) to make the bar readable.
        filled = min(METER_WIDTH, int(self.level * 5 * METER_WIDTH))
        return "#" * filled + "-" * (METER_WIDTH - filled)


def resolve_devices(config: Config, devices: list[dict]) -> tuple[int, int, int | None]:
    """Return (input, output, monitor) device indices, refusing setups that would feed back."""
    input_idx = find_device(config.input_device, "input", devices)
    if config.output_device == DEFAULT_OUTPUT_DEVICE:
        output_idx = find_blackhole(devices)  # error message includes install instructions
    else:
        output_idx = find_device(config.output_device, "output", devices)
    monitor_idx = find_device(config.monitor_device, "output", devices) if config.monitor_device else None

    names = {d["index"]: d["name"] for d in devices}
    if "blackhole" in names[input_idx].lower():
        raise DeviceError("Input is BlackHole, which would loop the output back into itself. "
                          "Pick your real mic with --input.")
    if monitor_idx is not None and monitor_idx == output_idx:
        raise DeviceError("--monitor is the same device as the output. Pick your headphones.")
    return input_idx, output_idx, monitor_idx


def build_outputs(output_idx: int, monitor_idx: int | None, rate: int, devices: list[dict],
                  max_buffer_ms: int = PASSTHROUGH_BUFFER_MS) -> tuple[OutputGroup, list[str]]:
    """BlackHole output plus the optional headphone monitor, all at `rate`. Returns (group, warnings)."""
    names = {d["index"]: d["name"] for d in devices}
    warnings: list[str] = []
    outputs = [AudioOutput(output_idx, rate, output_channels_for(output_idx, devices),
                           max_buffer_ms=max_buffer_ms, name="output")]
    if monitor_idx is not None:
        if looks_like_speakers(names[monitor_idx]):
            warnings.append(f"WARNING: monitoring through '{names[monitor_idx]}' will feed back into the mic. "
                            "Use headphones.")
        outputs.append(AudioOutput(monitor_idx, rate, output_channels_for(monitor_idx, devices),
                                   max_buffer_ms=max_buffer_ms, name="monitor"))
    return OutputGroup(outputs), warnings


def describe_route(input_idx: int, output_idx: int, monitor_idx: int | None, devices: list[dict]) -> str:
    names = {d["index"]: d["name"] for d in devices}
    route = f"[{input_idx}] {names[input_idx]} -> [{output_idx}] {names[output_idx]}"
    if monitor_idx is not None:
        route += f" + monitor [{monitor_idx}] {names[monitor_idx]}"
    return route


def latency_line(timing: UtteranceTiming, label: str) -> str:
    """"latency: 1240 ms (end of speech -> first audio out: pause 500 + voice 740)"."""
    stages = " + ".join(f"{name} {ms:.0f}" for name, ms in timing.stages() if ms >= MIN_SHOWN_STAGE_MS)
    detail = f"{label}: {stages}" if stages else label
    return f"latency: {timing.latency_ms:.0f} ms ({detail})"


def credit_cap_line(config: Config) -> str:
    if config.max_speech_s > 0:
        return (f"Credit cap: stops after {config.max_speech_s:.0f} s of speech is sent "
                "(--max-speech to change, 0 = no cap).")
    return "WARNING: no speech cap. Everything you say uses ElevenLabs credits."


@dataclass(frozen=True)
class StartInfo:
    route: str  # "[0] Mic -> [2] BlackHole 2ch (+ monitor ...)"
    output_name: str  # what to pick as the microphone in Discord/Zoom
    warnings: list[str] = field(default_factory=list)


class VoiceSession:
    """Owns one run. `get_devices`, `make_client`, `check_account` and `clock` are injectable for tests.

    Not restartable: create a new session for each Start.
    """

    def __init__(
        self,
        config: Config,
        *,
        get_devices: Callable[[], list[dict]] = default_get_devices,
        make_client: Callable[[str], object] = default_make_client,
        check_account: Callable[[object, str], list[str]] = default_check_account,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._get_devices = get_devices
        self._make_client = make_client
        self._check_account = check_account
        self._clock = clock

        self.meter = LevelMeter()
        self.latency = LatencyTracker(on_utterance=self._on_utterance)
        self.pipeline: Pipeline | None = None
        self.title = ""
        self.how_to = ""
        self.latency_label = ""
        self._make_pipeline: PipelineFactory | None = None
        self._prepared = False
        self._group: OutputGroup | None = None
        self._mic: MicCapture | None = None
        self._started_at = 0.0
        self._hinted = False
        self._to_log: queue.Queue[tuple[UtteranceTiming, str, str]] = queue.Queue()  # (timing, mode, voice)
        self._log_failed = False

    # --- setup ---

    def prepare(self) -> list[str]:
        """Check credentials/account and pick the pipeline. Returns startup info lines. No audio is opened."""
        config = self.config
        lines: list[str] = []
        if config.mode == "passthrough":
            self.title = f"Passthrough @ {PASSTHROUGH_SAMPLE_RATE // 1000} kHz"
        elif config.mode == "sts":
            lines = self._prepare_sts()
        else:
            lines = self._prepare_stt_tts()
        self._prepared = True
        return lines

    def _prepare_sts(self) -> list[str]:
        from pipelines.sts import StsPipeline

        config = self.config
        client = None
        if config.dry_run:
            lines = ["DRY RUN: no ElevenLabs calls, no credits. Each detected phrase is echoed back unchanged."]
        else:
            require_credentials(config)
            client = self._make_client(config.api_key)
            lines = list(self._check_account(client, config.voice_id))  # free; raises ApiFatalError on a bad key
            lines.append(credit_cap_line(config))
            if config.word_check:
                lines.append("Word check: on. Sounds with no real words (hmm, uh, sighs) are dropped "
                             "(--no-word-check to disable).")

        def make(on_audio_out: Callable[[bytes], None], latency: LatencyTracker) -> Pipeline:
            return StsPipeline(config, on_audio_out, client=client, latency=latency)

        self._make_pipeline = make
        fmt = "dry run" if config.dry_run else config.output_format
        self.title = f"Speech-to-speech ({fmt})"
        self.how_to = "Speak a sentence, then pause."
        self.latency_label = "end of speech -> first audio out"
        return lines

    def _prepare_stt_tts(self) -> list[str]:
        config = self.config
        if config.dry_run:
            raise ConfigError("--dry-run isn't available for stt-tts (it needs the realtime APIs). "
                              "Use --mode sts --dry-run to test your mic and speech detection for free.")
        import pipelines.stt_tts  # imported here so other modes don't need websockets

        require_credentials(config)
        lines = list(self._check_account(self._make_client(config.api_key), config.voice_id))  # free check
        lines.append(credit_cap_line(config))

        def make(on_audio_out: Callable[[bytes], None], latency: LatencyTracker) -> Pipeline:
            return pipelines.stt_tts.SttTtsPipeline(config, on_audio_out, latency=latency)

        self._make_pipeline = make
        self.title = (f"Speech-to-text -> text-to-speech ({config.tts_model}, {config.timing} timing, "
                      f"{config.output_format})")
        if config.timing == "smooth":
            self.how_to = "Speak a sentence, then pause; it's read out in your voice."
            self.latency_label = "end of speech -> first audio out"
        else:
            self.how_to = "Just talk; your voice starts playing after a few words."
            self.latency_label = "start of speech -> first audio out"
        return lines

    def start(self) -> StartInfo:
        """Open outputs and mic and start the pipeline. On failure, closes whatever it opened and re-raises."""
        if not self._prepared:
            raise RuntimeError("call prepare() before start()")
        devices = self._get_devices()
        input_idx, output_idx, monitor_idx = resolve_devices(self.config, devices)

        if self._make_pipeline is None:  # passthrough
            rate, buffer_ms = PASSTHROUGH_SAMPLE_RATE, PASSTHROUGH_BUFFER_MS
            on_frame = self._passthrough_frame
        else:
            self.pipeline = self._make_pipeline(self._write_out, self.latency)
            rate, buffer_ms = self.pipeline.output_sample_rate, AI_OUTPUT_BUFFER_MS
            on_frame = self._pipeline_frame
        group, warnings = build_outputs(output_idx, monitor_idx, rate, devices, max_buffer_ms=buffer_ms)
        mic = MicCapture(input_idx, PASSTHROUGH_SAMPLE_RATE if self.pipeline is None else SAMPLE_RATE, on_frame)

        group.start()
        try:
            mic.start()
        except BaseException:
            group.stop()
            raise
        self._group, self._mic = group, mic
        if self.pipeline is not None:
            self.pipeline.start()
        self._started_at = self._clock()

        names = {d["index"]: d["name"] for d in devices}
        return StartInfo(describe_route(input_idx, output_idx, monitor_idx, devices), names[output_idx], warnings)

    # --- audio callbacks (PortAudio / pipeline threads: keep them short) ---

    def _passthrough_frame(self, frame: bytes) -> None:
        self._group.write(frame)
        self.meter.update(frame)

    def _pipeline_frame(self, frame: bytes) -> None:
        self.pipeline.feed_audio(frame)  # VAD + enqueue only, never network calls
        self.meter.update(frame)

    def _write_out(self, pcm: bytes) -> None:
        self._group.write(pcm)  # set in start() before the pipeline (or mic) can produce audio

    def _on_utterance(self, timing: UtteranceTiming) -> None:
        # Runs on a pipeline thread: only queue things up.
        if self.pipeline is not None:
            self.pipeline.events.put(latency_line(timing, self.latency_label))
        if self.config.latency_log:
            self._to_log.put((timing, self.config.mode, self.config.voice_id or ""))

    def _write_latency_log(self) -> list[str]:
        """Append queued timings to the CSV (header on a new file). Returns a warning line on failure."""
        rows = []
        while True:
            try:
                timing, mode, voice = self._to_log.get_nowait()
            except queue.Empty:
                break
            stages = dict(timing.stages())
            rows.append([datetime.now().isoformat(timespec="seconds"), mode, voice, f"{timing.latency_ms:.0f}",
                         *(f"{stages[s]:.0f}" if s in stages else "" for s in LOG_STAGES)])
        if not rows or self._log_failed:
            return []
        path = Path(self.config.latency_log).expanduser()
        try:
            new = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(LOG_FIELDS)
                writer.writerows(rows)
        except OSError as exc:
            self._log_failed = True
            return [f"WARNING: can't write the latency log ({exc}); logging is off for this run."]
        return []

    def set_voice(self, voice_id: str) -> None:
        """Switch to another voice while running (or before start); the next phrase uses it."""
        self.config = dataclasses.replace(self.config, voice_id=voice_id)
        if self.pipeline is not None:
            self.pipeline.set_voice(voice_id)

    # --- while running ---

    @property
    def running(self) -> bool:
        return self._mic is not None

    def poll(self) -> list[str]:
        """New pipeline event lines, plus a one-time hint if the mic has been completely silent."""
        lines = self.pipeline.drain_events() if self.pipeline is not None else []
        lines += self._write_latency_log()
        if (not self._hinted and self.running and self.meter.peak == 0.0
                and self._clock() - self._started_at > SILENCE_HINT_AFTER_S):
            self._hinted = True
            lines.append(SILENCE_HINT)
        return lines

    @property
    def done(self) -> bool:
        """The pipeline stopped itself (cap reached / fatal error) and its last audio finished playing."""
        return (self.pipeline is not None and self.pipeline.finished.is_set()
                and self._group is not None and self._group.outputs[0].buffered_ms == 0)

    @property
    def error(self) -> Exception | None:
        return self.pipeline.error if self.pipeline is not None else None

    @property
    def speech_sent_s(self) -> float:
        return getattr(self.pipeline, "speech_sent_s", 0.0) if self.pipeline is not None else 0.0

    def stop(self) -> list[str]:
        """Close the mic, the pipeline and the outputs. Returns final lines ([] if start() never succeeded).

        Safe to call more than once.
        """
        mic, group = self._mic, self._group
        if mic is None:
            return []
        self._mic = None
        try:
            mic.stop()
            if self.pipeline is not None:
                self.pipeline.stop()
        finally:
            if group is not None:
                group.stop()

        if self.pipeline is None:
            return ["Stopped."]
        lines = self.pipeline.drain_events()  # a fatal error was already emitted as an "ERROR: ..." line
        lines += self._write_latency_log()
        if self.config.latency_log and not self._log_failed:
            lines.append(f"Latency log: {self.config.latency_log}")
        lines.append(f"Stopped. {self.latency.summary()}")
        if not self.config.dry_run:
            lines.append(f"Speech sent this run: {self.speech_sent_s:.1f} s")
        return lines
