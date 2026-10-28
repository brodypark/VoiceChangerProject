"""Audio I/O: mic capture, virtual-mic (BlackHole) playback, and headphone monitor.

Internal format everywhere: mono, signed 16-bit little-endian PCM bytes at the
sample rate the caller picks (48 kHz for passthrough, 16 kHz for the AI
pipelines; see config.py). This module is the only place audio is converted:

- Streams are opened at the requested rate; CoreAudio/PortAudio resamples to
  the device's native rate (BlackHole and Mac mics run at 48 kHz).
- Outputs are opened with min(2, device channels) and mono is duplicated into
  every channel. Writing mono to a stereo device would put audio in the left
  channel only, and many call apps read just one side.

Capture and playback are decoupled: MicCapture pushes fixed-size frames to a
callback, while AudioOutput keeps its own buffer that any thread (e.g. an
asyncio websocket handler) can write irregular chunks into.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Sequence

import numpy as np
import sounddevice as sd

import config
from devices import MIC_PERMISSION_HINT, get_devices
from vad import iter_frames

logger = logging.getLogger(__name__)

SAMPLE_WIDTH = config.SAMPLE_WIDTH  # bytes per int16 sample
DTYPE = "int16"


class AudioIOError(Exception):
    """Raised when an audio stream can't be opened/started, with a user-facing message."""


# --- Pure helpers (no hardware) ---

def mono_to_channels(pcm: bytes, channels: int) -> bytes:
    """Duplicate mono int16 samples into interleaved `channels`-channel int16."""
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}.")
    if len(pcm) % SAMPLE_WIDTH:
        raise ValueError(f"PCM length must be a multiple of {SAMPLE_WIDTH} bytes, got {len(pcm)}.")
    if channels == 1:
        return bytes(pcm)
    return np.repeat(np.frombuffer(pcm, dtype="<i2"), channels).tobytes()


def frame_samples_for(sample_rate: int, frame_ms: int = config.FRAME_MS) -> int:
    """Samples per frame, e.g. 960 for 20 ms at 48 kHz. The frame must be a whole number of samples."""
    if sample_rate <= 0 or frame_ms <= 0:
        raise ValueError(f"sample_rate and frame_ms must be positive, got {sample_rate}, {frame_ms}.")
    if (sample_rate * frame_ms) % 1000:
        raise ValueError(f"{frame_ms} ms at {sample_rate} Hz is not a whole number of samples.")
    return sample_rate * frame_ms // 1000


def ms_to_bytes(ms: float, sample_rate: int) -> int:
    """Byte length of `ms` of mono int16 audio, rounded down to a whole sample."""
    return int(sample_rate * ms / 1000) * SAMPLE_WIDTH


def bytes_to_ms(n_bytes: int, sample_rate: int) -> float:
    """Duration in ms of `n_bytes` of mono int16 audio."""
    return (n_bytes // SAMPLE_WIDTH) * 1000 / sample_rate


def looks_like_speakers(device_name: str) -> bool:
    """True if the device is probably speakers (monitoring through it would feed back into the mic)."""
    return "speaker" in device_name.lower()


def output_channels_for(device_index: int, devices: list[dict] | None = None) -> int:
    """Channels to open an output with: min(2, device's max output channels)."""
    if devices is None:
        devices = get_devices()
    for dev in devices:
        if dev["index"] == device_index:
            if dev["max_output_channels"] < 1:
                raise AudioIOError(f"Device [{device_index}] {dev['name']} has no output channels.")
            return min(2, dev["max_output_channels"])
    raise AudioIOError(f"No audio device with index {device_index}.")


def _open_error(kind: str, device: int | None, sample_rate: int, exc: Exception) -> AudioIOError:
    where = "the default device" if device is None else f"device [{device}]"
    message = f"Could not open {kind} stream on {where} at {sample_rate} Hz: {exc}"
    if kind == "input":
        message += f"\n{MIC_PERMISSION_HINT}"
    return AudioIOError(message)


# --- Capture ---

class MicCapture:
    """Captures mono int16 from a mic and calls `on_frame` with exactly one frame at a time.

    `on_frame` runs on the PortAudio callback thread: keep it fast and never block.
    Exceptions it raises are logged and counted (`errors`, `last_error`) instead
    of killing the stream. The stream is only opened by start().
    """

    def __init__(
        self,
        device: int | None,
        sample_rate: int,
        on_frame: Callable[[bytes], None],
        frame_ms: int = config.FRAME_MS,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.on_frame = on_frame
        self.frame_samples = frame_samples_for(sample_rate, frame_ms)
        self.frame_bytes = self.frame_samples * SAMPLE_WIDTH
        self.overflows = 0  # PortAudio reported dropped input (we were too slow)
        self.errors = 0
        self.last_error: BaseException | None = None
        self._pending = b""  # partial frame left over when a block isn't a frame multiple
        self._stream: sd.RawInputStream | None = None

    @property
    def running(self) -> bool:
        return self._stream is not None

    def _handle_block(self, data: bytes) -> None:
        """Split a captured block into frames and deliver them. Called from the audio thread."""
        if not self._pending and len(data) == self.frame_bytes:
            frames = [data]  # the usual case: blocksize == frame size
        else:
            frames, self._pending = iter_frames(self._pending + data, self.frame_bytes)
        for frame in frames:
            try:
                self.on_frame(frame)
            except Exception as exc:  # noqa: BLE001 - must not propagate into PortAudio
                self.errors += 1
                self.last_error = exc
                if self.errors == 1:
                    logger.exception("on_frame raised; the stream keeps running (further errors logged at debug)")
                else:
                    logger.debug("on_frame raised again (%d total): %r", self.errors, exc)

    def _callback(self, indata, frames: int, time, status: sd.CallbackFlags) -> None:  # noqa: ANN001
        if status.input_overflow:
            self.overflows += 1
        self._handle_block(bytes(indata))

    def start(self) -> None:
        """Open and start the input stream. No-op if already running."""
        if self._stream is not None:
            return
        self._pending = b""
        try:
            stream = sd.RawInputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=1,
                dtype=DTYPE,
                blocksize=self.frame_samples,
                latency="low",
                callback=self._callback,
            )
        except (sd.PortAudioError, ValueError) as exc:
            raise _open_error("input", self.device, self.sample_rate, exc) from exc
        try:
            stream.start()
        except sd.PortAudioError as exc:
            stream.close()
            raise _open_error("input", self.device, self.sample_rate, exc) from exc
        self._stream = stream

    def stop(self) -> None:
        """Stop and close the stream. Safe to call when not running."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()

    def __enter__(self) -> MicCapture:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


# --- Playback ---

class AudioOutput:
    """Plays mono int16 PCM, written from any thread, to one output device.

    Buffering:
    - write() appends to a buffer. If it grows past `max_buffer_ms`, the OLDEST
      audio is dropped (counted in `dropped_bytes`/`dropped_ms`) so latency stays bounded.
    - Playback (re)starts only once `prefill_ms` is buffered, so a cold start or
      a resume after running dry doesn't crackle. If a short tail (< prefill) sits
      in the buffer for `prefill_ms` without more arriving, it is played anyway.
    - `underruns` counts each time the buffer ran dry *while playing*. That
      includes the natural end of each burst of AI audio; in continuous passthrough,
      a growing count means glitches. Sitting idle with an empty buffer is not counted.

    Odd-length writes are fine: only whole samples are ever played or dropped.
    The stream is only opened by start().
    """

    def __init__(
        self,
        device: int,
        sample_rate: int,
        channels: int = 2,
        max_buffer_ms: int = 200,
        prefill_ms: int = 20,
        name: str = "",
    ) -> None:
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}.")
        if prefill_ms < 0 or max_buffer_ms <= prefill_ms:
            raise ValueError(f"Need 0 <= prefill_ms ({prefill_ms}) < max_buffer_ms ({max_buffer_ms}).")
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.name = name or f"device {device}"
        self.max_buffer_bytes = ms_to_bytes(max_buffer_ms, sample_rate)
        self.prefill_bytes = ms_to_bytes(prefill_ms, sample_rate)
        self.underruns = 0
        self.dropped_bytes = 0
        self.output_underflows = 0  # PortAudio reported a device-level underflow
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._playing = False  # False = waiting for prefill
        self._waited_frames = 0  # frames spent waiting with a partial (sub-prefill) buffer
        self._stream: sd.RawOutputStream | None = None

    @property
    def running(self) -> bool:
        return self._stream is not None

    @property
    def buffered_ms(self) -> float:
        with self._lock:
            return bytes_to_ms(len(self._buffer), self.sample_rate)

    @property
    def dropped_ms(self) -> float:
        return bytes_to_ms(self.dropped_bytes, self.sample_rate)

    def write(self, pcm: bytes) -> None:
        """Queue mono int16 PCM for playback. Thread-safe and non-blocking."""
        if not pcm:
            return
        with self._lock:
            self._buffer += pcm
            excess = len(self._buffer) - self.max_buffer_bytes
            if excess > 0:
                excess += excess % SAMPLE_WIDTH  # drop whole samples to keep alignment
                del self._buffer[:excess]
                self.dropped_bytes += excess

    def clear(self) -> None:
        """Drop all queued audio (e.g. on barge-in or mode switch) and wait for prefill again."""
        with self._lock:
            self._buffer.clear()
            self._playing = False
            self._waited_frames = 0

    def _fill(self, n_frames: int) -> bytes:
        """Return exactly n_frames * channels * 2 bytes of interleaved int16 for the device."""
        want = n_frames * SAMPLE_WIDTH
        with self._lock:
            available = len(self._buffer) - len(self._buffer) % SAMPLE_WIDTH
            if not self._playing:
                if available == 0:
                    self._waited_frames = 0
                elif available >= self.prefill_bytes or self._waited_frames * SAMPLE_WIDTH >= self.prefill_bytes:
                    self._playing = True
                    self._waited_frames = 0
                else:
                    self._waited_frames += n_frames

            if self._playing:
                take = min(want, available)
                chunk = bytes(self._buffer[:take])
                del self._buffer[:take]
                if take < want:
                    self.underruns += 1
                    self._playing = False
            else:
                chunk = b""

        mono = chunk + bytes(want - len(chunk))  # pad with silence
        return mono_to_channels(mono, self.channels)

    def _callback(self, outdata, frames: int, time, status: sd.CallbackFlags) -> None:  # noqa: ANN001
        if status.output_underflow:
            self.output_underflows += 1
        outdata[:] = self._fill(frames)

    def start(self) -> None:
        """Open and start the output stream. No-op if already running."""
        if self._stream is not None:
            return
        try:
            stream = sd.RawOutputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=DTYPE,
                latency="low",
                callback=self._callback,
            )
        except (sd.PortAudioError, ValueError) as exc:
            raise _open_error("output", self.device, self.sample_rate, exc) from exc
        try:
            stream.start()
        except sd.PortAudioError as exc:
            stream.close()
            raise _open_error("output", self.device, self.sample_rate, exc) from exc
        self._stream = stream

    def stop(self) -> None:
        """Stop and close the stream (queued audio is kept; call clear() to drop it)."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()

    def __enter__(self) -> AudioOutput:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


class OutputGroup:
    """Fans each write out to several outputs (BlackHole + optional headphone monitor)."""

    def __init__(self, outputs: Sequence[AudioOutput]) -> None:
        self.outputs = list(outputs)

    def write(self, pcm: bytes) -> None:
        for out in self.outputs:
            out.write(pcm)

    def clear(self) -> None:
        for out in self.outputs:
            out.clear()

    def start(self) -> None:
        """Start every output; if one fails, stop the ones already started and re-raise."""
        started: list[AudioOutput] = []
        try:
            for out in self.outputs:
                out.start()
                started.append(out)
        except BaseException:
            for out in started:
                try:
                    out.stop()
                except Exception:  # noqa: BLE001 - the original error matters more
                    logger.exception("Error stopping %s during failed start", out.name)
            raise

    def stop(self) -> None:
        """Stop every output, even if some fail; re-raise the first failure afterwards."""
        first_error: Exception | None = None
        for out in self.outputs:
            try:
                out.stop()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error stopping %s", out.name)
                first_error = first_error or exc
        if first_error is not None:
            raise AudioIOError(f"Error while stopping audio output: {first_error}") from first_error

    def __enter__(self) -> OutputGroup:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
