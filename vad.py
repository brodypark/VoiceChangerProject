"""Voice activity detection and phrase segmentation (used by Pipeline B).

Input everywhere is 16 kHz mono int16 little-endian PCM bytes in 20 ms frames
(see config.py). The detector uses webrtcvad when it's installed, otherwise a
simple RMS-energy threshold.

PhraseSegmenter turns a frame stream into whole phrases:
    idle --(start_frames consecutive speech frames)--> in speech
    in speech --(end_silence_ms of consecutive non-speech)--> emit phrase, idle
Emitted phrase = pre-roll + voiced audio + a short tail pad. Most of the
trailing silence is trimmed (it's dead air we'd otherwise upload and pay for);
only `pre_roll_ms` of it is kept so word endings/decays aren't clipped.

It is also a free first-pass filter for non-word sounds (hm, uh, clicks,
breaths, sighs), so fewer of them reach the paid API: a finished phrase is
dropped if its voiced span is shorter than `min_phrase_ms`, or if its level is
under `min_snr` x the room's noise floor (tracked while idle).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from config import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, SAMPLE_WIDTH

VALID_SAMPLE_RATES = (8_000, 16_000, 32_000, 48_000)  # the only rates webrtcvad accepts
VALID_FRAME_MS = (10, 20, 30)  # the only frame durations webrtcvad accepts

# Noise floor (PhraseSegmenter): a slow average of the room's level while idle.
NOISE_FLOOR_MIN = 0.001  # clamp, so digital silence (all zeros) doesn't make every sound pass the SNR gate
NOISE_FLOOR_TC_MS = 2_000  # EMA time constant: slow, so a stray click barely moves it


def iter_frames(pcm: bytes, frame_bytes: int = FRAME_BYTES) -> tuple[list[bytes], bytes]:
    """Split `pcm` into whole frames. Returns (frames, remainder).

    Audio callbacks rarely deliver exact frame multiples, so the caller should
    prepend `remainder` to the next chunk.
    """
    if frame_bytes <= 0:
        raise ValueError(f"frame_bytes must be positive, got {frame_bytes}.")
    whole = len(pcm) - len(pcm) % frame_bytes
    frames = [pcm[i:i + frame_bytes] for i in range(0, whole, frame_bytes)]
    return frames, pcm[whole:]


def rms(frame: bytes) -> float:
    """Root-mean-square level of int16 PCM, normalized to 0.0..1.0. Empty -> 0.0."""
    # Ignore a dangling odd byte rather than failing on it.
    usable = len(frame) - len(frame) % SAMPLE_WIDTH
    if usable == 0:
        return 0.0
    samples = np.frombuffer(frame[:usable], dtype="<i2").astype(np.float64)
    return min(1.0, math.sqrt(float(np.mean(samples * samples))) / 32768.0)


class VoiceActivityDetector:
    """Per-frame speech/non-speech classifier.

    `backend` is "webrtc" if webrtcvad is in use, else "rms" (energy threshold).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = FRAME_MS,
        aggressiveness: int = 2,
        rms_threshold: float = 0.01,
        use_webrtc: bool = True,
    ) -> None:
        if sample_rate not in VALID_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {VALID_SAMPLE_RATES} Hz, got {sample_rate}.")
        if frame_ms not in VALID_FRAME_MS:
            raise ValueError(f"frame_ms must be one of {VALID_FRAME_MS}, got {frame_ms}.")
        if not 0 <= aggressiveness <= 3:
            raise ValueError(f"aggressiveness must be 0..3, got {aggressiveness}.")
        if not 0.0 <= rms_threshold <= 1.0:
            raise ValueError(f"rms_threshold must be 0.0..1.0, got {rms_threshold}.")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.aggressiveness = aggressiveness
        self.rms_threshold = rms_threshold
        self.frame_bytes = sample_rate * frame_ms // 1000 * SAMPLE_WIDTH

        self._webrtcvad = None  # the module, if available
        if use_webrtc:
            try:
                import webrtcvad  # optional; imported lazily so the module works without it
            except ImportError:
                pass
            else:
                self._webrtcvad = webrtcvad
        self.backend = "webrtc" if self._webrtcvad is not None else "rms"
        self._vad = None
        self.reset()

    def reset(self) -> None:
        """Clear internal state. webrtcvad smooths decisions over recent frames
        (a ~100-160 ms "hangover" after speech), so start fresh between streams."""
        if self._webrtcvad is not None:
            self._vad = self._webrtcvad.Vad(self.aggressiveness)

    def is_speech(self, frame: bytes) -> bool:
        """True if this single frame contains speech. Frame must be exactly `frame_bytes` long."""
        if len(frame) != self.frame_bytes:
            raise ValueError(
                f"Frame must be {self.frame_bytes} bytes ({self.frame_ms} ms at {self.sample_rate} Hz, "
                f"int16 mono), got {len(frame)}."
            )
        if self._vad is not None:
            return self._vad.is_speech(frame, self.sample_rate)
        return rms(frame) >= self.rms_threshold


class PhraseSegmenter:
    """Groups frames into complete phrases using a VoiceActivityDetector.

    Lengths are checked on the *voiced span* (first start frame .. last speech
    frame), not including pre-roll or tail pad, so `min_phrase_ms` really
    filters short blips. It's rounded up to whole frames (450 ms -> 23 x 20 ms
    = 460 ms). The default sits between non-words and short words: a quick
    "hm"/"uh"/click is ~100-300 ms of sound, while "yes"/"no" said normally is
    ~300-450 ms; webrtcvad's hangover then adds ~60-120 ms to the span, so the
    words land at ~400-550 ms and the blips mostly under 460. A clipped, very
    brisk monosyllable can still be dropped; 500 ms would drop many more.

    Quiet-sound gate: while idle, `noise_floor` follows the RMS level of
    non-speech frames (running mean for the first NOISE_FLOOR_TC_MS, then a
    slow EMA; never below NOISE_FLOOR_MIN; frozen during phrases). A finished
    phrase is dropped if the RMS level of its voiced span is below
    `noise_floor * min_snr`. Breaths and sighs at a mic are far quieter than
    speech (room ~0.01, speech ~0.05-0.2), which is what this catches.
    min_snr=3.0 is about +10 dB; 0 disables the gate.

    `max_phrase_ms` is checked on the total buffered audio; when reached, the
    buffer is force-emitted and the phrase continues in a new buffer
    (continuations are emitted if they contain any speech, regardless of
    `min_phrase_ms` and the quiet gate, so the end of a long monologue isn't
    dropped).

    `last_drop_reason` says why the most recent started phrase was discarded
    (e.g. "too short (0.32 s)" or "too quiet (level 0.018 < 0.036)"); it is None
    after a phrase is emitted, and set again on the next discard.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        start_frames: int = 4,
        end_silence_ms: int = 500,
        pre_roll_ms: int = 200,
        min_phrase_ms: int = 450,
        max_phrase_ms: int = 10_000,
        min_snr: float = 3.0,
    ) -> None:
        if start_frames < 1:
            raise ValueError(f"start_frames must be >= 1, got {start_frames}.")
        if end_silence_ms <= 0:
            raise ValueError(f"end_silence_ms must be positive, got {end_silence_ms}.")
        if pre_roll_ms < 0 or min_phrase_ms < 0:
            raise ValueError("pre_roll_ms and min_phrase_ms must be >= 0.")
        if max_phrase_ms <= min_phrase_ms:
            raise ValueError(f"max_phrase_ms ({max_phrase_ms}) must exceed min_phrase_ms ({min_phrase_ms}).")
        if not (math.isfinite(min_snr) and min_snr >= 0):
            raise ValueError(f"min_snr must be a finite number >= 0 (0 disables the gate), got {min_snr}.")

        self.vad = vad
        fm = vad.frame_ms
        self.start_frames = start_frames
        self.end_silence_frames = max(1, math.ceil(end_silence_ms / fm))
        self.pre_roll_frames = math.ceil(pre_roll_ms / fm)
        self.tail_frames = self.pre_roll_frames  # trailing silence kept after the last voiced frame
        self.min_frames = math.ceil(min_phrase_ms / fm)
        self.max_frames = max(1, max_phrase_ms // fm)
        self.min_snr = min_snr
        self._floor_alpha = fm / NOISE_FLOOR_TC_MS  # per-frame EMA weight
        self.last_drop_reason: str | None = None

        # While idle, this ring holds the pre-roll plus the current candidate run
        # of speech frames, so when speech is confirmed the first syllable
        # (and a little lead-in) is already captured.
        self._ring: deque[bytes] = deque(maxlen=self.pre_roll_frames + start_frames)
        self.reset()

    @property
    def in_speech(self) -> bool:
        """True while a phrase is being collected."""
        return self._in_speech

    @property
    def noise_floor(self) -> float:
        """Current room noise estimate (RMS, 0.0..1.0), never below NOISE_FLOOR_MIN."""
        return max(NOISE_FLOOR_MIN, self._noise_floor)

    def reset(self) -> None:
        """Fresh stream: drop buffered audio, the noise floor and last_drop_reason,
        reset the detector's state, and return to idle."""
        self.vad.reset()
        self._clear()
        self._noise_floor = NOISE_FLOOR_MIN
        self._floor_frames = 0  # non-speech frames averaged so far (for the warm-up mean)
        self.last_drop_reason = None

    def _clear(self) -> None:
        """Return to idle without touching detector state (used between phrases mid-stream)."""
        self._ring.clear()
        self._in_speech = False
        self._run = 0  # consecutive speech frames while idle
        self._buffer: list[bytes] = []
        self._voiced_start = 0  # index in _buffer of the first voiced frame
        self._voiced_end = 0  # index in _buffer just past the last voiced frame
        self._silence_run = 0  # consecutive non-speech frames while in speech
        self._continuation = False  # True after a max-length force-emit

    def feed(self, frame: bytes) -> bytes | None:
        """Process one frame. Returns a completed phrase (PCM bytes) or None."""
        speech = self.vad.is_speech(frame)  # raises ValueError on wrong frame length

        if not self._in_speech:
            self._ring.append(frame)
            self._run = self._run + 1 if speech else 0
            if not speech:
                self._update_noise_floor(frame)
            if self._run >= self.start_frames:
                # Speech confirmed: phrase begins with the pre-roll + the start run.
                self._buffer = list(self._ring)
                self._voiced_start = len(self._buffer) - self.start_frames
                self._voiced_end = len(self._buffer)
                self._ring.clear()
                self._in_speech = True
                self._silence_run = 0
                self._continuation = False
            return None

        self._buffer.append(frame)
        if speech:
            self._silence_run = 0
            self._voiced_end = len(self._buffer)
        else:
            self._silence_run += 1
            if self._silence_run >= self.end_silence_frames:
                return self._finish()

        if len(self._buffer) >= self.max_frames:
            # Force-emit so long monologues keep flowing; stay in speech.
            phrase = b"".join(self._buffer)
            self._buffer = []
            self._voiced_start = self._voiced_end = 0
            self._continuation = True
            self.last_drop_reason = None
            return phrase
        return None

    def _update_noise_floor(self, frame: bytes) -> None:
        """Fold one idle, non-speech frame into the noise floor. A running mean
        until NOISE_FLOOR_TC_MS of frames have been seen (so it settles quickly
        at startup), then an EMA with that time constant."""
        self._floor_frames += 1
        weight = max(self._floor_alpha, 1.0 / self._floor_frames)
        self._noise_floor += weight * (rms(frame) - self._noise_floor)

    def flush(self) -> bytes | None:
        """End of stream: emit any in-progress phrase that passes the length and
        level rules, then reset(). `last_drop_reason` survives the reset."""
        phrase = self._finish() if self._in_speech else None
        reason = self.last_drop_reason
        self.reset()
        self.last_drop_reason = reason
        return phrase

    def _finish(self) -> bytes | None:
        """End the current phrase: trim trailing silence, apply the min length
        and quiet gate (not to continuations), go idle."""
        voiced = self._voiced_end - self._voiced_start
        if self._continuation:
            # Rest of a force-emitted phrase: keep it if it has any speech. An empty
            # one is just trailing silence, so nothing was lost and no reason is set.
            reason = None
            keep = voiced > 0
        else:
            reason = self._drop_reason(voiced)
            keep = reason is None
        if keep:
            phrase = b"".join(self._buffer[:self._voiced_end + self.tail_frames])
            self.last_drop_reason = None
        else:
            phrase = None
            if reason is not None:
                self.last_drop_reason = reason
        self._clear()
        return phrase

    def _drop_reason(self, voiced: int) -> str | None:
        """Why a (non-continuation) phrase with `voiced` voiced frames should be dropped, or None to keep it."""
        if voiced < self.min_frames:
            return f"too short ({voiced * self.vad.frame_ms / 1000:.2f} s)"
        if self.min_snr > 0:
            # RMS over the whole voiced span (pre-roll and tail excluded): energy-weighted,
            # so short pauses between words don't drag real speech under the threshold.
            level = rms(b"".join(self._buffer[self._voiced_start:self._voiced_end]))
            threshold = self.noise_floor * self.min_snr
            if level < threshold:
                return f"too quiet (level {level:.3f} < {threshold:.3f})"
        return None
