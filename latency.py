"""Latency tracking: end of speech -> first audio chunk out, per utterance.

Thread-safe: called from both the sounddevice callback thread (first audio out)
and the asyncio loop (speech end, intermediate checkpoints). All state is
guarded by one threading.Lock; the on_utterance callback runs outside it.

Typical flow per utterance:
    tracker.mark_speech_end()        # VAD / STT says the speaker stopped
    tracker.mark("pause")            # optional checkpoints (ms offset from speech end); each one ENDS
    tracker.mark("voice")            #   the stage named after it, so stages() gives a breakdown
    tracker.mark_first_audio_out()   # first chunk written to the output device

Percentiles use linear interpolation between closest ranks (same as
numpy.percentile's default and statistics.quantiles(method="inclusive")).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_HISTORY = 1000  # timings kept for summary(); older ones are dropped


@dataclass(frozen=True)
class UtteranceTiming:
    utterance_id: int
    speech_end: float  # clock seconds
    first_audio_out: float  # clock seconds
    marks: dict[str, float] = field(default_factory=dict)  # name -> ms after speech_end

    @property
    def latency_ms(self) -> float:
        return (self.first_audio_out - self.speech_end) * 1000.0

    def stages(self) -> list[tuple[str, float]]:
        """Marks as consecutive stage durations in ms: each mark ends the stage named after it.

        Time after the last mark (if >= 1 ms) is reported as "other". No marks -> [].
        """
        out: list[tuple[str, float]] = []
        prev = 0.0
        for name, ms in self.marks.items():
            out.append((name, ms - prev))
            prev = ms
        rest = self.latency_ms - prev
        if out and rest >= 1.0:
            out.append(("other", rest))
        return out

    def __str__(self) -> str:
        extra = "".join(f", {name}={ms:.0f}ms" for name, ms in self.marks.items())
        return f"utterance #{self.utterance_id}: {self.latency_ms:.0f} ms{extra}"


@dataclass(frozen=True)
class LatencySummary:
    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    def __str__(self) -> str:
        if self.count == 0:
            return "latency: no utterances yet"
        return (f"latency over {self.count} utterance(s): mean {self.mean_ms:.0f} ms, "
                f"p50 {self.p50_ms:.0f} ms, p95 {self.p95_ms:.0f} ms, "
                f"min {self.min_ms:.0f} ms, max {self.max_ms:.0f} ms")


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile (0-100) of an already sorted, non-empty sequence."""
    rank = (len(sorted_values) - 1) * pct / 100.0
    lo, hi = math.floor(rank), math.ceil(rank)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (rank - lo)


def summarize(latencies_ms: Sequence[float]) -> LatencySummary:
    """Build a LatencySummary from raw latencies in ms (empty -> all zeros)."""
    if not latencies_ms:
        return LatencySummary()
    values = sorted(latencies_ms)
    return LatencySummary(
        count=len(values),
        mean_ms=sum(values) / len(values),
        p50_ms=percentile(values, 50),
        p95_ms=percentile(values, 95),
        min_ms=values[0],
        max_ms=values[-1],
    )


@dataclass
class _Pending:
    utterance_id: int
    speech_end: float
    marks: dict[str, float]


class LatencyTracker:
    """Records end-of-speech -> first-audio-out latency for each utterance."""

    def __init__(
        self,
        clock: Callable[[], float] = time.perf_counter,
        on_utterance: Callable[[UtteranceTiming], None] | None = None,
        history: int = DEFAULT_HISTORY,
    ) -> None:
        if history < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        self._clock = clock
        self._on_utterance = on_utterance
        self._lock = threading.Lock()
        self._timings: deque[UtteranceTiming] = deque(maxlen=history)
        self._pending: _Pending | None = None
        self._next_id = 1

    def mark_speech_end(self, at: float | None = None) -> int:
        """Start a new utterance and return its id.

        `at` is the speech-end time on this tracker's clock, for callers that
        detect the end on one thread and start timing later on another
        (e.g. a queued phrase); defaults to now.
        If the previous utterance never produced audio, it is discarded and
        replaced (e.g. the pipeline dropped an empty/noisy phrase).
        """
        now = self._clock() if at is None else at
        with self._lock:
            utterance_id = self._next_id
            self._next_id += 1
            self._pending = _Pending(utterance_id, now, {})
        return utterance_id

    def now(self) -> float:
        """Current time on this tracker's clock (for passing to mark_speech_end(at=...) later)."""
        return self._clock()

    def mark(self, name: str, at: float | None = None) -> None:
        """Record a named checkpoint (ms after speech end) for the pending utterance.

        `at` is the checkpoint time on this tracker's clock (default: now).
        No-op if nothing is pending. Re-marking the same name overwrites it.
        """
        now = self._clock() if at is None else at
        with self._lock:
            if self._pending is not None:
                self._pending.marks[name] = (now - self._pending.speech_end) * 1000.0

    def mark_first_audio_out(self) -> UtteranceTiming | None:
        """Complete the pending utterance on its first audio chunk.

        Returns the timing, or None if nothing is pending (later chunks of the
        same utterance, or audio with no preceding mark_speech_end). Cheap
        enough to call on every output chunk from the audio callback.
        """
        now = self._clock()
        with self._lock:
            pending, self._pending = self._pending, None
            if pending is None:
                return None
            timing = UtteranceTiming(pending.utterance_id, pending.speech_end, now, pending.marks)
            self._timings.append(timing)

        # Outside the lock so a slow callback can't block other threads' marks.
        # Exceptions are logged, not raised: this may run in the audio callback,
        # where an exception would abort the stream.
        if self._on_utterance is not None:
            try:
                self._on_utterance(timing)
            except Exception:
                logger.exception("on_utterance callback failed")
        return timing

    @property
    def timings(self) -> list[UtteranceTiming]:
        """Completed timings, oldest first (a copy)."""
        with self._lock:
            return list(self._timings)

    def summary(self) -> LatencySummary:
        """Stats over the kept history of completed utterances."""
        with self._lock:
            latencies = [t.latency_ms for t in self._timings]
        return summarize(latencies)

    def reset(self) -> None:
        """Clear history and any pending utterance. Ids keep counting up."""
        with self._lock:
            self._timings.clear()
            self._pending = None
