from __future__ import annotations

import logging

import pytest

from latency import LatencySummary, LatencyTracker, percentile, summarize


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms / 1000.0


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def utterance(tracker: LatencyTracker, clock: FakeClock, latency_ms: float):
    tracker.mark_speech_end()
    clock.advance(latency_ms)
    return tracker.mark_first_audio_out()


def test_basic_latency(clock):
    tracker = LatencyTracker(clock=clock)
    assert tracker.mark_speech_end() == 1
    clock.advance(250)
    timing = tracker.mark_first_audio_out()
    assert timing.utterance_id == 1
    assert timing.latency_ms == pytest.approx(250)
    assert str(timing) == "utterance #1: 250 ms"
    assert tracker.timings == [timing]


def test_only_first_audio_counts(clock):
    tracker = LatencyTracker(clock=clock)
    utterance(tracker, clock, 100)
    clock.advance(50)
    assert tracker.mark_first_audio_out() is None
    assert len(tracker.timings) == 1


def test_audio_without_speech_end_is_ignored(clock):
    assert LatencyTracker(clock=clock).mark_first_audio_out() is None


def test_marks_are_ms_offsets(clock):
    tracker = LatencyTracker(clock=clock)
    tracker.mark("ignored")  # nothing pending: no-op
    tracker.mark_speech_end()
    clock.advance(40)
    tracker.mark("stt_committed")
    clock.advance(60)
    tracker.mark("tts_first_chunk")
    clock.advance(20)
    timing = tracker.mark_first_audio_out()
    assert timing.marks == pytest.approx({"stt_committed": 40, "tts_first_chunk": 100})
    assert str(timing) == "utterance #1: 120 ms, stt_committed=40ms, tts_first_chunk=100ms"


def test_pending_is_replaced(clock):
    tracker = LatencyTracker(clock=clock)
    tracker.mark_speech_end()
    tracker.mark("old")
    clock.advance(500)
    assert tracker.mark_speech_end() == 2
    clock.advance(80)
    timing = tracker.mark_first_audio_out()
    assert (timing.utterance_id, timing.marks) == (2, {})
    assert timing.latency_ms == pytest.approx(80)


def test_callback_invoked(clock):
    seen = []
    tracker = LatencyTracker(clock=clock, on_utterance=seen.append)
    timing = utterance(tracker, clock, 10)
    assert seen == [timing]


def test_callback_exception_swallowed(clock, caplog):
    def bad(_):
        raise RuntimeError("callback broke")

    tracker = LatencyTracker(clock=clock, on_utterance=bad)
    with caplog.at_level(logging.ERROR, logger="latency"):
        timing = utterance(tracker, clock, 10)
    assert timing is not None and tracker.timings == [timing]
    assert "on_utterance callback failed" in caplog.text


def test_empty_summary(clock):
    summary = LatencyTracker(clock=clock).summary()
    assert summary == LatencySummary() == summarize([])
    assert str(summary) == "latency: no utterances yet"


def test_summary_percentiles(clock):
    tracker = LatencyTracker(clock=clock)
    for ms in (100, 30, 70, 10, 90, 50, 20, 80, 40, 60):  # 10..100, shuffled
        utterance(tracker, clock, ms)
    s = tracker.summary()
    assert s.count == 10
    assert (s.mean_ms, s.p50_ms, s.p95_ms, s.min_ms, s.max_ms) == pytest.approx((55, 55, 95.5, 10, 100))
    assert str(s).startswith("latency over 10 utterance(s): mean 55 ms")


@pytest.mark.parametrize("values, pct, expected", [([7.0], 95, 7.0), ([1, 2, 3, 4], 0, 1), ([1, 2, 3, 4], 100, 4)])
def test_percentile_edges(values, pct, expected):
    assert percentile(values, pct) == expected


def test_history_cap(clock):
    tracker = LatencyTracker(clock=clock, history=3)
    for ms in (10, 20, 30, 40, 50):
        utterance(tracker, clock, ms)
    assert [t.utterance_id for t in tracker.timings] == [3, 4, 5]
    assert tracker.summary().min_ms == pytest.approx(30)


def test_history_must_be_positive():
    with pytest.raises(ValueError):
        LatencyTracker(history=0)


def test_reset_keeps_ids_increasing(clock):
    tracker = LatencyTracker(clock=clock)
    utterance(tracker, clock, 10)
    tracker.mark_speech_end()  # pending, dropped by reset
    tracker.reset()
    assert tracker.timings == [] and tracker.mark_first_audio_out() is None
    assert tracker.mark_speech_end() == 3


def test_mark_at_and_stages():
    clock = FakeClock()
    tracker = LatencyTracker(clock=clock)
    start = clock()
    tracker.mark_speech_end(at=start)
    tracker.mark("pause", at=start + 0.5)
    clock.advance(600)
    tracker.mark("queue")
    clock.advance(400)
    tracker.mark("voice")
    clock.advance(20)
    timing = tracker.mark_first_audio_out()
    assert [(name, round(ms)) for name, ms in timing.stages()] == \
        [("pause", 500), ("queue", 100), ("voice", 400), ("other", 20)]


def test_stages_empty_without_marks():
    clock = FakeClock()
    tracker = LatencyTracker(clock=clock)
    tracker.mark_speech_end()
    clock.advance(100)
    assert tracker.mark_first_audio_out().stages() == []
