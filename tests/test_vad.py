from __future__ import annotations

import sys

import numpy as np
import pytest

from config import FRAME_BYTES, FRAME_MS, SAMPLE_RATE
from vad import NOISE_FLOOR_MIN, NOISE_FLOOR_TC_MS, PhraseSegmenter, VoiceActivityDetector, iter_frames, rms

SAMPLES_PER_MS = SAMPLE_RATE // 1000
BYTES_PER_MS = SAMPLES_PER_MS * 2


def silence(ms: int) -> bytes:
    return bytes(ms * BYTES_PER_MS)


def noise(ms: int, std: float, seed: int = 0) -> bytes:
    """Seeded white noise with the given std (full scale = 1.0)."""
    samples = np.random.default_rng(seed).normal(0, std, ms * SAMPLES_PER_MS)
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def speech(ms: int, seed: int = 0) -> bytes:
    """Seeded white noise, std 0.3 full-scale: webrtcvad flags it reliably (a pure sine does not)."""
    return noise(ms, 0.3, seed)


def room(ms: int, seed: int = 100) -> bytes:
    """Room noise for the RMS detector: frame RMS ~0.008, under its 0.01 threshold, so it's non-speech."""
    return noise(ms, 0.008, seed)


def quiet(ms: int, seed: int = 200) -> bytes:
    """Breath-like sound: frame RMS ~0.015, over the RMS threshold (voiced) but under 3 x a ~0.008 floor."""
    return noise(ms, 0.015, seed)


def level(amplitude: int, ms: int) -> bytes:
    """Square wave of +/-amplitude: every frame's rms() is exactly amplitude / 32768."""
    return np.resize(np.array([amplitude, -amplitude], dtype="<i2"), ms * SAMPLES_PER_MS).tobytes()


def ms_of(pcm: bytes | None) -> int:
    assert pcm is not None
    return len(pcm) // BYTES_PER_MS


def feed_all(seg: PhraseSegmenter, pcm: bytes) -> list[bytes]:
    frames, rest = iter_frames(pcm)
    assert rest == b""
    return [p for p in map(seg.feed, frames) if p is not None]


@pytest.fixture(params=["rms", "webrtc"])
def vad(request) -> VoiceActivityDetector:
    v = VoiceActivityDetector(use_webrtc=request.param == "webrtc")
    assert v.backend == request.param
    return v


# --- iter_frames / rms ---

@pytest.mark.parametrize("n_bytes, n_frames, rest", [(0, 0, 0), (639, 0, 639), (640, 1, 0), (1500, 2, 220)])
def test_iter_frames(n_bytes, n_frames, rest):
    pcm = bytes(range(256)) * 6
    frames, remainder = iter_frames(pcm[:n_bytes])
    assert len(frames) == n_frames and all(len(f) == FRAME_BYTES for f in frames)
    assert len(remainder) == rest
    assert b"".join(frames) + remainder == pcm[:n_bytes]


@pytest.mark.parametrize("frame_bytes", [0, -1])
def test_iter_frames_rejects_bad_size(frame_bytes):
    with pytest.raises(ValueError):
        iter_frames(b"abc", frame_bytes)


@pytest.mark.parametrize("samples, expected", [
    ([], 0.0),
    ([0, 0, 0], 0.0),
    ([16384, -16384], 0.5),
    ([-32768] * 4, 1.0),
    ([32767], 32767 / 32768),
])
def test_rms(samples, expected):
    assert rms(np.array(samples, dtype="<i2").tobytes()) == pytest.approx(expected)


def test_rms_ignores_odd_trailing_byte():
    assert rms(np.array([16384], dtype="<i2").tobytes() + b"\x7f") == pytest.approx(0.5)
    assert rms(b"\x01") == 0.0


# --- VoiceActivityDetector ---

@pytest.mark.parametrize("kwargs", [
    {"sample_rate": 44_100}, {"frame_ms": 25}, {"aggressiveness": 4},
    {"aggressiveness": -1}, {"rms_threshold": 1.5}, {"rms_threshold": -0.1},
])
def test_vad_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        VoiceActivityDetector(**kwargs)


def test_vad_frame_bytes():
    assert VoiceActivityDetector(sample_rate=8_000, frame_ms=30, use_webrtc=False).frame_bytes == 480


def test_vad_rejects_wrong_frame_length(vad):
    with pytest.raises(ValueError, match="Frame must be 640 bytes"):
        vad.is_speech(bytes(600))


def test_vad_falls_back_to_rms_without_webrtcvad(monkeypatch):
    monkeypatch.setitem(sys.modules, "webrtcvad", None)  # makes `import webrtcvad` raise ImportError
    assert VoiceActivityDetector().backend == "rms"


def test_vad_classifies_silence_and_noise(vad):
    assert not vad.is_speech(silence(FRAME_MS))
    frames, _ = iter_frames(speech(200))
    assert sum(map(vad.is_speech, frames)) >= len(frames) - 2


def test_rms_threshold():
    v = VoiceActivityDetector(use_webrtc=False, rms_threshold=0.5)
    assert not v.is_speech(np.full(320, 16000, dtype="<i2").tobytes())
    assert v.is_speech(np.full(320, 16384, dtype="<i2").tobytes())


# --- PhraseSegmenter ---

def test_segmenter_rejects_bad_args(vad):
    for kwargs in ({"start_frames": 0}, {"end_silence_ms": 0}, {"pre_roll_ms": -1},
                   {"min_phrase_ms": 500, "max_phrase_ms": 500},
                   {"min_snr": -1.0}, {"min_snr": float("nan")}, {"min_snr": float("inf")}):
        with pytest.raises(ValueError):
            PhraseSegmenter(vad, **kwargs)


def test_one_phrase(vad):
    seg = PhraseSegmenter(vad)
    phrases = feed_all(seg, silence(1000) + speech(1500) + silence(1000))
    assert len(phrases) == 1
    if vad.backend == "rms":
        assert ms_of(phrases[0]) == 200 + 1500 + 200  # pre-roll + speech + tail pad
    else:
        assert 1900 <= ms_of(phrases[0]) <= 1900 + 200  # webrtc hangover extends the voiced span
    assert not seg.in_speech


def test_phrase_starts_with_pre_roll(vad):
    seg = PhraseSegmenter(vad)
    [phrase] = feed_all(seg, silence(1000) + speech(1500) + silence(1000))
    assert phrase[:200 * BYTES_PER_MS] == silence(200)
    assert phrase[200 * BYTES_PER_MS:1700 * BYTES_PER_MS] == speech(1500)


def test_in_speech_tracks_state(vad):
    seg = PhraseSegmenter(vad)
    assert not seg.in_speech
    feed_all(seg, silence(200) + speech(200))
    assert seg.in_speech


def test_segmenter_defaults():
    seg = PhraseSegmenter(VoiceActivityDetector(use_webrtc=False))
    assert (seg.start_frames, seg.min_frames, seg.min_snr) == (4, 23, 3.0)  # 450 ms -> 23 frames (460 ms)
    assert seg.last_drop_reason is None


def test_short_blip_discarded(vad):
    seg = PhraseSegmenter(vad)
    assert feed_all(seg, silence(500) + speech(100) + silence(1000)) == []
    assert not seg.in_speech
    assert seg.last_drop_reason.startswith("too short (")


def test_too_short_reason():
    seg = PhraseSegmenter(VoiceActivityDetector(use_webrtc=False))
    assert feed_all(seg, silence(500) + speech(440) + silence(600)) == []  # 22 frames < 23
    assert seg.last_drop_reason == "too short (0.44 s)"
    assert len(feed_all(seg, speech(460) + silence(600))) == 1  # 23 frames: kept
    assert seg.last_drop_reason is None


def test_max_phrase_force_emits_and_continues(vad):
    seg = PhraseSegmenter(vad, max_phrase_ms=1000)
    phrases = feed_all(seg, silence(500) + speech(2000) + silence(1000))
    # 200 pre-roll + 800 speech | 1000 speech | last 200 ms of speech + 200 tail pad
    assert [ms_of(p) for p in phrases[:2]] == [1000, 1000]
    assert len(phrases) == 3  # 200 ms continuation kept even though it's < min_phrase_ms (450)
    if vad.backend == "rms":
        assert ms_of(phrases[2]) == 400
    else:
        assert 400 <= ms_of(phrases[2]) <= 600
    assert b"".join(phrases).startswith(silence(200) + speech(2000))


def test_flush_emits_in_progress_phrase(vad):
    seg = PhraseSegmenter(vad)
    feed_all(seg, silence(500) + speech(800))
    assert seg.in_speech
    assert ms_of(seg.flush()) == 200 + 800
    assert not seg.in_speech
    assert seg.flush() is None


def test_flush_drops_too_short_phrase(vad):
    seg = PhraseSegmenter(vad)
    feed_all(seg, silence(500) + speech(100))
    assert seg.in_speech
    assert seg.flush() is None
    assert seg.last_drop_reason.startswith("too short (")  # survives flush's reset()


# --- noise floor and quiet-sound gate (RMS detector, for exact levels) ---

@pytest.fixture
def rms_seg() -> PhraseSegmenter:
    return PhraseSegmenter(VoiceActivityDetector(use_webrtc=False))


def test_noise_floor_clamped_to_minimum(rms_seg):
    assert rms_seg.noise_floor == NOISE_FLOOR_MIN
    feed_all(rms_seg, silence(3000))  # digital silence
    assert rms_seg.noise_floor == NOISE_FLOOR_MIN
    feed_all(rms_seg, level(10, 3000))  # quieter than the clamp
    assert rms_seg.noise_floor == NOISE_FLOOR_MIN


def test_noise_floor_running_mean_then_ema(rms_seg):
    a, b, c = 131, 262, 196  # rms 0.0040, 0.0080, 0.0060: all non-speech (< 0.01)
    warm_up = NOISE_FLOOR_TC_MS // FRAME_MS  # frames averaged before the EMA takes over
    feed_all(rms_seg, level(a, FRAME_MS) + level(b, FRAME_MS))
    assert rms_seg.noise_floor == pytest.approx((a + b) / 2 / 32768)  # settles fast at startup
    feed_all(rms_seg, level(c, (warm_up - 2) * FRAME_MS))
    expected = (a + b + c * (warm_up - 2)) / warm_up / 32768
    assert rms_seg.noise_floor == pytest.approx(expected)
    feed_all(rms_seg, level(b, FRAME_MS))  # past warm-up: one slow EMA step
    alpha = FRAME_MS / NOISE_FLOOR_TC_MS
    assert rms_seg.noise_floor == pytest.approx(expected + alpha * (b / 32768 - expected))
    rms_seg.reset()
    assert rms_seg.noise_floor == NOISE_FLOOR_MIN


def test_noise_floor_adapts_to_room_noise(rms_seg):
    feed_all(rms_seg, room(2000))
    assert rms_seg.noise_floor == pytest.approx(0.008, rel=0.05)


def test_noise_floor_frozen_during_speech(rms_seg):
    feed_all(rms_seg, level(196, 1000))
    floor = rms_seg.noise_floor
    # Speech, a 400 ms pause (too short to end the phrase), more speech, then exactly enough silence to end it.
    phrases = feed_all(rms_seg, speech(1000) + silence(400) + speech(500, seed=1) + silence(500))
    assert len(phrases) == 1 and not rms_seg.in_speech
    assert rms_seg.noise_floor == floor  # neither the speech nor the silences inside the phrase counted
    feed_all(rms_seg, silence(FRAME_MS))  # idle again: adapts
    assert rms_seg.noise_floor < floor


def test_quiet_phrase_dropped(rms_seg):
    voiced = quiet(1000)
    # Exactly enough trailing room noise to end the phrase, so the floor is what the gate used.
    assert feed_all(rms_seg, room(2000) + voiced + room(500)) == []
    threshold = rms_seg.noise_floor * 3.0
    assert 0.01 < rms(voiced) < threshold  # voiced for the detector, but quiet for this room
    assert rms_seg.last_drop_reason == f"too quiet (level {rms(voiced):.3f} < {threshold:.3f})"
    assert rms_seg.last_drop_reason == "too quiet (level 0.015 < 0.024)"
    assert not rms_seg.in_speech


def test_loud_phrase_passes_and_clears_drop_reason(rms_seg):
    feed_all(rms_seg, room(2000) + quiet(1000) + room(500))
    assert rms_seg.last_drop_reason is not None
    voiced = speech(1000)
    tail = room(600)
    [phrase] = feed_all(rms_seg, voiced + tail)
    assert phrase == voiced + tail[:200 * BYTES_PER_MS]  # no pre-roll: the ring restarts after a drop
    assert rms_seg.last_drop_reason is None


def test_min_snr_zero_disables_quiet_gate():
    seg = PhraseSegmenter(VoiceActivityDetector(use_webrtc=False), min_snr=0)
    voiced = quiet(1000)
    [phrase] = feed_all(seg, room(2000) + voiced + room(500))
    assert ms_of(phrase) == 200 + 1000 + 200
    assert phrase[200 * BYTES_PER_MS:1200 * BYTES_PER_MS] == voiced
    assert seg.last_drop_reason is None


def test_quiet_gate_applies_on_flush(rms_seg):
    feed_all(rms_seg, room(2000) + quiet(1000))
    assert rms_seg.in_speech
    assert rms_seg.flush() is None
    assert rms_seg.last_drop_reason.startswith("too quiet (")


def test_continuation_not_gated():
    seg = PhraseSegmenter(VoiceActivityDetector(use_webrtc=False), max_phrase_ms=2000)
    loud, soft, tail = speech(1800), quiet(600), room(500)
    # 200 ms pre-roll + 1800 ms loud fills the 2000 ms buffer exactly; the quiet rest is a continuation.
    first, rest = feed_all(seg, room(1000) + loud + soft + tail)
    assert first.endswith(loud) and ms_of(first) == 2000
    assert rms(soft) < seg.noise_floor * seg.min_snr  # a fresh phrase this quiet would be dropped
    assert rest == soft + tail[:200 * BYTES_PER_MS]
    assert seg.last_drop_reason is None
