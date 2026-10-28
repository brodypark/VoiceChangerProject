"""audio_io tests. Streams are always fakes (see conftest.fake_sd); no hardware is opened."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice as sd

from audio_io import (
    AudioIOError,
    AudioOutput,
    MicCapture,
    OutputGroup,
    bytes_to_ms,
    frame_samples_for,
    looks_like_speakers,
    mono_to_channels,
    ms_to_bytes,
    output_channels_for,
)
from devices import MIC_PERMISSION_HINT

# 1 kHz makes the buffer math readable: 1 ms = 1 sample = 2 bytes.
RATE = 1000

OK = SimpleNamespace(input_overflow=False, output_underflow=False)  # stand-in for sd.CallbackFlags


def pcm(samples) -> bytes:
    """int16 little-endian PCM from a sequence of sample values."""
    return np.asarray(samples, dtype="<i2").tobytes()


def ramp(start: int, stop: int) -> bytes:
    """Distinct, ordered samples so tests can tell which audio was played or dropped."""
    return pcm(np.arange(start, stop))


def dev(index: int, name: str, ins: int = 0, outs: int = 0) -> dict:
    return {"index": index, "name": name, "max_input_channels": ins,
            "max_output_channels": outs, "default_samplerate": 48_000.0}


# --- Pure helpers ---

def test_mono_to_channels_one_channel_is_unchanged():
    data = pcm([1, -2, 3])
    assert mono_to_channels(data, 1) == data


def test_mono_to_channels_duplicates_into_stereo():
    assert mono_to_channels(pcm([1, -2, 32767]), 2) == pcm([1, 1, -2, -2, 32767, 32767])


def test_mono_to_channels_empty():
    assert mono_to_channels(b"", 2) == b""


def test_mono_to_channels_odd_length():
    with pytest.raises(ValueError, match="multiple of 2"):
        mono_to_channels(b"\x00\x01\x02", 2)


@pytest.mark.parametrize("channels", [0, -1])
def test_mono_to_channels_needs_a_channel(channels):
    with pytest.raises(ValueError, match="channels must be >= 1"):
        mono_to_channels(pcm([1]), channels)


@pytest.mark.parametrize("rate, frame_ms, expected", [
    (48_000, 20, 960),
    (16_000, 20, 320),
    (16_000, 10, 160),
    (44_100, 20, 882),
])
def test_frame_samples_for(rate, frame_ms, expected):
    assert frame_samples_for(rate, frame_ms) == expected


def test_frame_samples_for_defaults_to_20ms():
    assert frame_samples_for(48_000) == 960


def test_frame_samples_for_non_integer():
    with pytest.raises(ValueError, match="not a whole number of samples"):
        frame_samples_for(22_050, 10)  # 220.5 samples


@pytest.mark.parametrize("rate, frame_ms", [(0, 20), (-16_000, 20), (16_000, 0)])
def test_frame_samples_for_non_positive(rate, frame_ms):
    with pytest.raises(ValueError, match="must be positive"):
        frame_samples_for(rate, frame_ms)


@pytest.mark.parametrize("ms, rate, expected", [
    (20, 48_000, 1920),
    (20, 16_000, 640),
    (0, 48_000, 0),
    (0.5, 16_000, 16),
    (0.03, 48_000, 2),  # 1.44 samples rounds down to one whole sample
])
def test_ms_to_bytes(ms, rate, expected):
    assert ms_to_bytes(ms, rate) == expected


@pytest.mark.parametrize("n_bytes, rate, expected", [
    (1920, 48_000, 20.0),
    (640, 16_000, 20.0),
    (3, 16_000, 0.0625),  # dangling odd byte ignored
    (0, 16_000, 0.0),
])
def test_bytes_to_ms(n_bytes, rate, expected):
    assert bytes_to_ms(n_bytes, rate) == expected


@pytest.mark.parametrize("ms", [20, 200, 1000])
def test_ms_bytes_round_trip(ms):
    assert bytes_to_ms(ms_to_bytes(ms, 48_000), 48_000) == ms


@pytest.mark.parametrize("name, expected", [
    ("MacBook Pro Speakers", True),
    ("External SPEAKER", True),
    ("AirPods", False),
    ("External Headphones", False),
    ("BlackHole 2ch", False),
])
def test_looks_like_speakers(name, expected):
    assert looks_like_speakers(name) is expected


DEVICES = [
    dev(0, "MacBook Pro Microphone", ins=1),
    dev(1, "Mono Out", outs=1),
    dev(2, "BlackHole 16ch", ins=16, outs=16),
    dev(5, "Headphones", outs=2),
]


@pytest.mark.parametrize("index, expected", [(1, 1), (2, 2), (5, 2)])
def test_output_channels_for(index, expected):
    assert output_channels_for(index, DEVICES) == expected


def test_output_channels_for_input_only_device():
    with pytest.raises(AudioIOError, match=r"\[0\] MacBook Pro Microphone has no output channels"):
        output_channels_for(0, DEVICES)


def test_output_channels_for_unknown_index():
    with pytest.raises(AudioIOError, match="No audio device with index 3"):
        output_channels_for(3, DEVICES)  # indices are matched, not list positions


# --- AudioOutput buffering (_fill / write), no stream needed ---

def make_output(channels: int = 1, max_buffer_ms: int = 200, prefill_ms: int = 20) -> AudioOutput:
    return AudioOutput(0, RATE, channels=channels, max_buffer_ms=max_buffer_ms, prefill_ms=prefill_ms)


def test_output_idle_plays_silence_without_counting_underruns():
    out = make_output()
    for _ in range(5):
        assert out._fill(10) == bytes(20)
    assert out.underruns == 0


def test_output_waits_for_prefill():
    out = make_output(prefill_ms=20)
    out.write(ramp(1, 11))  # 10 ms < 20 ms prefill
    assert out._fill(5) == bytes(10)
    assert out.buffered_ms == 10.0
    assert out.underruns == 0


def test_output_plays_once_prefilled():
    out = make_output(prefill_ms=20)
    out.write(ramp(1, 11))
    assert out._fill(5) == bytes(10)  # still waiting
    out.write(ramp(11, 31))  # now 30 ms buffered
    assert out._fill(20) == ramp(1, 21)
    assert out._fill(10) == ramp(21, 31)
    assert out.buffered_ms == 0.0
    assert out.underruns == 0


def test_output_duplicates_mono_into_stereo():
    out = make_output(channels=2)
    out.write(ramp(1, 21))
    assert out._fill(20) == pcm(np.repeat(np.arange(1, 21), 2))


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("buffered", [0, 5, 20, 100])
@pytest.mark.parametrize("n_frames", [1, 7, 20, 64])
def test_output_fill_length_is_exact(channels, buffered, n_frames):
    out = make_output(channels=channels)
    out.write(bytes(buffered * 2))
    for _ in range(3):
        assert len(out._fill(n_frames)) == n_frames * channels * 2


def test_output_underrun_pads_with_silence_and_counts_once():
    out = make_output()
    out.write(ramp(1, 21))
    assert out._fill(30) == ramp(1, 21) + bytes(20)
    assert out.underruns == 1
    for _ in range(5):  # idle afterwards: not more underruns
        assert out._fill(10) == bytes(20)
    assert out.underruns == 1


def test_output_waits_for_prefill_again_after_underrun():
    out = make_output(prefill_ms=20)
    out.write(ramp(1, 21))
    out._fill(30)  # underrun
    out.write(ramp(100, 110))  # 10 ms: below prefill
    assert out._fill(10) == bytes(20)
    out.write(ramp(110, 120))
    assert out._fill(20) == ramp(100, 120)
    assert out.underruns == 1


def test_output_short_tail_plays_after_waiting_prefill_ms():
    out = make_output(prefill_ms=20)
    out.write(ramp(1, 6))  # 5 ms tail, nothing more coming
    assert out._fill(10) == bytes(20)  # waited 10 ms
    assert out._fill(10) == bytes(20)  # waited 20 ms
    assert out._fill(10) == ramp(1, 6) + bytes(10)  # waited long enough: play it
    assert out.underruns == 1  # the tail ran dry while playing


def test_output_tail_wait_counts_frames_not_calls():
    out = make_output(prefill_ms=20)
    out.write(ramp(1, 6))
    assert out._fill(20) == bytes(40)  # one 20 ms callback is the whole wait
    assert out._fill(20) == ramp(1, 6) + bytes(30)


def test_output_prefill_zero_plays_immediately():
    out = make_output(prefill_ms=0)
    out.write(ramp(1, 3))
    assert out._fill(2) == ramp(1, 3)


def test_output_drops_oldest_beyond_max_buffer():
    out = make_output(max_buffer_ms=200)
    out.write(ramp(0, 250))
    assert out.dropped_bytes == 100
    assert out.dropped_ms == 50.0
    assert out.buffered_ms == 200.0
    assert out._fill(200) == ramp(50, 250)  # newest 200 ms kept


def test_output_drops_accumulate_across_writes():
    out = make_output(max_buffer_ms=200)
    out.write(ramp(0, 150))
    out.write(ramp(150, 300))
    assert out.dropped_ms == 100.0
    assert out._fill(200) == ramp(100, 300)


def test_output_odd_length_writes_reassemble():
    out = make_output()
    data = ramp(1, 21)
    out.write(data[:3])
    assert out.buffered_ms == 1.0  # the dangling byte isn't a sample yet
    out.write(data[3:])
    assert out._fill(20) == data


def test_output_single_odd_byte_never_plays():
    out = make_output(prefill_ms=0)
    out.write(b"\x01")
    assert out._fill(4) == bytes(8)
    assert out.underruns == 0


def test_output_odd_overflow_drops_whole_samples():
    out = make_output(max_buffer_ms=200)  # 400 bytes
    out.write(bytes(401))
    assert out.dropped_bytes == 2
    assert out.dropped_bytes % 2 == 0


def test_output_empty_write_is_noop():
    out = make_output()
    out.write(b"")
    assert out.buffered_ms == 0.0


def test_output_clear_drops_audio_and_rearms_prefill():
    out = make_output(prefill_ms=20)
    out.write(ramp(1, 41))
    assert out._fill(10) == ramp(1, 11)  # playing
    out.clear()
    assert out.buffered_ms == 0.0
    out.write(ramp(100, 110))  # 10 ms: must wait for prefill again
    assert out._fill(10) == bytes(20)


def test_output_clear_resets_tail_wait():
    out = make_output(prefill_ms=20)
    out.write(ramp(1, 6))
    out._fill(10)  # waited 10 ms
    out.clear()
    out.write(ramp(1, 6))
    assert out._fill(10) == bytes(20)  # wait restarted, so still silent
    assert out._fill(10) == bytes(20)


def test_output_buffered_ms_at_48k():
    out = AudioOutput(0, 48_000, channels=2)
    out.write(bytes(1920))
    assert out.buffered_ms == 20.0


def test_output_callback_fills_outdata_and_counts_underflow():
    out = make_output(channels=2)
    out.write(ramp(1, 21))
    outdata = bytearray(20 * 2 * 2)
    out._callback(outdata, 20, None, SimpleNamespace(output_underflow=True))
    assert bytes(outdata) == pcm(np.repeat(np.arange(1, 21), 2))
    assert out.output_underflows == 1


@pytest.mark.parametrize("kwargs", [
    {"channels": 0},
    {"prefill_ms": -1},
    {"max_buffer_ms": 20, "prefill_ms": 20},
    {"max_buffer_ms": 10, "prefill_ms": 20},
])
def test_output_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        AudioOutput(0, RATE, **kwargs)


def test_output_constructor_defaults_and_no_stream(fake_sd):
    out = AudioOutput(3, 48_000)
    assert (out.channels, out.name, out.running) == (2, "device 3", False)
    assert out.max_buffer_bytes == ms_to_bytes(200, 48_000)
    assert out.prefill_bytes == ms_to_bytes(20, 48_000)
    assert fake_sd.streams == []


# --- MicCapture framing (_handle_block), no stream needed ---

def make_mic(rate: int = 16_000) -> tuple[MicCapture, list[bytes]]:
    received: list[bytes] = []
    return MicCapture(None, rate, received.append), received


def test_mic_constructor_does_not_open_stream(fake_sd):
    mic, _ = make_mic(48_000)
    assert (mic.frame_samples, mic.frame_bytes) == (960, 1920)
    assert mic.running is False
    assert fake_sd.streams == []


def test_mic_constructor_rejects_bad_frame():
    with pytest.raises(ValueError):
        MicCapture(None, 22_050, lambda f: None, frame_ms=10)


def test_mic_exact_frame_passes_through():
    mic, received = make_mic()
    frame = ramp(0, 320)
    mic._handle_block(frame)
    assert received == [frame]
    assert mic._pending == b""


def test_mic_irregular_blocks_split_with_remainder_carried():
    mic, received = make_mic()  # 640-byte frames
    data = ramp(0, 1000)  # 2000 bytes
    blocks = [data[:500], data[500:1000], data[1000:1500], data[1500:]]
    counts = []
    for block in blocks:
        mic._handle_block(block)
        counts.append(len(received))
    assert counts == [0, 1, 2, 3]
    assert all(len(f) == 640 for f in received)
    assert b"".join(received) == data[:1920]
    assert mic._pending == data[1920:]


def test_mic_frame_sized_block_after_remainder_keeps_order():
    mic, received = make_mic()
    data = ramp(0, 1000)
    mic._handle_block(data[:100])  # leaves a 100-byte remainder
    mic._handle_block(data[100:740])  # frame-sized, but must join the remainder first
    assert received == [data[:640]]
    assert mic._pending == data[640:740]


def test_mic_big_block_yields_several_frames():
    mic, received = make_mic()
    data = ramp(0, 1120)  # 3.5 frames
    mic._handle_block(data)
    assert received == [data[:640], data[640:1280], data[1280:1920]]
    assert mic._pending == data[1920:]


def test_mic_on_frame_errors_are_counted_not_raised():
    calls: list[bytes] = []
    boom = RuntimeError("callback broke")

    def on_frame(frame: bytes) -> None:
        calls.append(frame)
        raise boom

    mic = MicCapture(None, 16_000, on_frame)
    frame = bytes(640)
    mic._handle_block(frame)
    mic._handle_block(frame + frame)
    assert len(calls) == 3  # every frame still delivered
    assert mic.errors == 3
    assert mic.last_error is boom


def test_mic_callback_counts_overflow_and_delivers():
    mic, received = make_mic()
    mic._callback(bytearray(640), 320, None, SimpleNamespace(input_overflow=True))
    mic._callback(bytearray(640), 320, None, OK)
    assert mic.overflows == 1
    assert received == [bytes(640), bytes(640)]


# --- MicCapture start/stop with fake streams ---

def test_mic_start_opens_one_stream(fake_sd):
    mic = MicCapture(3, 48_000, lambda f: None)
    mic.start()
    mic.start()  # no-op
    assert len(fake_sd.streams) == 1
    stream = fake_sd.streams[0]
    assert stream.kind == "input" and stream.started
    assert stream.kwargs == {
        "device": 3, "samplerate": 48_000, "channels": 1, "dtype": "int16",
        "blocksize": 960, "latency": "low", "callback": mic._callback,
    }
    assert mic.running


def test_mic_stop_closes_stream(fake_sd):
    mic = MicCapture(3, 48_000, lambda f: None)
    mic.start()
    mic.stop()
    stream = fake_sd.streams[0]
    assert stream.stopped and stream.closed
    assert not mic.running
    mic.stop()  # safe when not running


def test_mic_stop_without_start(fake_sd):
    MicCapture(3, 48_000, lambda f: None).stop()
    assert fake_sd.streams == []


def test_mic_start_clears_stale_remainder(fake_sd):
    mic, _ = make_mic()
    mic._handle_block(bytes(100))
    mic.start()
    assert mic._pending == b""


def test_mic_context_manager(fake_sd):
    with MicCapture(3, 48_000, lambda f: None) as mic:
        assert mic.running
    assert fake_sd.streams[0].closed and not mic.running


@pytest.mark.parametrize("error", [sd.PortAudioError("device busy"), ValueError("bad rate")])
def test_mic_open_error_has_permission_hint(fake_sd, error):
    fake_sd.fail_open[("input", 3)] = error
    mic = MicCapture(3, 48_000, lambda f: None)
    with pytest.raises(AudioIOError) as info:
        mic.start()
    msg = str(info.value)
    assert f"Could not open input stream on device [3] at 48000 Hz: {error}" in msg
    assert MIC_PERMISSION_HINT in msg
    assert info.value.__cause__ is error
    assert not mic.running


def test_mic_open_error_on_default_device(fake_sd):
    fake_sd.fail_open[("input", None)] = sd.PortAudioError("nope")
    with pytest.raises(AudioIOError, match="on the default device"):
        MicCapture(None, 16_000, lambda f: None).start()


def test_mic_start_error_closes_stream(fake_sd):
    fake_sd.fail_start.add(("input", 3))
    mic = MicCapture(3, 48_000, lambda f: None)
    with pytest.raises(AudioIOError, match="start failed") as info:
        mic.start()
    assert MIC_PERMISSION_HINT in str(info.value)
    assert fake_sd.streams[0].closed
    assert not mic.running


# --- AudioOutput start/stop with fake streams ---

def test_output_start_opens_one_stream(fake_sd):
    out = AudioOutput(2, 48_000, channels=1)
    out.start()
    out.start()
    assert len(fake_sd.streams) == 1
    stream = fake_sd.streams[0]
    assert stream.kind == "output" and stream.started
    assert stream.kwargs == {
        "device": 2, "samplerate": 48_000, "channels": 1, "dtype": "int16",
        "latency": "low", "callback": out._callback,
    }
    assert out.running


def test_output_stop_closes_stream_and_keeps_buffer(fake_sd):
    out = AudioOutput(2, 48_000)
    out.start()
    out.write(bytes(1920))
    out.stop()
    assert fake_sd.streams[0].stopped and fake_sd.streams[0].closed
    assert not out.running
    assert out.buffered_ms == 20.0


def test_output_stop_error_still_closes(fake_sd):
    fake_sd.fail_stop.add(("output", 2))
    out = AudioOutput(2, 48_000)
    out.start()
    with pytest.raises(sd.PortAudioError):
        out.stop()
    assert fake_sd.streams[0].closed
    assert not out.running


def test_output_open_error_has_no_mic_hint(fake_sd):
    fake_sd.fail_open[("output", 2)] = sd.PortAudioError("busy")
    with pytest.raises(AudioIOError, match=r"Could not open output stream on device \[2\] at 48000 Hz: busy") as info:
        AudioOutput(2, 48_000).start()
    assert MIC_PERMISSION_HINT not in str(info.value)


def test_output_start_error_closes_stream(fake_sd):
    fake_sd.fail_start.add(("output", 2))
    out = AudioOutput(2, 48_000)
    with pytest.raises(AudioIOError, match="start failed"):
        out.start()
    assert fake_sd.streams[0].closed and not out.running


# --- OutputGroup ---

def outputs(*device_indices: int) -> list[AudioOutput]:
    return [AudioOutput(i, RATE, channels=1, name=f"out{i}") for i in device_indices]


def test_group_write_and_clear_fan_out():
    outs = outputs(1, 2)
    group = OutputGroup(outs)
    group.write(bytes(40))
    assert [o.buffered_ms for o in outs] == [20.0, 20.0]
    group.clear()
    assert [o.buffered_ms for o in outs] == [0.0, 0.0]


def test_group_start_and_stop_all(fake_sd):
    outs = outputs(1, 2)
    with OutputGroup(outs):
        assert all(o.running for o in outs)
    assert [s.device for s in fake_sd.streams] == [1, 2]
    assert all(s.stopped and s.closed for s in fake_sd.streams)
    assert not any(o.running for o in outs)


def test_group_start_rolls_back_on_failure(fake_sd):
    fake_sd.fail_open[("output", 3)] = sd.PortAudioError("busy")
    outs = outputs(1, 2, 3)
    with pytest.raises(AudioIOError, match=r"device \[3\]"):
        OutputGroup(outs).start()
    assert [s.device for s in fake_sd.streams] == [1, 2]
    assert all(s.stopped and s.closed for s in fake_sd.streams)
    assert not any(o.running for o in outs)


def test_group_rollback_error_does_not_mask_original(fake_sd):
    fake_sd.fail_stop.add(("output", 1))
    fake_sd.fail_open[("output", 2)] = sd.PortAudioError("busy")
    outs = outputs(1, 2)
    with pytest.raises(AudioIOError, match="busy"):
        OutputGroup(outs).start()
    assert fake_sd.streams[0].closed


def test_group_stop_stops_all_and_raises_first_failure(fake_sd):
    fake_sd.fail_stop.update({("output", 1), ("output", 2)})
    outs = outputs(1, 2, 3)
    group = OutputGroup(outs)
    group.start()
    with pytest.raises(AudioIOError, match="Error while stopping audio output: stop failed") as info:
        group.stop()
    assert isinstance(info.value.__cause__, sd.PortAudioError)
    assert all(s.stopped and s.closed for s in fake_sd.streams)
    assert not any(o.running for o in outs)
