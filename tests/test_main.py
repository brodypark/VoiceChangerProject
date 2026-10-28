"""main.py tests. Uses a fake .env, fake device lists, and fake streams (conftest.fake_sd)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice as sd
from elevenlabs.core.api_error import ApiError

import devices
import main
import pipelines.base
from audio_io import AudioIOError
from pipelines.base import ApiFatalError, Pipeline
from config import Config, load_config
from devices import MIC_PERMISSION_HINT, DeviceError
from main import METER_WIDTH, LevelMeter, resolve_devices

OK = SimpleNamespace(input_overflow=False, output_underflow=False)  # stand-in for sd.CallbackFlags


def dev(index: int, name: str, ins: int = 0, outs: int = 0) -> dict:
    return {"index": index, "name": name, "max_input_channels": ins,
            "max_output_channels": outs, "default_samplerate": 48_000.0}


FAKE = [
    dev(0, "MacBook Pro Microphone", ins=1),
    dev(1, "MacBook Pro Speakers", outs=2),
    dev(2, "BlackHole 2ch", ins=2, outs=2),
    dev(3, "AirPods", ins=1, outs=2),
    dev(4, "External Headphones", outs=2),
]
NO_BLACKHOLE = [d for d in FAKE if "BlackHole" not in d["name"]]


def make_config(**overrides) -> Config:
    values = dict(api_key=None, voice_id=None, mode="passthrough", input_device="0",
                  output_device="BlackHole 2ch", monitor_device=None,
                  vad_aggressiveness=2, dry_run=False, list_devices=False)
    return Config(**{**values, **overrides})


def tone(value: int, samples: int = 960) -> bytes:
    return np.full(samples, value, dtype="<i2").tobytes()


# --- resolve_devices ---

def test_resolve_default_output_is_blackhole():
    assert resolve_devices(make_config(), FAKE) == (0, 2, None)


def test_resolve_input_by_name_and_monitor():
    cfg = make_config(input_device="macbook pro microphone", monitor_device="External Headphones")
    assert resolve_devices(cfg, FAKE) == (0, 2, 4)


def test_resolve_default_input_uses_system_default(monkeypatch):
    monkeypatch.setattr(devices, "system_defaults", lambda: (3, 1))
    assert resolve_devices(make_config(input_device=None), FAKE) == (3, 2, None)


def test_resolve_missing_blackhole_has_brew_hint():
    with pytest.raises(DeviceError, match="BlackHole virtual mic not found") as info:
        resolve_devices(make_config(), NO_BLACKHOLE)
    assert "brew install blackhole-2ch" in str(info.value)


@pytest.mark.parametrize("query", ["2", "BlackHole"])
def test_resolve_rejects_blackhole_input(query):
    with pytest.raises(DeviceError, match="Input is BlackHole"):
        resolve_devices(make_config(input_device=query), FAKE)


@pytest.mark.parametrize("output, monitor", [
    ("BlackHole 2ch", "BlackHole"),  # default output
    ("AirPods", "3"),                # explicit output, monitor by index
])
def test_resolve_rejects_monitor_equal_to_output(output, monitor):
    with pytest.raises(DeviceError, match="same device as the output"):
        resolve_devices(make_config(output_device=output, monitor_device=monitor), FAKE)


def test_resolve_explicit_output_name():
    cfg = make_config(output_device="External Headphones", monitor_device="AirPods")
    assert resolve_devices(cfg, FAKE) == (0, 4, 3)


def test_resolve_explicit_output_works_without_blackhole():
    assert resolve_devices(make_config(output_device="AirPods"), NO_BLACKHOLE) == (0, 3, None)


def test_resolve_unknown_output():
    with pytest.raises(DeviceError, match="No output device matching 'Zoom'"):
        resolve_devices(make_config(output_device="Zoom"), FAKE)


# --- LevelMeter ---

def test_meter_starts_empty():
    meter = LevelMeter()
    assert (meter.level, meter.peak) == (0.0, 0.0)
    assert meter.bar() == "-" * METER_WIDTH


def test_meter_zeros_give_empty_bar():
    meter = LevelMeter()
    meter.update(bytes(1920))
    assert meter.level == 0.0 and meter.peak == 0.0
    assert "#" not in meter.bar()


def test_meter_full_scale_caps_bar():
    meter = LevelMeter()
    meter.update(tone(32767))
    assert meter.level == pytest.approx(1.0, abs=1e-4)
    assert meter.bar() == "#" * METER_WIDTH


@pytest.mark.parametrize("level, filled", [
    (0.0, 0), (0.005, 0), (0.02, 3), (0.1, 15), (0.2, 30), (0.5, 30), (1.0, 30),
])
def test_meter_bar_scaling(level, filled):
    meter = LevelMeter()
    meter.level = level
    bar = meter.bar()
    assert len(bar) == METER_WIDTH
    assert bar == "#" * filled + "-" * (METER_WIDTH - filled)


def test_meter_level_from_frame():
    meter = LevelMeter()
    meter.update(tone(3277))  # ~0.1 full scale -> half the bar
    assert meter.level == pytest.approx(0.1, abs=1e-3)
    assert meter.bar().count("#") == 15


def test_meter_peak_holds_max():
    meter = LevelMeter()
    meter.update(tone(16384))
    meter.update(tone(100))
    assert meter.level == pytest.approx(100 / 32768)
    assert meter.peak == pytest.approx(0.5)


# --- main() ---

@pytest.fixture
def isolated_main(monkeypatch, env_file):
    """main.load_config bound to a fake .env, and devices never touch hardware."""
    path = env_file()
    monkeypatch.setattr(main, "load_config", lambda argv: load_config(argv, env_path=path))
    monkeypatch.setattr(main, "list_devices", lambda: "FAKE DEVICE LIST")
    monkeypatch.setattr(main, "get_devices", lambda: FAKE)
    return monkeypatch


@pytest.fixture
def passthrough_calls(isolated_main) -> list[Config]:
    """Replace run_passthrough with a recorder that returns 0."""
    calls: list[Config] = []

    def fake_run(config: Config) -> int:
        calls.append(config)
        return 0

    isolated_main.setattr(main, "run_passthrough", fake_run)
    return calls


@pytest.fixture
def sts_calls(isolated_main) -> list[Config]:
    """Replace run_sts with a recorder that returns 0."""
    calls: list[Config] = []

    def fake_run(config: Config) -> int:
        calls.append(config)
        return 0

    isolated_main.setattr(main, "run_sts", fake_run)
    return calls


def test_main_list_devices(isolated_main, capsys):
    assert main.main(["--list-devices"]) == 0
    assert "FAKE DEVICE LIST" in capsys.readouterr().out


def test_main_without_flags_runs_stt_tts(isolated_main):
    calls: list[Config] = []
    isolated_main.setattr(main, "run_stt_tts", lambda config: calls.append(config) or 0)
    assert main.main([]) == 0
    assert [c.mode for c in calls] == ["stt-tts"]


def test_main_sts_calls_run_sts(sts_calls, passthrough_calls):
    assert main.main(["--mode", "sts", "--dry-run", "--max-speech", "5"]) == 0
    (cfg,) = sts_calls
    assert (cfg.mode, cfg.dry_run, cfg.max_speech_s) == ("sts", True, 5.0)
    assert passthrough_calls == []


def test_main_config_error_returns_2(isolated_main, capsys):
    isolated_main.setenv("VOICE_CHANGER_MODE", "bogus")
    assert main.main([]) == 2
    assert "Invalid mode" in capsys.readouterr().err


def test_main_device_error_returns_2(isolated_main, capsys):
    def boom() -> str:
        raise DeviceError("no devices here")

    isolated_main.setattr(main, "list_devices", boom)
    assert main.main(["--list-devices"]) == 2
    assert "no devices here" in capsys.readouterr().err


def test_main_passthrough_calls_run_passthrough(passthrough_calls):
    assert main.main(["--mode", "passthrough", "--input", "0", "--monitor", "AirPods"]) == 0
    assert len(passthrough_calls) == 1
    cfg = passthrough_calls[0]
    assert (cfg.mode, cfg.input_device, cfg.monitor_device) == ("passthrough", "0", "AirPods")


def test_main_passthrough_mode_from_env(passthrough_calls, isolated_main):
    isolated_main.setenv("VOICE_CHANGER_MODE", "passthrough")
    assert main.main([]) == 0
    assert passthrough_calls[0].mode == "passthrough"


def test_main_stt_tts_calls_run_stt_tts(isolated_main, passthrough_calls, sts_calls):
    calls: list[Config] = []
    isolated_main.setattr(main, "run_stt_tts", lambda config: calls.append(config) or 0)
    assert main.main(["--mode", "stt-tts", "--max-speech", "5"]) == 0
    (cfg,) = calls
    assert (cfg.mode, cfg.max_speech_s) == ("stt-tts", 5.0)
    assert passthrough_calls == [] and sts_calls == []


def test_main_stt_tts_dry_run_returns_2_with_hint(isolated_main, fake_sd, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("a dry run must not touch ElevenLabs")

    isolated_main.setattr(main, "make_client", forbidden)
    isolated_main.setattr(main, "check_account", forbidden)
    assert main.main(["--mode", "stt-tts", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "Error: --dry-run isn't available for stt-tts" in err
    assert "Use --mode sts --dry-run" in err
    assert fake_sd.streams == []


def test_main_audio_error_returns_2(isolated_main, capsys):
    def fail(config: Config) -> int:
        raise AudioIOError("mic exploded")

    isolated_main.setattr(main, "run_passthrough", fail)
    assert main.main(["--mode", "passthrough"]) == 2
    assert "Error: mic exploded" in capsys.readouterr().err


def test_main_ctrl_c_during_startup_returns_130(isolated_main):
    def interrupted(config: Config) -> int:
        raise KeyboardInterrupt

    isolated_main.setattr(main, "run_passthrough", interrupted)
    assert main.main(["--mode", "passthrough"]) == 130


# --- run_passthrough end to end with fake streams (no hardware) ---

class FakeClock:
    """Replaces main.time: monotonic() ticks 1 s per call; sleep() runs a hook, then Ctrl+C after `ticks`."""

    def __init__(self, ticks: int = 1, on_sleep=None) -> None:
        self.ticks = ticks
        self.on_sleep = on_sleep
        self.now = 0.0
        self.sleeps = 0

    def monotonic(self) -> float:
        self.now += 1.0
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        if self.on_sleep is not None:
            self.on_sleep(self.sleeps)
        if self.sleeps >= self.ticks:
            raise KeyboardInterrupt


def test_passthrough_loops_mic_into_blackhole(isolated_main, fake_sd, capsys):
    frame = tone(1000)
    played: list[bytes] = []

    def on_sleep(n: int) -> None:
        (mic,) = fake_sd.of_kind("input")
        (out,) = fake_sd.of_kind("output")
        mic.kwargs["callback"](bytearray(frame), 960, None, OK)
        outdata = bytearray(960 * 2 * 2)
        out.kwargs["callback"](outdata, 960, None, OK)
        played.append(bytes(outdata))

    isolated_main.setattr(main, "time", FakeClock(ticks=1, on_sleep=on_sleep))
    assert main.main(["--mode", "passthrough", "--input", "0"]) == 0

    out_stream, = fake_sd.of_kind("output")
    mic_stream, = fake_sd.of_kind("input")
    assert (out_stream.device, out_stream.kwargs["channels"], out_stream.kwargs["samplerate"]) == (2, 2, 48_000)
    assert (mic_stream.device, mic_stream.kwargs["blocksize"]) == (0, 960)
    assert played == [np.repeat(np.frombuffer(frame, dtype="<i2"), 2).tobytes()]  # stereo copy of the mic frame
    assert all(s.stopped and s.closed for s in fake_sd.streams)

    out = capsys.readouterr().out
    assert "Passthrough @ 48 kHz: [0] MacBook Pro Microphone -> [2] BlackHole 2ch" in out
    assert "choose 'BlackHole 2ch' as the microphone" in out
    assert "Stopped." in out


def test_passthrough_monitor_and_speaker_warning(isolated_main, fake_sd, capsys):
    isolated_main.setattr(main, "time", FakeClock())
    assert main.main(["--mode", "passthrough", "--input", "0", "--monitor", "Speakers"]) == 0
    assert [s.device for s in fake_sd.of_kind("output")] == [2, 1]
    captured = capsys.readouterr()
    assert "WARNING: monitoring through 'MacBook Pro Speakers'" in captured.err
    assert "+ monitor [1] MacBook Pro Speakers" in captured.out


def test_passthrough_headphone_monitor_has_no_warning(isolated_main, fake_sd, capsys):
    isolated_main.setattr(main, "time", FakeClock())
    assert main.main(["--mode", "passthrough", "--input", "0", "--monitor", "External Headphones"]) == 0
    assert "WARNING" not in capsys.readouterr().err


def test_passthrough_hints_when_mic_is_silent(isolated_main, fake_sd, capsys):
    isolated_main.setattr(main, "time", FakeClock(ticks=6))
    assert main.main(["--mode", "passthrough", "--input", "0"]) == 0
    assert capsys.readouterr().out.count("No sound from the mic at all") == 1


def test_passthrough_mic_open_failure_returns_2_and_closes_outputs(isolated_main, fake_sd, capsys):
    fake_sd.fail_open[("input", 0)] = sd.PortAudioError("mic denied")
    isolated_main.setattr(main, "time", FakeClock())
    assert main.main(["--mode", "passthrough", "--input", "0"]) == 2
    err = capsys.readouterr().err
    assert "Could not open input stream on device [0]" in err
    assert MIC_PERMISSION_HINT in err
    (out,) = fake_sd.streams
    assert out.kind == "output" and out.stopped and out.closed


# --- run_sts end to end: fake streams, fake ElevenLabs client, no hardware, no network ---

FAKE_KEY = "FAKE-KEY-not-a-real-key"
STS_FRAME_BYTES = 640  # 20 ms at 16 kHz
BYTES_PER_MS_16K = 32


def speech(ms: int, seed: int = 0) -> bytes:
    """Seeded white noise, std 0.3 full scale: webrtcvad flags it reliably."""
    noise = np.random.default_rng(seed).normal(0, 0.3, ms * 16)
    return (np.clip(noise, -1, 1) * 32767).astype("<i2").tobytes()


def utterance(ms: int = 1000, seed: int = 0) -> bytes:
    """Speech wrapped in silence; the trailing silence is long enough to end the phrase despite webrtc's hangover."""
    return bytes(300 * BYTES_PER_MS_16K) + speech(ms, seed) + bytes(900 * BYTES_PER_MS_16K)


def feed_mic(fake_sd, pcm: bytes) -> None:
    """Push 16 kHz audio through MicCapture's PortAudio callback, one 20 ms block at a time."""
    (mic,) = fake_sd.of_kind("input")
    for i in range(0, len(pcm), STS_FRAME_BYTES):
        mic.kwargs["callback"](bytearray(pcm[i:i + STS_FRAME_BYTES]), STS_FRAME_BYTES // 2, None, OK)


def pull_output(fake_sd, frames: int = 1600) -> bytes:
    """Run the BlackHole output callback once; returns the left channel as mono int16 bytes."""
    out = fake_sd.of_kind("output")[0]
    channels = out.kwargs["channels"]
    outdata = bytearray(frames * channels * 2)
    out.kwargs["callback"](outdata, frames, None, OK)
    return np.frombuffer(bytes(outdata), dtype="<i2")[::channels].tobytes()


class FakeStsClient:
    """ElevenLabs client stand-in. speech_to_speech.stream() pops `script`: chunks, or an exception to raise."""

    def __init__(self) -> None:
        self.script: list = []
        self.calls: list[dict] = []
        self.speech_to_speech = self

    def stream(self, voice_id: str, **kwargs):
        self.calls.append({"voice_id": voice_id, **kwargs})
        step = self.script.pop(0) if self.script else [b"\x01\x00" * 480]

        def gen():
            if isinstance(step, BaseException):
                raise step
            yield from step

        return gen()


@pytest.fixture
def sts_client(isolated_main) -> FakeStsClient:
    """Fake credentials, client and (passing) startup check for non-dry-run run_sts tests."""
    isolated_main.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    isolated_main.setenv("ELEVENLABS_VOICE_ID", "fake-voice-id")
    client = FakeStsClient()
    isolated_main.setattr(main, "make_client", lambda api_key: client)
    isolated_main.setattr(main, "check_account", lambda c, voice_id: ["Voice: Fake Voice"])
    return client


def test_run_sts_dry_run_echoes_phrase_to_blackhole(isolated_main, fake_sd, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("a dry run must not touch ElevenLabs")

    isolated_main.setattr(main, "make_client", forbidden)
    isolated_main.setattr(main, "check_account", forbidden)
    voiced = speech(1000, seed=7)
    played = bytearray()

    def on_sleep(n: int) -> None:
        if n == 1:
            feed_mic(fake_sd, bytes(300 * BYTES_PER_MS_16K) + voiced + bytes(900 * BYTES_PER_MS_16K))
            deadline = time.monotonic() + 1.0
            while voiced not in played and time.monotonic() < deadline:
                played.extend(pull_output(fake_sd))
                time.sleep(0.001)

    isolated_main.setattr(main, "time", FakeClock(ticks=2, on_sleep=on_sleep))
    assert main.main(["--mode", "sts", "--dry-run", "--input", "0"]) == 0

    assert voiced in played  # the whole phrase came out of BlackHole, unchanged
    (out_stream,) = fake_sd.of_kind("output")
    (mic_stream,) = fake_sd.of_kind("input")
    assert (out_stream.device, out_stream.kwargs["samplerate"], out_stream.kwargs["channels"]) == (2, 16_000, 2)
    assert (mic_stream.device, mic_stream.kwargs["samplerate"], mic_stream.kwargs["blocksize"]) == (0, 16_000, 320)
    assert all(s.stopped and s.closed for s in fake_sd.streams)

    out = capsys.readouterr().out
    assert "DRY RUN: no ElevenLabs calls" in out
    assert "Speech-to-speech (dry run): [0] MacBook Pro Microphone -> [2] BlackHole 2ch" in out
    assert out.count("echoed (dry run, no credits") == 1
    assert "latency: " in out and "latency over 1 utterance(s)" in out
    assert "Speech sent this run" not in out


def test_run_sts_fatal_account_check_returns_2(sts_client, isolated_main, fake_sd, capsys):
    def bad_voice(client, voice_id):
        raise ApiFatalError("Voice not found. Check ELEVENLABS_VOICE_ID in .env.")

    isolated_main.setattr(main, "check_account", bad_voice)
    assert main.main(["--mode", "sts", "--input", "0"]) == 2
    captured = capsys.readouterr()
    assert "Error: Voice not found. Check ELEVENLABS_VOICE_ID in .env." in captured.err
    assert fake_sd.streams == []  # failed before any audio stream was opened


def test_run_sts_real_account_check_404_returns_2_without_leaking_key(isolated_main, sts_client, fake_sd, capsys):
    def get(voice_id):
        raise ApiError(status_code=404, headers={"xi-api-key": FAKE_KEY},
                       body={"detail": {"status": "voice_not_found", "message": "No voice fake-voice-id"}})

    isolated_main.setattr(main, "check_account", pipelines.base.check_account)
    sts_client.voices = SimpleNamespace(get=get)
    assert main.main(["--mode", "sts", "--input", "0"]) == 2
    captured = capsys.readouterr()
    assert "Error: Voice not found: No voice fake-voice-id." in captured.err
    assert FAKE_KEY not in captured.out + captured.err
    assert sts_client.calls == [] and fake_sd.streams == []


def test_run_sts_missing_credentials_returns_2(isolated_main, fake_sd, capsys):
    assert main.main(["--mode", "sts", "--input", "0"]) == 2
    assert "Missing ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID" in capsys.readouterr().err
    assert fake_sd.streams == []


def test_run_sts_cap_ends_loop_once_output_drains(sts_client, isolated_main, fake_sd, capsys):
    chunk = b"\x10\x00" * 2400  # 50 ms at 48 kHz
    sts_client.script = [[chunk, b"", chunk + b"\x01"]]  # an odd trailing byte must not keep the loop alive

    def on_sleep(n: int) -> None:
        if n == 1:
            feed_mic(fake_sd, utterance(1000, 1) + utterance(1000, 2))  # ~1.4 s each; cap is 2 s
        time.sleep(0.002)  # let the worker thread run
        pull_output(fake_sd, 4800)

    clock = FakeClock(ticks=500, on_sleep=on_sleep)  # Ctrl+C only as a safety net
    isolated_main.setattr(main, "time", clock)
    assert main.main(["--mode", "sts", "--input", "0", "--max-speech", "2"]) == 0
    assert clock.sleeps < 500  # the loop ended by itself

    assert len(sts_client.calls) == 1  # the second phrase was never sent
    (out_stream,) = fake_sd.of_kind("output")
    assert out_stream.kwargs["samplerate"] == 48_000
    assert all(s.stopped and s.closed for s in fake_sd.streams)
    out = capsys.readouterr().out
    assert "Voice: Fake Voice" in out
    assert "Credit cap: stops after 2 s of speech" in out
    assert "Speech-to-speech (pcm_48000)" in out
    assert out.count("Speech cap reached (2 s)") == 1
    assert "Speech sent this run: 1." in out


def test_run_sts_pipeline_fatal_error_returns_2_and_prints_once(sts_client, isolated_main, fake_sd, capsys):
    sts_client.script = [ApiError(status_code=401, headers={"xi-api-key": FAKE_KEY},
                                  body={"detail": {"status": "invalid_api_key", "message": "Invalid API key"}})]

    def on_sleep(n: int) -> None:
        if n == 1:
            feed_mic(fake_sd, utterance(1000))
        time.sleep(0.002)

    clock = FakeClock(ticks=500, on_sleep=on_sleep)
    isolated_main.setattr(main, "time", clock)
    assert main.main(["--mode", "sts", "--input", "0"]) == 2
    assert clock.sleeps < 500

    captured = capsys.readouterr()
    assert captured.out.count("ERROR: ElevenLabs rejected the API key: Invalid API key.") == 1
    assert "ERROR" not in captured.err and "Error:" not in captured.err  # main() doesn't print it again
    assert FAKE_KEY not in captured.out + captured.err
    assert "Stopped." in captured.out
    assert all(s.stopped and s.closed for s in fake_sd.streams)


def test_run_sts_ctrl_c_without_cap_warns_and_reports_usage(sts_client, isolated_main, fake_sd, capsys):
    isolated_main.setattr(main, "time", FakeClock(ticks=1))
    assert main.main(["--mode", "sts", "--input", "0", "--max-speech", "0", "--output-format", "pcm_24000"]) == 0
    assert fake_sd.of_kind("output")[0].kwargs["samplerate"] == 24_000
    out = capsys.readouterr().out
    assert "WARNING: no speech cap" in out
    assert "Speech sent this run: 0.0 s" in out
    assert sts_client.calls == []


# --- run_stt_tts end to end: fake pipeline (the real one is covered by test_stt_tts.py), fake streams ---

class FakeSttTtsPipeline(Pipeline):
    """Stands in for SttTtsPipeline: echoes each fed frame's arrival as a 48 kHz chunk on start()."""

    instances: list[FakeSttTtsPipeline] = []

    def __init__(self, config: Config, on_audio_out, latency=None) -> None:
        super().__init__(on_audio_out)
        self.config = config
        self.latency = latency
        self.input_sample_rate = 16_000
        self.output_sample_rate = config.output_rate
        self.speech_sent_s = 0.0
        self.frames = 0
        self.started = self.stopped = False
        FakeSttTtsPipeline.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def feed_audio(self, frame: bytes) -> None:
        self.frames += 1

    def speak(self, pcm: bytes) -> None:
        """What the loop thread does when TTS audio arrives."""
        self.latency.mark_speech_end()
        self.latency.mark_first_audio_out()
        self.speech_sent_s += 1.5
        self._emit('heard: "Hello there." (13 chars spoken)')
        self.on_audio_out(pcm)


@pytest.fixture
def stt_tts_env(isolated_main):
    """Fake credentials, startup check and SttTtsPipeline for run_stt_tts tests."""
    import pipelines.stt_tts

    isolated_main.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    isolated_main.setenv("ELEVENLABS_VOICE_ID", "fake-voice-id")
    isolated_main.setattr(main, "make_client", lambda api_key: object())
    isolated_main.setattr(main, "check_account", lambda c, voice_id: ["Voice: Fake Voice"])
    FakeSttTtsPipeline.instances = []
    isolated_main.setattr(pipelines.stt_tts, "SttTtsPipeline", FakeSttTtsPipeline)
    return isolated_main


@pytest.mark.parametrize("extra_args, title, latency_label", [
    ([], "Speech-to-text -> text-to-speech (eleven_v4_turbo, smooth timing, pcm_48000)",
     "(end of speech -> first audio out)"),
    (["--timing", "balanced"], "Speech-to-text -> text-to-speech (eleven_flash_v2_5, balanced timing, pcm_48000)",
     "(start of speech -> first audio out)"),
])
def test_run_stt_tts_plays_pipeline_audio_on_blackhole(stt_tts_env, fake_sd, capsys, extra_args, title,
                                                       latency_label):
    voice = (np.arange(4800) % 2000 - 1000).astype("<i2").tobytes()  # 100 ms at 48 kHz
    played = bytearray()

    def on_sleep(n: int) -> None:
        if n == 1:
            feed_mic(fake_sd, bytes(200 * BYTES_PER_MS_16K))
            FakeSttTtsPipeline.instances[0].speak(voice)
            played.extend(pull_output(fake_sd, 4800))

    stt_tts_env.setattr(main, "time", FakeClock(ticks=2, on_sleep=on_sleep))
    assert main.main(["--mode", "stt-tts", "--input", "0", "--max-speech", "30", *extra_args]) == 0

    assert bytes(played) == voice  # the pipeline's audio came out of BlackHole unchanged
    (pipeline,) = FakeSttTtsPipeline.instances
    assert pipeline.started and pipeline.stopped
    assert pipeline.frames == 10  # 200 ms of 20 ms mic frames reached feed_audio
    (out_stream,) = fake_sd.of_kind("output")
    (mic_stream,) = fake_sd.of_kind("input")
    assert (out_stream.device, out_stream.kwargs["samplerate"]) == (2, 48_000)
    assert (mic_stream.device, mic_stream.kwargs["samplerate"], mic_stream.kwargs["blocksize"]) == (0, 16_000, 320)
    assert all(s.stopped and s.closed for s in fake_sd.streams)

    out = capsys.readouterr().out
    assert "Voice: Fake Voice" in out
    assert "Credit cap: stops after 30 s of speech" in out
    assert f"{title}: [0] MacBook Pro Microphone -> [2] BlackHole 2ch" in out
    assert 'heard: "Hello there." (13 chars spoken)' in out
    assert latency_label in out
    assert "latency over 1 utterance(s)" in out
    assert "Speech sent this run: 1.5 s" in out


def test_run_stt_tts_fatal_pipeline_error_returns_2(stt_tts_env, fake_sd, capsys):
    def on_sleep(n: int) -> None:
        if n == 1:
            p = FakeSttTtsPipeline.instances[0]
            p.error = ApiFatalError("Out of ElevenLabs credits.")
            p._emit("ERROR: Out of ElevenLabs credits.")
            p.finished.set()

    clock = FakeClock(ticks=50, on_sleep=on_sleep)
    stt_tts_env.setattr(main, "time", clock)
    assert main.main(["--mode", "stt-tts", "--input", "0"]) == 2
    assert clock.sleeps < 50  # the loop ended by itself once finished
    out = capsys.readouterr().out
    assert out.count("ERROR: Out of ElevenLabs credits.") == 1


def test_run_stt_tts_missing_credentials_returns_2(isolated_main, fake_sd, capsys):
    assert main.main(["--mode", "stt-tts", "--input", "0"]) == 2
    assert "Missing ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID" in capsys.readouterr().err
    assert fake_sd.streams == []
