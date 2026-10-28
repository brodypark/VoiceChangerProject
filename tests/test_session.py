"""session.py tests not already covered end to end by test_main.py."""

from __future__ import annotations

import pytest

from config import Config
from session import VoiceSession, build_outputs, credit_cap_line


def make_config(**overrides) -> Config:
    values = dict(api_key=None, voice_id=None, mode="passthrough", input_device="0",
                  output_device="BlackHole 2ch", monitor_device=None,
                  vad_aggressiveness=2, dry_run=False, list_devices=False)
    return Config(**{**values, **overrides})


DEVICES = [
    {"index": 0, "name": "Mic", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48e3},
    {"index": 1, "name": "MacBook Pro Speakers", "max_input_channels": 0, "max_output_channels": 2,
     "default_samplerate": 48e3},
    {"index": 2, "name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2,
     "default_samplerate": 48e3},
]


def test_start_requires_prepare():
    with pytest.raises(RuntimeError, match="prepare"):
        VoiceSession(make_config(), get_devices=lambda: DEVICES).start()


def test_stop_before_start_is_a_no_op():
    session = VoiceSession(make_config(), get_devices=lambda: DEVICES)
    assert session.stop() == []
    assert not session.running and not session.done and session.error is None and session.speech_sent_s == 0.0


def test_passthrough_prepare_has_no_lines_and_no_network():
    def forbidden(*args):
        raise AssertionError("passthrough must not touch ElevenLabs")

    session = VoiceSession(make_config(), make_client=forbidden, check_account=forbidden)
    assert session.prepare() == []
    assert session.title == "Passthrough @ 48 kHz"


def test_speaker_monitor_returns_warning_instead_of_printing(capsys):
    _, warnings = build_outputs(2, 1, 48_000, DEVICES)
    assert warnings == ["WARNING: monitoring through 'MacBook Pro Speakers' will feed back into the mic. "
                        "Use headphones."]
    assert capsys.readouterr() == ("", "")


def test_session_start_stop_with_fake_streams(fake_sd):
    session = VoiceSession(make_config(), get_devices=lambda: DEVICES, clock=lambda: 0.0)
    session.prepare()
    info = session.start()
    assert (info.route, info.output_name) == ("[0] Mic -> [2] BlackHole 2ch", "BlackHole 2ch")
    assert session.running
    assert session.stop() == ["Stopped."]
    assert session.stop() == []  # second stop is a no-op
    assert all(s.stopped and s.closed for s in fake_sd.streams)


def test_credit_cap_line():
    assert "stops after 30 s" in credit_cap_line(make_config(max_speech_s=30))
    assert credit_cap_line(make_config(max_speech_s=0)).startswith("WARNING: no speech cap")


def timing_with_stages():
    from latency import LatencyTracker
    now = [10.0]
    tracker = LatencyTracker(clock=lambda: now[0])
    tracker.mark_speech_end(at=9.5)
    tracker.mark("pause")
    tracker.mark("queue")  # 0 ms: left out of the line
    now[0] = 10.74
    tracker.mark("voice")
    return tracker.mark_first_audio_out()


def test_latency_line_shows_stages():
    from session import latency_line
    assert latency_line(timing_with_stages(), "end of speech -> first audio out") == \
        "latency: 1240 ms (end of speech -> first audio out: pause 500 + voice 740)"


def test_latency_log_writes_csv_rows(tmp_path):
    log = tmp_path / "lat.csv"
    session = VoiceSession(make_config(mode="sts", voice_id="v1", latency_log=str(log)))
    session._on_utterance(timing_with_stages())
    session.set_voice("v2")
    session._on_utterance(timing_with_stages())
    assert session.poll() == []
    rows = log.read_text().splitlines()
    assert rows[0] == "time,mode,voice_id,latency_ms,pause_ms,queue_ms,voice_ms,word_check_ms,other_ms"
    assert [r.split(",")[1:] for r in rows[1:]] == [["sts", "v1", "1240", "500", "0", "740", "", ""],
                                                    ["sts", "v2", "1240", "500", "0", "740", "", ""]]


def test_latency_log_failure_warns_once(tmp_path):
    session = VoiceSession(make_config(mode="sts", latency_log=str(tmp_path / "missing-dir" / "lat.csv")))
    session._on_utterance(timing_with_stages())
    lines = session.poll()
    assert len(lines) == 1 and lines[0].startswith("WARNING: can't write the latency log")
    session._on_utterance(timing_with_stages())
    assert session.poll() == []


def test_set_voice_reaches_the_pipeline():
    calls = []
    session = VoiceSession(make_config(voice_id="v1"))
    session.pipeline = type("P", (), {"set_voice": lambda self, v: calls.append(v)})()
    session.set_voice("v2")
    assert session.config.voice_id == "v2" and calls == ["v2"]
