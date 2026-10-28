"""GUI tests: offscreen Qt, fake devices, fake sessions. No hardware, no network, no real settings file."""

from __future__ import annotations

import dataclasses
import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before any QApplication exists

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import gui.window as window_module  # noqa: E402
from elevenlabs.core.api_error import ApiError  # noqa: E402

from config import Config, ConfigError  # noqa: E402
from gui.window import MainWindow  # noqa: E402
from latency import LatencyTracker  # noqa: E402
from session import LevelMeter, StartInfo  # noqa: E402

WAIT_S = 2.0


def dev(index: int, name: str, ins: int = 0, outs: int = 0) -> dict:
    return {"index": index, "name": name, "max_input_channels": ins,
            "max_output_channels": outs, "default_samplerate": 48_000.0}


FAKE = [
    dev(0, "MacBook Pro Microphone", ins=1),
    dev(1, "MacBook Pro Speakers", outs=2),
    dev(2, "BlackHole 2ch", ins=2, outs=2),
    dev(3, "AirPods", ins=1, outs=2),
]


def base_config(**overrides) -> Config:
    values = dict(api_key="k", voice_id="v", mode="stt-tts", input_device=None, output_device="BlackHole 2ch",
                  monitor_device=None, vad_aggressiveness=3, dry_run=False, list_devices=False)
    return Config(**{**values, **overrides})


class FakeSession:
    """Stands in for VoiceSession. `fail` = exception raised by prepare()."""

    instances: list[FakeSession] = []

    def __init__(self, config: Config, fail: Exception | None = None) -> None:
        self.config = config
        self.fail = fail
        self.meter = LevelMeter()
        self.latency = LatencyTracker()
        self.latency_label = "end of speech -> first audio out"
        self.title = "Speech-to-text -> text-to-speech (test)"
        self.how_to = "Speak, then pause."
        self.events: list[str] = []
        self.done = False
        self.error = None
        self.speech_sent_s = 0.0
        self.running = False
        self.stopped = False
        self.voices: list[str] = []  # set_voice() calls
        FakeSession.instances.append(self)

    def prepare(self) -> list[str]:
        if self.fail is not None:
            raise self.fail
        return ["Voice: Fake"]

    def start(self) -> StartInfo:
        self.running = True
        return StartInfo("[0] Mic -> [2] BlackHole 2ch", "BlackHole 2ch", ["WARNING: test warning"])

    def poll(self) -> list[str]:
        lines, self.events = self.events, []
        return lines

    def set_voice(self, voice_id: str) -> None:
        self.config = dataclasses.replace(self.config, voice_id=voice_id)
        self.voices.append(voice_id)

    def stop(self) -> list[str]:
        self.running = False
        self.stopped = True
        return ["Stopped. latency over 1 utterance(s)"]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def make_window(app, tmp_path, monkeypatch):
    monkeypatch.setattr(window_module, "system_defaults", lambda: (0, 1))
    FakeSession.instances = []
    made: list[MainWindow] = []

    def make(config: Config | None = None, devices=FAKE, session_factory=FakeSession, settings_name="s.ini",
             client=None):
        settings = QSettings(str(tmp_path / settings_name), QSettings.Format.IniFormat)
        w = MainWindow(config or base_config(), session_factory=session_factory, get_devices=lambda: list(devices),
                       make_client=lambda api_key: client or FakeVoicesClient(), settings=settings)
        made.append(w)
        return w

    yield make
    for w in made:
        w.close()


class FakeVoicesClient:
    """client.voices.get(voice_id): known IDs return a voice with a name, others raise `error`."""

    def __init__(self, known: dict[str, str] | None = None, error: Exception | None = None) -> None:
        self.known = known if known is not None else {"newvoice1234567890ab": "Narrator"}
        self.error = error or ApiError(status_code=404, headers={}, body={"detail": {"message": "not found"}})
        self.voices = self
        self.calls: list[str] = []

    def get(self, voice_id: str):
        self.calls.append(voice_id)
        if voice_id in self.known:
            return SimpleNamespace(name=self.known[voice_id])
        raise self.error


def wait_for(predicate, w: MainWindow) -> bool:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        w._tick()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def wait_state(w: MainWindow, state: str) -> bool:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        w._tick()
        if w.state == state:
            return True
        time.sleep(0.005)
    return False


def log_text(w: MainWindow) -> str:
    return w.log_view.toPlainText()


# --- controls -> config ---

def test_defaults_pick_blackhole_headphones_and_system_mic(make_window):
    w = make_window()
    assert w.selected_mode() == "stt-tts"
    assert w.input_combo.currentText() == "MacBook Pro Microphone"
    assert w.output_combo.currentText() == "BlackHole 2ch"
    assert w.monitor_combo.currentText() == "AirPods"  # headphone-like name preferred over the default output
    assert [w.input_combo.itemText(i) for i in range(w.input_combo.count())] == \
        ["MacBook Pro Microphone", "AirPods"]  # BlackHole is never offered as the mic
    assert not w.monitor_check.isChecked() and not w.monitor_combo.isEnabled()


def test_config_from_controls(make_window):
    w = make_window()
    w.mode_buttons["sts"].setChecked(True)
    w.monitor_check.setChecked(True)
    w.stability.set_value(0.3)
    w.similarity.set_value(0.8)
    w.cap_spin.setValue(120)
    w.dry_run_check.setChecked(True)
    cfg = w.config_from_controls()
    assert (cfg.mode, cfg.input_device, cfg.output_device, cfg.monitor_device) == ("sts", "0", "2", "3")
    assert (cfg.stability, cfg.similarity, cfg.max_speech_s, cfg.dry_run) == (0.3, 0.8, 120, True)
    assert cfg.api_key == "k"  # everything else comes from the base config


def test_dry_run_only_applies_to_sts(make_window):
    w = make_window()
    w.dry_run_check.setChecked(True)
    assert not w.dry_run_check.isEnabled()
    assert w.config_from_controls().dry_run is False


def test_mode_enables_relevant_controls(make_window):
    w = make_window()
    assert w.speed.isEnabled() and w.voice_box.isEnabled()
    w.mode_buttons["sts"].setChecked(True)
    assert not w.speed.isEnabled() and w.voice_box.isEnabled()
    w.mode_buttons["passthrough"].setChecked(True)
    assert not w.voice_box.isEnabled() and not w.credit_box.isEnabled()


def test_speaker_monitor_and_missing_blackhole_warnings(make_window):
    w = make_window()
    w.monitor_check.setChecked(True)
    w._select(w.monitor_combo, "MacBook Pro Speakers")
    assert "feed back" in w.device_warning.text()
    w = make_window(devices=[d for d in FAKE if "BlackHole" not in d["name"]], settings_name="other.ini")
    assert "brew install blackhole-2ch" in w.device_warning.text()


def test_settings_are_remembered(make_window):
    w = make_window()
    w.mode_buttons["sts"].setChecked(True)
    w.stability.set_value(0.25)
    w.monitor_check.setChecked(True)
    w._select(w.input_combo, "AirPods")
    w.start()
    assert wait_state(w, "running")
    w2 = make_window()
    assert w2.selected_mode() == "sts"
    assert w2.stability.value() == 0.25
    assert w2.input_combo.currentText() == "AirPods"
    assert w2.monitor_check.isChecked()


# --- start / stop ---

def test_start_run_and_stop(make_window):
    w = make_window()
    w.start()
    assert w.state == "starting" and not w.start_button.isEnabled() and not w.input_combo.isEnabled()
    assert wait_state(w, "running")
    assert w.start_button.text() == "Stop"
    (session,) = FakeSession.instances
    assert session.config.mode == "stt-tts"
    log = log_text(w)
    for expected in ("Voice: Fake", "WARNING: test warning", "Speech-to-text -> text-to-speech (test): [0] Mic",
                     "choose 'BlackHole 2ch' as the microphone. Speak, then pause."):
        assert expected in log

    session.events.append('heard: "hi" (2 chars spoken)')
    session.meter.level = 0.1
    session.latency.mark_speech_end()
    session.latency.mark_first_audio_out()
    session.speech_sent_s = 1.5
    w._tick()
    assert 'heard: "hi"' in log_text(w)
    assert w.level_bar.value() == 50
    assert "1 phrase(s)" in w.latency_label.text()
    assert w.speech_label.text() == "1.5 s of 60 s"

    w.stop()
    assert wait_state(w, "idle")
    assert session.stopped and w.session is None and w.start_button.text() == "Start"
    assert "Stopped. latency over 1" in log_text(w)


def test_session_that_finishes_itself_is_stopped(make_window):
    w = make_window()
    w.start()
    assert wait_state(w, "running")
    FakeSession.instances[0].done = True
    w._tick()
    assert wait_state(w, "idle")
    assert FakeSession.instances[0].stopped


def test_start_error_is_logged_and_returns_to_idle(make_window):
    w = make_window(session_factory=lambda cfg: FakeSession(cfg, fail=ConfigError("Missing ELEVENLABS_API_KEY.")))
    w.start()
    assert wait_state(w, "idle")
    assert "Error: Missing ELEVENLABS_API_KEY." in log_text(w)
    assert w.input_combo.isEnabled()


def test_unexpected_start_error_is_logged(make_window):
    w = make_window(session_factory=lambda cfg: FakeSession(cfg, fail=ZeroDivisionError("boom")))
    w.start()
    assert wait_state(w, "idle")
    assert "Unexpected error (ZeroDivisionError: boom)" in log_text(w)


@pytest.mark.parametrize("answer, starts", [(QMessageBox.StandardButton.No, False),
                                            (QMessageBox.StandardButton.Yes, True)])
def test_no_cap_asks_before_starting(make_window, monkeypatch, answer, starts):
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a) or answer)
    w = make_window()
    w.cap_spin.setValue(0)
    w.start()
    assert len(asked) == 1
    assert (w.state != "idle") is starts


def test_close_while_running_stops_session(make_window):
    w = make_window()
    w.start()
    assert wait_state(w, "running")
    w.close()
    assert FakeSession.instances[0].stopped


# --- voices ---

def voice_items(w: MainWindow) -> list[str]:
    return [w.voice_combo.itemData(i) for i in range(w.voice_combo.count())]


def add(w: MainWindow, text: str) -> None:
    w.voice_edit.setText(text)
    w.add_voice()


def test_env_voice_is_listed_and_cannot_be_removed(make_window):
    w = make_window()
    assert voice_items(w) == ["v"]
    assert w.voice_combo.currentText() == "Voice from .env  (v)"
    assert not w.remove_voice_button.isEnabled()
    assert w.config_from_controls().voice_id == "v"


def test_add_voice_checks_it_and_selects_it(make_window):
    client = FakeVoicesClient()
    w = make_window(client=client)
    add(w, "  'newvoice1234567890ab'  ")  # whitespace and quotes from copy-paste are stripped
    assert wait_for(lambda: w.add_voice_button.text() == "Add", w)
    assert "newvoice1234567890ab" in client.calls  # ("v" is looked up too, for its name)
    assert voice_items(w) == ["v", "newvoice1234567890ab"]
    assert w.voice_combo.currentText() == "Narrator  (newvoice1234567890ab)"
    assert w.config_from_controls().voice_id == "newvoice1234567890ab"
    assert "Added voice Narrator." in log_text(w) and w.voice_edit.text() == ""
    assert w.remove_voice_button.isEnabled()

    w2 = make_window()  # remembered, with its name
    assert voice_items(w2) == ["v", "newvoice1234567890ab"]
    assert w2.voice_combo.currentText() == "Narrator  (newvoice1234567890ab)"


def test_unknown_voice_is_not_added(make_window):
    w = make_window(client=FakeVoicesClient(known={}))
    add(w, "missing1234567890abcd")
    assert wait_for(lambda: w.add_voice_button.text() == "Add", w)
    assert voice_items(w) == ["v"]
    assert "Error: No voice with ID 'missing1234567890abcd'" in log_text(w)


def test_malformed_voice_id_is_rejected_without_a_call(make_window):
    client = FakeVoicesClient()
    w = make_window(client=client)
    add(w, "not a voice id")
    assert "doesn't look like a voice ID" in log_text(w)
    time.sleep(0.05)
    assert "not a voice id" not in client.calls


def test_voice_added_without_read_permission(make_window):
    no_permission = ApiError(status_code=401, headers={},
                             body={"detail": {"status": "missing_permissions", "message": "missing voices_read permission"}})
    w = make_window(client=FakeVoicesClient(known={}, error=no_permission))
    add(w, "unchecked1234567890ab")
    assert wait_for(lambda: w.add_voice_button.text() == "Add", w)
    assert voice_items(w) == ["v", "unchecked1234567890ab"]
    assert "couldn't check the voice" in log_text(w)


def test_remove_voice(make_window):
    w = make_window()
    add(w, "newvoice1234567890ab")
    assert wait_for(lambda: w.voice_combo.count() == 2, w)
    w.remove_voice()
    assert voice_items(w) == ["v"]
    assert voice_items(make_window()) == ["v"]


def test_switching_voice_while_running_is_live(make_window):
    w = make_window()
    add(w, "newvoice1234567890ab")
    assert wait_for(lambda: w.voice_combo.count() == 2, w)
    w.voice_combo.setCurrentIndex(0)
    w.start()
    assert wait_state(w, "running")
    session = FakeSession.instances[0]
    assert session.config.voice_id == "v"
    w.voice_combo.setCurrentIndex(1)  # stays enabled while running
    assert session.voices == ["newvoice1234567890ab"]
    assert "Voice: switched to Narrator. Your next phrase uses it." in log_text(w)
    assert w.state == "running"


def test_pause_setting(make_window):
    w = make_window()
    assert w.pause.value() == 0.5
    w.pause.set_value(0.35)
    assert w.config_from_controls().pause_s == 0.35
    assert w.pause.label.text() == "0.35 s"


def test_env_voice_gets_its_real_name(make_window):
    w = make_window(client=FakeVoicesClient(known={"v": "Blooprint"}))
    assert wait_for(lambda: w.voice_combo.currentText() == "Blooprint  (v)", w)
    assert not w.remove_voice_button.isEnabled()  # still the .env voice
