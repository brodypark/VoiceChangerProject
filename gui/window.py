"""PyQt6 window around VoiceSession (session.py). No pipeline logic lives here.

Threads: Qt runs on the main thread. Starting a session (free account check over the network,
opening audio devices) and stopping one (joining pipeline threads, up to a few seconds) run on a
short-lived worker thread so the window never freezes. The worker reports back through a queue
that a 100 ms QTimer drains; the same timer polls the running session for event lines, the mic
level and latency.

Settings shown in the window are remembered between runs (QSettings). Options without a control
here (--timing, --tts-model, --delivery, --output-format, --denoise, --vad, --latency-log) come from
the command line / .env, e.g. `python -m gui --timing balanced`.

Voices: pasted voice IDs are checked with ElevenLabs (free voices.get call, on a worker thread) and
saved with their names. The .env voice is always in the list. Picking a voice while running
switches live (VoiceSession.set_voice): the next phrase uses it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import queue
import re
import threading
from typing import Any, Callable

from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton,
                             QSlider, QVBoxLayout, QWidget)

from audio_io import AudioIOError, looks_like_speakers
from config import (DEFAULT_OUTPUT_DEVICE, DEFAULT_PAUSE_S, PAUSE_RANGE, SPEED_RANGE, TTS_VOICE_SETTINGS, Config,
                    ConfigError)
from devices import DeviceError, system_defaults
from devices import get_devices as default_get_devices
from pipelines.base import ApiFatalError, lookup_voice
from pipelines.base import make_client as default_make_client
from session import VoiceSession

logger = logging.getLogger(__name__)

TICK_MS = 100
WORKER_JOIN_ON_CLOSE_S = 10.0  # closing mid-start: wait for the account check / device open to finish
USER_ERRORS = (ConfigError, DeviceError, AudioIOError, ApiFatalError)  # messages are already user-facing
MODES = (  # (config mode, label, tooltip)
    ("stt-tts", "Speech-to-text → text-to-speech",
     "Your words, newly spoken in your voice. Clean, no mumbles. Plays after each pause."),
    ("sts", "Speech-to-speech",
     "Converts your actual audio: keeps your delivery and emotion, but mumbles can come through."),
    ("passthrough", "Passthrough (no AI)",
     "Your real voice straight to the virtual mic. No credits; use it to test routing."),
)
VOICE_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")  # ElevenLabs voice IDs are 20 alphanumerics; be lenient
ENV_VOICE_NAME = "Voice from .env"
HEADPHONE_HINTS = ("headphone", "airpods", "buds", "headset")
WARNING_STYLE = "color: #d9822b;"

SessionFactory = Callable[[Config], VoiceSession]


class ValueSlider(QWidget):
    """Horizontal slider for a float in [lo, hi] (0.01 steps) with its value shown next to it."""

    def __init__(self, lo: float, hi: float, value: float, suffix: str = "") -> None:
        super().__init__()
        self.suffix = suffix
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(lo * 100), round(hi * 100))
        self.label = QLabel()
        self.label.setMinimumWidth(48)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.slider)
        row.addWidget(self.label)
        self.slider.valueChanged.connect(lambda v: self.label.setText(f"{v / 100:.2f}{self.suffix}"))
        self.set_value(value)
        self.label.setText(f"{self.value():.2f}{self.suffix}")

    def value(self) -> float:
        return self.slider.value() / 100

    def set_value(self, value: float) -> None:
        self.slider.setValue(round(value * 100))


class MainWindow(QWidget):
    """`session_factory`, `get_devices`, `make_client` and `settings` are injectable for tests."""

    def __init__(
        self,
        config: Config,
        *,
        session_factory: SessionFactory = VoiceSession,
        get_devices: Callable[[], list[dict]] = default_get_devices,
        make_client: Callable[[str], Any] = default_make_client,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self.base_config = config
        self._session_factory = session_factory
        self._get_devices = get_devices
        self._make_client = make_client
        self._settings = settings if settings is not None else QSettings("VoiceChanger", "VoiceChanger")
        self.session: VoiceSession | None = None
        self.state = "idle"  # idle -> starting -> running -> stopping -> idle
        self._results: queue.Queue[tuple[str, Any]] = queue.Queue()  # worker thread -> UI thread
        self._worker: threading.Thread | None = None
        self._names: dict[int, str] = {}  # device index -> name
        self._voice_names: dict[str, str] = {}  # voice ID -> name

        self._build_ui()
        self.refresh_devices()
        self._load_voices()
        self._name_env_voice()
        self._load_settings()
        self._update_enabled()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(TICK_MS)

    # --- layout ---

    def _build_ui(self) -> None:
        self.setWindowTitle("Voice Changer")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)

        voice_box = QGroupBox("Voice")
        voice_layout = QVBoxLayout(voice_box)
        row = QHBoxLayout()
        self.voice_combo = QComboBox()
        self.voice_combo.setToolTip("Switch any time. While running, your next phrase uses the new voice.")
        self.remove_voice_button = QPushButton("Remove")
        self.remove_voice_button.clicked.connect(self.remove_voice)
        row.addWidget(self.voice_combo, stretch=1)
        row.addWidget(self.remove_voice_button)
        voice_layout.addLayout(row)
        row = QHBoxLayout()
        self.voice_edit = QLineEdit()
        self.voice_edit.setPlaceholderText("Paste a voice ID (ElevenLabs > Voices > Copy voice ID)")
        self.voice_edit.returnPressed.connect(self.add_voice)
        self.add_voice_button = QPushButton("Add")
        self.add_voice_button.clicked.connect(self.add_voice)
        row.addWidget(self.voice_edit, stretch=1)
        row.addWidget(self.add_voice_button)
        voice_layout.addLayout(row)
        self.voice_combo.currentIndexChanged.connect(self._on_voice_changed)
        layout.addWidget(voice_box)

        mode_box = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_box)
        self._mode_group = QButtonGroup(self)
        self.mode_buttons: dict[str, QRadioButton] = {}
        for mode, label, tip in MODES:
            button = QRadioButton(label)
            button.setToolTip(tip)
            self._mode_group.addButton(button)
            self.mode_buttons[mode] = button
            mode_layout.addWidget(button)
        self._mode_group.buttonToggled.connect(lambda *_: self._update_enabled())
        layout.addWidget(mode_box)

        device_box = QGroupBox("Audio devices")
        form = QFormLayout(device_box)
        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        self.output_combo.setToolTip("Pick this device as the microphone in Discord, Zoom, OBS, ...")
        self.monitor_check = QCheckBox("Monitor")
        self.monitor_check.setToolTip("Also play the converted voice here. Headphones only, or it feeds back.")
        self.monitor_combo = QComboBox()
        self.device_warning = QLabel()
        self.device_warning.setStyleSheet(WARNING_STYLE)
        self.device_warning.setWordWrap(True)
        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self.refresh_devices)
        self.refresh_button = refresh
        form.addRow("Microphone", self.input_combo)
        form.addRow("Virtual mic output", self.output_combo)
        form.addRow(self.monitor_check, self.monitor_combo)
        form.addRow(self.device_warning)
        form.addRow(refresh)
        for signal in (self.monitor_check.toggled, self.monitor_combo.currentIndexChanged,
                       self.output_combo.currentIndexChanged):
            signal.connect(lambda *_: self._update_enabled())
        layout.addWidget(device_box)

        self.voice_box = QGroupBox("Voice settings (applied when you press Start)")
        form = QFormLayout(self.voice_box)
        self.stability = ValueSlider(0.0, 1.0, TTS_VOICE_SETTINGS["stability"])
        self.stability.setToolTip("Lower = more expressive and varied; higher = more even and consistent.")
        self.similarity = ValueSlider(0.0, 1.0, TTS_VOICE_SETTINGS["similarity_boost"])
        self.similarity.setToolTip("How closely to match your voice. Very high can copy recording artifacts.")
        self.speed = ValueSlider(SPEED_RANGE[0], SPEED_RANGE[1], 1.0)
        self.speed.setToolTip("Speech-to-text -> text-to-speech only.")
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self.reset_voice_settings)
        form.addRow("Stability", self.stability)
        form.addRow("Similarity", self.similarity)
        form.addRow("Speed", self.speed)
        form.addRow(reset)
        layout.addWidget(self.voice_box)

        self.credit_box = QGroupBox("Phrases and credits")
        form = QFormLayout(self.credit_box)
        self.pause = ValueSlider(PAUSE_RANGE[0], PAUSE_RANGE[1], DEFAULT_PAUSE_S, suffix=" s")
        self.pause.setToolTip("Silence that ends a phrase. Shorter = your voice plays sooner, "
                              "but a pause mid-sentence can split it in two.")
        form.addRow("Pause to end a phrase", self.pause)
        self.cap_spin = QDoubleSpinBox()
        self.cap_spin.setRange(0, 3600)
        self.cap_spin.setDecimals(0)
        self.cap_spin.setSingleStep(30)
        self.cap_spin.setSuffix(" s of speech")
        self.cap_spin.setSpecialValueText("No cap")
        self.cap_spin.setToolTip("Stop automatically after this much of your speech has been sent (0 = no cap).")
        self.dry_run_check = QCheckBox("Dry run: echo phrases back, no ElevenLabs calls (speech-to-speech only)")
        form.addRow("Stop after", self.cap_spin)
        form.addRow(self.dry_run_check)
        self.dry_run_check.toggled.connect(lambda *_: self._update_enabled())
        layout.addWidget(self.credit_box)

        self.start_button = QPushButton("Start")
        self.start_button.setMinimumHeight(36)
        self.start_button.clicked.connect(self._start_or_stop)
        layout.addWidget(self.start_button)

        status = QFormLayout()
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setTextVisible(False)
        self.latency_label = QLabel("–")
        self.speech_label = QLabel("–")
        status.addRow("Mic level", self.level_bar)
        status.addRow("Latency", self.latency_label)
        status.addRow("Speech sent", self.speech_label)
        layout.addLayout(status)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        self.log_view.setMinimumHeight(160)
        layout.addWidget(self.log_view, stretch=1)

    # --- devices ---

    def refresh_devices(self) -> None:
        """Re-query devices and refill the dropdowns, keeping the current choices where they still exist."""
        keep = (self._selected_name(self.input_combo), self._selected_name(self.output_combo),
                self._selected_name(self.monitor_combo))
        try:
            devices = self._get_devices()
        except DeviceError as exc:
            self.log(f"Error: {exc}")
            return
        self._names = {d["index"]: d["name"] for d in devices}
        inputs = [d for d in devices if d["max_input_channels"] > 0 and "blackhole" not in d["name"].lower()]
        outputs = [d for d in devices if d["max_output_channels"] > 0]
        for combo, items in ((self.input_combo, inputs), (self.output_combo, outputs), (self.monitor_combo, outputs)):
            combo.blockSignals(True)
            combo.clear()
            for d in items:
                combo.addItem(d["name"], d["index"])
            combo.blockSignals(False)

        default_in, default_out = system_defaults()
        if not self._select(self.input_combo, keep[0]):
            self._select_index(self.input_combo, default_in)
        if not self._select(self.output_combo, keep[1]):
            blackhole = next((d["name"] for d in outputs if "blackhole" in d["name"].lower()), None)
            self._select(self.output_combo, blackhole)
        if not self._select(self.monitor_combo, keep[2]):
            headphones = next((d["name"] for d in outputs
                               if any(h in d["name"].lower() for h in HEADPHONE_HINTS)), None)
            if not self._select(self.monitor_combo, headphones):
                self._select_index(self.monitor_combo, default_out)
        self._update_enabled()

    @staticmethod
    def _selected_name(combo: QComboBox) -> str | None:
        return combo.currentText() or None

    @staticmethod
    def _select(combo: QComboBox, name: str | None) -> bool:
        index = combo.findText(name) if name else -1
        if index >= 0:
            combo.setCurrentIndex(index)
        return index >= 0

    @staticmethod
    def _select_index(combo: QComboBox, device_index: int | None) -> None:
        position = combo.findData(device_index) if device_index is not None else -1
        if position >= 0:
            combo.setCurrentIndex(position)

    def _device_warning_text(self) -> str:
        output_name = self.output_combo.currentText()
        if not output_name:
            return "No output devices found."
        warnings = []
        if "blackhole" not in output_name.lower():
            warnings.append("The output isn't BlackHole, so call apps can't use it as a mic. "
                            "Install it with: brew install blackhole-2ch")
        if self.monitor_check.isChecked():
            monitor_name = self.monitor_combo.currentText()
            if monitor_name == output_name:
                warnings.append("Monitor is the same device as the output. Pick your headphones.")
            elif looks_like_speakers(monitor_name):
                warnings.append(f"Monitoring through '{monitor_name}' will feed back into the mic. Use headphones.")
        return "\n".join(warnings)

    # --- voices ---

    def selected_voice_id(self) -> str | None:
        return self.voice_combo.currentData() or self.base_config.voice_id

    def _add_voice_item(self, voice_id: str, name: str) -> None:
        self._voice_names[voice_id] = name
        self.voice_combo.addItem(f"{name}  ({voice_id})", voice_id)

    def _load_voices(self) -> None:
        """Saved voices, plus the .env voice (always first if it isn't saved under a name already)."""
        try:
            saved = json.loads(self._settings.value("voices", "[]"))
        except (TypeError, ValueError):
            saved = []
        env_id = self.base_config.voice_id
        self.voice_combo.blockSignals(True)
        if env_id and env_id not in [v for v, _ in saved]:
            self._add_voice_item(env_id, ENV_VOICE_NAME)
        for voice_id, name in saved:
            self._add_voice_item(voice_id, name)
        self.voice_combo.blockSignals(False)
        position = self.voice_combo.findData(self._settings.value("voice_id", env_id))
        if position >= 0:
            self.voice_combo.setCurrentIndex(position)
        self._update_voice_buttons()

    def _name_env_voice(self) -> None:
        """Replace "Voice from .env" with the voice's real name (free lookup; failures are ignored)."""
        env_id, api_key = self.base_config.voice_id, self.base_config.api_key
        if env_id and api_key and self._voice_names.get(env_id) == ENV_VOICE_NAME:
            self._in_background("env_voice_named", lambda: (env_id, lookup_voice(self._make_client(api_key), env_id)),
                                error_kind="ignore", track=False)

    def _rename_voice(self, voice_id: str, name: str | None) -> None:
        position = self.voice_combo.findData(voice_id)
        if name and position >= 0:
            self._voice_names[voice_id] = name
            self.voice_combo.setItemText(position, f"{name}  ({voice_id})")

    def _save_voices(self) -> None:
        env_id = self.base_config.voice_id
        voices = [[self.voice_combo.itemData(i), self._voice_names[self.voice_combo.itemData(i)]]
                  for i in range(self.voice_combo.count())]
        voices = [v for v in voices if not (v[0] == env_id and v[1] == ENV_VOICE_NAME)]  # re-added on load
        self._settings.setValue("voices", json.dumps(voices))
        self._settings.setValue("voice_id", self.voice_combo.currentData() or "")
        self._settings.sync()

    def _update_voice_buttons(self) -> None:
        current = self.voice_combo.currentData()
        self.remove_voice_button.setEnabled(current is not None and current != self.base_config.voice_id)

    def add_voice(self) -> None:
        """Check a pasted voice ID with ElevenLabs (free) on a worker thread, then add and select it."""
        voice_id = self.voice_edit.text().strip().strip("'\"")
        if not voice_id or not self.add_voice_button.isEnabled():
            return
        if not VOICE_ID_RE.fullmatch(voice_id):
            self.log(f"Error: {voice_id!r} doesn't look like a voice ID (letters and digits, usually 20 of them).")
            return
        position = self.voice_combo.findData(voice_id)
        if position >= 0:
            self.voice_combo.setCurrentIndex(position)
            self.voice_edit.clear()
            return
        api_key = self.base_config.api_key
        if not api_key:
            self.log("Error: set ELEVENLABS_API_KEY in .env first (needed to check the voice).")
            return
        self.add_voice_button.setEnabled(False)
        self.add_voice_button.setText("Checking…")
        self._in_background("voice_added", lambda: (voice_id, lookup_voice(self._make_client(api_key), voice_id)),
                            error_kind="voice_error", track=False)

    def _voice_added(self, voice_id: str, name: str | None) -> None:
        self._reset_add_button()
        self.voice_edit.clear()
        if name is None:
            name = voice_id
            self.log("WARNING: couldn't check the voice (the API key has no Voices: Read permission). "
                     "Added it anyway; a wrong ID shows up as an error on the first phrase.")
        if self.voice_combo.findData(voice_id) < 0:
            self._add_voice_item(voice_id, name)
            self.log(f"Added voice {name}.")
        self.voice_combo.setCurrentIndex(self.voice_combo.findData(voice_id))  # switches live if running
        self._save_voices()

    def _reset_add_button(self) -> None:
        self.add_voice_button.setEnabled(True)
        self.add_voice_button.setText("Add")

    def remove_voice(self) -> None:
        voice_id = self.voice_combo.currentData()
        if voice_id is None or voice_id == self.base_config.voice_id:
            return  # the .env voice can only be changed in .env
        self.voice_combo.removeItem(self.voice_combo.currentIndex())
        self.log(f"Removed voice {self._voice_names.pop(voice_id, voice_id)}.")
        self._save_voices()

    def _on_voice_changed(self) -> None:
        self._update_voice_buttons()
        voice_id = self.voice_combo.currentData()
        if voice_id is None:
            return
        self._save_voices()
        session = self.session
        if self.state == "running" and session is not None and session.config.voice_id != voice_id:
            session.set_voice(voice_id)
            if session.config.mode != "passthrough":
                self.log(f"Voice: switched to {self._voice_names.get(voice_id, voice_id)}. "
                         "Your next phrase uses it.")

    # --- settings <-> controls ---

    def selected_mode(self) -> str:
        return next((mode for mode, button in self.mode_buttons.items() if button.isChecked()), "stt-tts")

    def config_from_controls(self) -> Config:
        """The base config (.env + command line) with everything set in the window applied."""
        mode = self.selected_mode()
        output = self.output_combo.currentData()
        monitor = self.monitor_combo.currentData() if self.monitor_check.isChecked() else None
        input_device = self.input_combo.currentData()
        return dataclasses.replace(
            self.base_config,
            mode=mode,
            input_device=None if input_device is None else str(input_device),  # find_device accepts an index
            output_device=DEFAULT_OUTPUT_DEVICE if output is None else str(output),
            monitor_device=None if monitor is None else str(monitor),
            voice_id=self.selected_voice_id(),
            stability=self.stability.value(),
            similarity=self.similarity.value(),
            speed=self.speed.value(),
            max_speech_s=self.cap_spin.value(),
            pause_s=self.pause.value(),
            dry_run=mode == "sts" and self.dry_run_check.isChecked(),
            list_devices=False,
        )

    def reset_voice_settings(self) -> None:
        self.stability.set_value(TTS_VOICE_SETTINGS["stability"])
        self.similarity.set_value(TTS_VOICE_SETTINGS["similarity_boost"])
        self.speed.set_value(1.0)

    def _load_settings(self) -> None:
        s, c = self._settings, self.base_config
        mode = s.value("mode", c.mode)
        self.mode_buttons.get(mode, self.mode_buttons["stt-tts"]).setChecked(True)
        self._select(self.input_combo, s.value("input"))
        self._select(self.output_combo, s.value("output"))
        self._select(self.monitor_combo, s.value("monitor"))
        self.monitor_check.setChecked(s.value("monitor_on", c.monitor_device is not None, type=bool))
        stability = TTS_VOICE_SETTINGS["stability"] if c.stability is None else c.stability
        similarity = TTS_VOICE_SETTINGS["similarity_boost"] if c.similarity is None else c.similarity
        self.stability.set_value(s.value("stability", stability, type=float))
        self.similarity.set_value(s.value("similarity", similarity, type=float))
        self.speed.set_value(s.value("speed", c.speed, type=float))
        self.cap_spin.setValue(s.value("max_speech_s", c.max_speech_s, type=float))
        self.pause.set_value(s.value("pause_s", c.pause_s, type=float))
        self.dry_run_check.setChecked(s.value("dry_run", c.dry_run, type=bool))

    def _save_settings(self) -> None:
        s = self._settings
        s.setValue("mode", self.selected_mode())
        s.setValue("input", self.input_combo.currentText())
        s.setValue("output", self.output_combo.currentText())
        s.setValue("monitor", self.monitor_combo.currentText())
        s.setValue("monitor_on", self.monitor_check.isChecked())
        s.setValue("stability", self.stability.value())
        s.setValue("similarity", self.similarity.value())
        s.setValue("speed", self.speed.value())
        s.setValue("max_speech_s", self.cap_spin.value())
        s.setValue("pause_s", self.pause.value())
        s.setValue("dry_run", self.dry_run_check.isChecked())
        s.sync()

    def _update_enabled(self) -> None:
        idle = self.state == "idle"
        mode = self.selected_mode()
        ai = mode != "passthrough"
        for widget in (*self.mode_buttons.values(), self.input_combo, self.output_combo, self.monitor_check,
                       self.refresh_button):
            widget.setEnabled(idle)
        self.monitor_combo.setEnabled(idle and self.monitor_check.isChecked())
        self.voice_box.setEnabled(idle and ai and not (mode == "sts" and self.dry_run_check.isChecked()))
        self.speed.setEnabled(mode == "stt-tts")
        self.credit_box.setEnabled(idle and ai)
        self.dry_run_check.setEnabled(mode == "sts")
        self.device_warning.setText(self._device_warning_text())
        self.device_warning.setVisible(bool(self.device_warning.text()))
        labels = {"idle": "Start", "starting": "Starting…", "running": "Stop", "stopping": "Stopping…"}
        self.start_button.setText(labels[self.state])
        self.start_button.setEnabled(self.state in ("idle", "running"))

    def _set_state(self, state: str) -> None:
        self.state = state
        self._update_enabled()

    # --- start / stop ---

    def _start_or_stop(self) -> None:
        if self.state == "idle":
            self.start()
        elif self.state == "running":
            self.stop()

    def start(self) -> None:
        if self.state != "idle":
            return
        config = self.config_from_controls()
        if config.mode != "passthrough" and not config.dry_run and config.max_speech_s == 0:
            answer = QMessageBox.question(
                self, "No credit cap",
                "There's no speech cap: everything you say uses ElevenLabs credits until you press Stop. Continue?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._save_settings()
        session = self._session_factory(config)
        self.session = session
        self.latency_label.setText("–")
        self.speech_label.setText("–")
        self._set_state("starting")
        self.log(f"Starting {dict((m, label) for m, label, _ in MODES)[config.mode]}…")
        self._in_background("started", lambda: (session.prepare(), session.start()))

    def stop(self) -> None:
        if self.state != "running" or self.session is None:
            return
        self._set_state("stopping")
        self._in_background("stopped", self.session.stop)

    def _in_background(self, kind: str, fn: Callable[[], Any], error_kind: str = "error", track: bool = True) -> None:
        """Run `fn` on a worker thread; its result (or a user-facing error, as `error_kind`) is handled by _tick.

        `track`: closing the window waits for this worker (session start/stop, not voice checks).
        """

        def run() -> None:
            try:
                self._results.put((kind, fn()))
            except USER_ERRORS as exc:
                self._results.put((error_kind, str(exc)))
            except Exception as exc:  # noqa: BLE001 - shown in the log instead of killing the worker silently
                logger.exception("%s failed", kind)
                self._results.put((error_kind, f"Unexpected error ({type(exc).__name__}: {exc})."))

        worker = threading.Thread(target=run, name=f"gui-{kind}", daemon=True)
        if track:
            self._worker = worker
        worker.start()

    def _on_result(self, kind: str, payload: Any) -> None:
        if kind == "started":
            lines, info = payload
            session = self.session
            for line in [*lines, *info.warnings, f"{session.title}: {info.route}"]:
                self.log(line)
            how_to = f" {session.how_to}" if session.how_to else ""
            self.log(f"In Discord/Zoom/etc., choose '{info.output_name}' as the microphone.{how_to}")
            self._set_state("running")
            if self.voice_combo.currentData() and self.voice_combo.currentData() != session.config.voice_id:
                self._on_voice_changed()  # the voice was switched while it was starting
        elif kind == "voice_added":
            self._voice_added(*payload)
        elif kind == "env_voice_named":
            self._rename_voice(*payload)
        elif kind == "ignore":
            pass
        elif kind == "voice_error":
            self._reset_add_button()
            self.log(f"Error: {payload}")
        elif kind == "stopped":
            for line in payload:
                self.log(line)
            self._finish()
        else:  # "error": prepare/start failed (and cleaned up) or stop failed
            self.log(f"Error: {payload}")
            self._finish()

    def _finish(self) -> None:
        self.level_bar.setValue(0)
        self.session = None
        self._set_state("idle")

    # --- timer ---

    def _tick(self) -> None:
        while True:
            try:
                kind, payload = self._results.get_nowait()
            except queue.Empty:
                break
            self._on_result(kind, payload)

        session = self.session
        if self.state != "running" or session is None:
            return
        for line in session.poll():
            self.log(line)
        # RMS of normal speech is ~0.01-0.1, so scale x5 (as the CLI meter does).
        self.level_bar.setValue(min(100, int(session.meter.level * 500)))
        self._show_stats(session)
        if session.done:  # cap reached or fatal error, and the last audio finished playing
            self.stop()

    def _show_stats(self, session: VoiceSession) -> None:
        config = session.config
        if config.mode == "passthrough":
            return
        summary = session.latency.summary()
        if summary.count:
            last = session.latency.timings[-1].latency_ms
            self.latency_label.setText(f"{last:.0f} ms last · {summary.mean_ms:.0f} ms avg · "
                                       f"{summary.count} phrase(s)  ({session.latency_label})")
        if config.dry_run:
            self.speech_label.setText("dry run, no credits")
        else:
            cap = f" of {config.max_speech_s:g} s" if config.max_speech_s > 0 else ""
            self.speech_label.setText(f"{session.speech_sent_s:.1f} s{cap}")

    def log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    # --- window close ---

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt naming
        """Close cleanly even mid-start/stop: wait for the worker, then stop a running session here."""
        self._timer.stop()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=WORKER_JOIN_ON_CLOSE_S)
        session = self.session
        if session is not None and session.running:
            try:
                session.stop()
            except Exception:  # noqa: BLE001 - closing anyway
                logger.exception("stopping the session on close failed")
        event.accept()
