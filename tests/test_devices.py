from __future__ import annotations

import pytest
import sounddevice as sd

import devices
from devices import DeviceError, find_blackhole, find_device, format_device_list, get_devices, list_devices


def dev(index: int, name: str, ins: int = 0, outs: int = 0, rate: float = 48_000.0) -> dict:
    return {"index": index, "name": name, "max_input_channels": ins,
            "max_output_channels": outs, "default_samplerate": rate}


FAKE = [
    dev(0, "MacBook Pro Microphone", ins=1),
    dev(1, "MacBook Pro Speakers", outs=2),
    dev(2, "BlackHole 2ch", ins=2, outs=2),
    dev(3, "USB Mic", ins=1, rate=44_100.0),
    dev(4, "USB Mic Pro", ins=1),
    dev(5, "AirPods", ins=1, outs=2),
]
NO_BLACKHOLE = [d for d in FAKE if "BlackHole" not in d["name"]]


@pytest.fixture
def defaults(monkeypatch):
    """Set what devices.system_defaults() returns."""
    def set_defaults(inp, out):
        monkeypatch.setattr(devices, "system_defaults", lambda: (inp, out))
    return set_defaults


# --- get_devices / list_devices ---

def test_get_devices_normalizes_and_fills_index():
    raw = [{"name": "A", "max_input_channels": 1}, {"index": 7, "name": "B", "max_output_channels": 2}]
    got = get_devices(lambda: raw)
    assert got == [dev(0, "A", ins=1, rate=0.0), dev(7, "B", outs=2, rate=0.0)]


@pytest.mark.parametrize("exc", [sd.PortAudioError("boom"), OSError("boom")])
def test_query_errors_wrapped_with_permission_hint(exc):
    def query():
        raise exc

    with pytest.raises(DeviceError, match="Could not query audio devices: boom") as info:
        get_devices(query)
    assert "Privacy & Security > Microphone" in str(info.value)
    assert info.value.__cause__ is exc


def test_list_devices_with_query_fn_has_no_defaults(defaults):
    defaults(0, 1)  # must be ignored when a query_fn is injected
    assert "*  [" not in list_devices(lambda: FAKE)


# --- format_device_list ---

def test_format_sections_and_lines():
    text = format_device_list(FAKE)
    inputs, outputs = text.split("Output devices:")
    assert inputs.startswith("Input devices (microphones):")
    assert "[ 0] MacBook Pro Microphone  (1 ch, 48000 Hz)" in inputs
    assert "[ 3] USB Mic  (1 ch, 44100 Hz)" in inputs
    assert "Speakers" not in inputs
    assert "[ 1] MacBook Pro Speakers  (2 ch, 48000 Hz)" in outputs
    assert "Microphone" not in outputs
    assert "AirPods" in inputs and "AirPods" in outputs  # duplex device in both
    assert "* = system default" in text


def test_format_marks_defaults():
    lines = format_device_list(FAKE, default_input=3, default_output=1).splitlines()
    starred = [line for line in lines if line.startswith("  *")]
    assert starred == ["  * [ 3] USB Mic  (1 ch, 44100 Hz)",
                       "  * [ 1] MacBook Pro Speakers  (2 ch, 48000 Hz)"]


def test_format_tags_blackhole_without_warning():
    text = format_device_list(FAKE)
    tagged = [line for line in text.splitlines() if "<- virtual mic" in line]
    assert len(tagged) == 2 and all("BlackHole 2ch" in line for line in tagged)
    assert "WARNING" not in text


def test_format_warns_when_blackhole_missing():
    text = format_device_list(NO_BLACKHOLE)
    assert "WARNING: BlackHole virtual mic not found" in text
    assert "brew install blackhole-2ch" in text
    assert "<- virtual mic" not in text


def test_format_empty_sections():
    text = format_device_list([dev(0, "Mic", ins=1)])
    assert text.split("Output devices:")[1].lstrip("\n").startswith("    (none)")


# --- find_device ---

@pytest.mark.parametrize("query, kind, expected", [
    (3, "input", 3),
    ("3", "input", 3),
    (" 2 ", "output", 2),
    ("usb mic", "input", 3),          # exact (case-insensitive) beats substring "USB Mic Pro"
    ("pro", "output", 1),             # unique substring among outputs
    ("blackhole", "output", 2),
    ("airpods", "input", 5),
])
def test_find_device_matches(query, kind, expected):
    assert find_device(query, kind, FAKE) == expected


@pytest.mark.parametrize("query, kind, message", [
    ("mic", "input", "ambiguous"),
    ("pro", "input", "ambiguous"),
    ("Zoom", "input", "No input device matching 'Zoom'"),
    ("Speakers", "input", "No input device matching"),   # exists, but output-only
    (1, "input", "has no input channels"),
    ("0", "output", "has no output channels"),
    (99, "input", "No audio device with index 99"),
])
def test_find_device_errors(query, kind, message):
    with pytest.raises(DeviceError, match=message):
        find_device(query, kind, FAKE)


def test_ambiguous_error_lists_matches():
    with pytest.raises(DeviceError) as info:
        find_device("usb", "input", FAKE)
    assert "[3] USB Mic" in str(info.value) and "[4] USB Mic Pro" in str(info.value)


def test_not_found_error_lists_available_and_hint():
    with pytest.raises(DeviceError) as info:
        find_device("Zoom", "output", FAKE)
    msg = str(info.value)
    assert "Available output devices:" in msg and "[1] MacBook Pro Speakers (2 ch)" in msg
    assert "--list-devices" in msg


@pytest.mark.parametrize("kind, expected", [("input", 0), ("output", 1)])
def test_find_device_none_uses_system_default(defaults, kind, expected):
    defaults(0, 1)
    assert find_device(None, kind, FAKE) == expected


@pytest.mark.parametrize("inp, out", [(None, None), (42, 42), (1, 0)])  # none / unknown / wrong kind
def test_find_device_none_without_usable_default(defaults, inp, out):
    defaults(inp, out)
    with pytest.raises(DeviceError, match="No system default input device"):
        find_device(None, "input", FAKE)


# --- find_blackhole ---

def test_find_blackhole_found():
    assert find_blackhole(FAKE) == 2


def test_find_blackhole_missing():
    with pytest.raises(DeviceError, match="BlackHole virtual mic not found") as info:
        find_blackhole(NO_BLACKHOLE)
    assert "brew install blackhole-2ch" in str(info.value)


def test_find_blackhole_ignores_input_only_blackhole():
    with pytest.raises(DeviceError):
        find_blackhole([dev(0, "BlackHole 2ch", ins=2)])
