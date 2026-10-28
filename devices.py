"""Audio device listing and selection (mic input, BlackHole virtual-mic output).

Devices are plain dicts with keys:
    index, name, max_input_channels, max_output_channels, default_samplerate

Every function that touches hardware accepts an injectable device list or query
function, so tests can run with fake devices and no audio hardware.
"""

from __future__ import annotations

from typing import Callable, Literal, Mapping, Sequence

import sounddevice as sd

BLACKHOLE_NAME = "BlackHole"
BLACKHOLE_INSTALL_HINT = (
    "Install it with:  brew install blackhole-2ch\n"
    "Then reboot or log out and back in so macOS loads the new audio driver."
)
MIC_PERMISSION_HINT = (
    "If this is a permissions problem, open System Settings > Privacy & Security > Microphone,\n"
    "enable the app you run this from (Terminal, VS Code, or iTerm), then quit and restart that app."
)
LIST_DEVICES_HINT = "Run `python main.py --list-devices` to see all devices."

Kind = Literal["input", "output"]


class DeviceError(Exception):
    """Raised for missing/invalid audio devices, with a user-facing message."""


def get_devices(query_fn: Callable[[], Sequence[Mapping]] | None = None) -> list[dict]:
    """Return all audio devices as plain dicts. `query_fn` defaults to sounddevice.query_devices."""
    query = query_fn or sd.query_devices
    try:
        raw = query()
    except (sd.PortAudioError, OSError) as exc:
        raise DeviceError(f"Could not query audio devices: {exc}\n{MIC_PERMISSION_HINT}") from exc

    devices = []
    for position, dev in enumerate(raw):
        devices.append({
            "index": int(dev.get("index", position)),
            "name": str(dev["name"]),
            "max_input_channels": int(dev.get("max_input_channels", 0)),
            "max_output_channels": int(dev.get("max_output_channels", 0)),
            "default_samplerate": float(dev.get("default_samplerate", 0.0)),
        })
    return devices


def system_defaults() -> tuple[int | None, int | None]:
    """Return the system (input, output) default device indices; None where there is no default."""
    try:
        pair = sd.default.device
        indices = (pair[0], pair[1])
    except (sd.PortAudioError, OSError, TypeError, IndexError):
        return None, None
    # sounddevice uses -1 (or None) for "no default device".
    return tuple(i if isinstance(i, int) and i >= 0 else None for i in indices)  # type: ignore[return-value]


def _channels(device: Mapping, kind: Kind) -> int:
    return device["max_input_channels"] if kind == "input" else device["max_output_channels"]


def _of_kind(devices: list[dict], kind: Kind) -> list[dict]:
    return [d for d in devices if _channels(d, kind) > 0]


def _is_blackhole(device: Mapping) -> bool:
    return BLACKHOLE_NAME.lower() in device["name"].lower()


def _describe(device: Mapping, kind: Kind) -> str:
    return f"[{device['index']}] {device['name']} ({_channels(device, kind)} ch)"


def format_device_list(
    devices: list[dict], default_input: int | None = None, default_output: int | None = None
) -> str:
    """Human-readable device list with input/output sections, defaults marked `*`, BlackHole tagged."""
    lines: list[str] = []
    sections: tuple[tuple[str, Kind, int | None], ...] = (
        ("Input devices (microphones)", "input", default_input),
        ("Output devices", "output", default_output),
    )
    for title, kind, default in sections:
        lines.append(f"{title}:")
        matching = _of_kind(devices, kind)
        if not matching:
            lines.append("    (none)")
        for dev in matching:
            mark = "*" if dev["index"] == default else " "
            line = (f"  {mark} [{dev['index']:>2}] {dev['name']}  "
                    f"({_channels(dev, kind)} ch, {dev['default_samplerate']:.0f} Hz)")
            if _is_blackhole(dev):
                line += "  <- virtual mic"
            lines.append(line)
        lines.append("")

    lines.append("* = system default")
    if not any(_is_blackhole(d) for d in devices):
        lines.append("")
        lines.append("WARNING: BlackHole virtual mic not found. Apps won't be able to hear the AI voice.")
        lines.append(BLACKHOLE_INSTALL_HINT)
    return "\n".join(lines)


def list_devices(query_fn: Callable[[], Sequence[Mapping]] | None = None) -> str:
    """Formatted device list. System defaults are only marked when querying real hardware."""
    devices = get_devices(query_fn)
    default_input, default_output = system_defaults() if query_fn is None else (None, None)
    return format_device_list(devices, default_input, default_output)


def _available(devices: list[dict], kind: Kind) -> str:
    candidates = _of_kind(devices, kind)
    if not candidates:
        return f"No {kind} devices found."
    listed = "\n".join(f"  {_describe(d, kind)}" for d in candidates)
    return f"Available {kind} devices:\n{listed}"


def find_device(query: str | int | None, kind: Kind, devices: list[dict] | None = None) -> int:
    """Resolve a device query to an index.

    None -> system default; int or digit string -> that index;
    otherwise case-insensitive name match (exact beats substring).
    """
    if devices is None:
        devices = get_devices()
    by_index = {d["index"]: d for d in devices}

    if query is None:
        default_input, default_output = system_defaults()
        default = default_input if kind == "input" else default_output
        if default is None or default not in by_index or _channels(by_index[default], kind) == 0:
            raise DeviceError(
                f"No system default {kind} device. Pick one explicitly.\n"
                f"{_available(devices, kind)}\n{LIST_DEVICES_HINT}"
            )
        return default

    if isinstance(query, int) or query.strip().isdigit():
        index = int(query)
        device = by_index.get(index)
        if device is None:
            raise DeviceError(f"No audio device with index {index}.\n{_available(devices, kind)}\n{LIST_DEVICES_HINT}")
        if _channels(device, kind) == 0:
            raise DeviceError(
                f"Device [{index}] {device['name']} has no {kind} channels.\n"
                f"{_available(devices, kind)}\n{LIST_DEVICES_HINT}"
            )
        return index

    needle = query.strip().lower()
    candidates = _of_kind(devices, kind)
    exact = [d for d in candidates if d["name"].lower() == needle]
    if exact:
        return exact[0]["index"]

    matches = [d for d in candidates if needle in d["name"].lower()]
    if len(matches) == 1:
        return matches[0]["index"]
    if len(matches) > 1:
        listed = "\n".join(f"  {_describe(d, kind)}" for d in matches)
        raise DeviceError(
            f"{kind.capitalize()} device {query!r} is ambiguous; it matches:\n{listed}\n"
            "Use a longer name or the device index."
        )
    raise DeviceError(f"No {kind} device matching {query!r}.\n{_available(devices, kind)}\n{LIST_DEVICES_HINT}")


def find_blackhole(devices: list[dict] | None = None) -> int:
    """Index of the BlackHole output device (the virtual mic)."""
    if devices is None:
        devices = get_devices()
    for dev in _of_kind(devices, "output"):
        if _is_blackhole(dev):
            return dev["index"]
    raise DeviceError(f"BlackHole virtual mic not found.\n{BLACKHOLE_INSTALL_HINT}")
