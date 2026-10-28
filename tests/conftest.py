"""Shared fixtures. Nothing here touches audio hardware, the network, or the real .env."""

from __future__ import annotations

from pathlib import Path

import pytest
import sounddevice as sd

ENV_VARS = ("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "VOICE_CHANGER_MODE")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove config env vars (load_dotenv uses override=False, so they'd win otherwise)."""
    for name in ENV_VARS:
        # setenv first so monkeypatch records the original state; otherwise values that
        # load_dotenv writes into os.environ during the test would leak into later tests.
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    return monkeypatch


@pytest.fixture
def env_file(tmp_path: Path, clean_env: pytest.MonkeyPatch):
    """Factory: write a fake .env in tmp_path and return its path."""

    def write(text: str = "") -> Path:
        path = tmp_path / ".env"
        path.write_text(text)
        return path

    return write


class FakeStream:
    """Stands in for sounddevice.RawInputStream/RawOutputStream: records calls, never opens hardware."""

    def __init__(self, fake_sd: FakeSounddevice, kind: str, kwargs: dict) -> None:
        self.fake_sd = fake_sd
        self.kind = kind
        self.kwargs = kwargs
        self.device = kwargs.get("device")
        self.started = self.stopped = self.closed = False

    def start(self) -> None:
        if (self.kind, self.device) in self.fake_sd.fail_start:
            raise sd.PortAudioError("start failed")
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        if (self.kind, self.device) in self.fake_sd.fail_stop:
            raise sd.PortAudioError("stop failed")

    def close(self) -> None:
        self.closed = True


class FakeSounddevice:
    """Controls the fake streams. Failures are keyed by (kind, device), kind = "input" | "output"."""

    def __init__(self) -> None:
        self.streams: list[FakeStream] = []
        self.fail_open: dict[tuple[str, int | None], Exception] = {}
        self.fail_start: set[tuple[str, int | None]] = set()
        self.fail_stop: set[tuple[str, int | None]] = set()

    def opener(self, kind: str):
        def open_stream(**kwargs) -> FakeStream:
            error = self.fail_open.get((kind, kwargs.get("device")))
            if error is not None:
                raise error
            stream = FakeStream(self, kind, kwargs)
            self.streams.append(stream)
            return stream

        return open_stream

    def of_kind(self, kind: str) -> list[FakeStream]:
        return [s for s in self.streams if s.kind == kind]


@pytest.fixture
def fake_sd(monkeypatch: pytest.MonkeyPatch) -> FakeSounddevice:
    """Replace sounddevice's raw stream classes so start()/stop() never touch audio hardware."""
    fake = FakeSounddevice()
    monkeypatch.setattr(sd, "RawInputStream", fake.opener("input"))
    monkeypatch.setattr(sd, "RawOutputStream", fake.opener("output"))
    return fake
