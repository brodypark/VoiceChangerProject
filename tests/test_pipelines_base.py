"""pipelines/base.py tests: API error messages, the startup account check, and the event queue.

No network: ApiError objects are built by hand and clients are SimpleNamespace fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from elevenlabs.core.api_error import ApiError
from elevenlabs.core.parse_error import ParsingError

from pipelines.base import (KEY_HINT, MAX_API_MESSAGE_CHARS, ApiFatalError, Pipeline, check_account,
                            describe_api_error)

FAKE_KEY = "FAKE-SECRET-KEY-0000"
HEADERS = {"xi-api-key": FAKE_KEY, "set-cookie": "session=FAKE-COOKIE", "request-id": "req-123"}
PCM_HINT = "try --output-format pcm_24000"


def api_error(status: int | None, body=None) -> ApiError:
    return ApiError(status_code=status, headers=dict(HEADERS), body=body)


def detail(status: str = "", message: str = "", **extra) -> dict:
    return {"detail": {"status": status, "message": message, **extra}}


def assert_no_secrets(message: str) -> None:
    for secret in (FAKE_KEY, "FAKE-COOKIE", "xi-api-key", "set-cookie", "req-123", "headers"):
        assert secret not in message


# --- describe_api_error ---

@pytest.mark.parametrize("exc, expected, fatal", [
    # 401 invalid key
    (api_error(401, detail("invalid_api_key", "Invalid API key")),
     f"ElevenLabs rejected the API key: Invalid API key. {KEY_HINT}", True),
    # 401 missing permission (message and tag forms), and 403
    (api_error(401, detail("missing_permissions", "The API key you used is missing the permission "
                                                  "speech_to_speech to execute this operation.")),
     "ElevenLabs API key is missing a permission: The API key you used is missing the permission "
     f"speech_to_speech to execute this operation.. {KEY_HINT}", True),
    (api_error(401, detail("missing_permissions")),
     f"ElevenLabs API key is missing a permission. {KEY_HINT}", True),
    (api_error(403, {"detail": {"message": "No permission for this voice"}}),
     f"ElevenLabs API key is missing a permission: No permission for this voice. {KEY_HINT}", True),
    (api_error(403, None), f"ElevenLabs rejected the API key. {KEY_HINT}", True),
    # quota: historically 401 + status quota_exceeded; also 402, a "credit" code, or "quota" in the message
    (api_error(401, detail("quota_exceeded", "This request exceeds your quota of 10000.")),
     "Out of ElevenLabs credits: This request exceeds your quota of 10000.. Check your usage at elevenlabs.io.",
     True),
    (api_error(402, detail("payment_required", "Upgrade your plan")),
     "Out of ElevenLabs credits: Upgrade your plan. Check your usage at elevenlabs.io.", True),
    (api_error(402, None), "Out of ElevenLabs credits. Check your usage at elevenlabs.io.", True),
    (api_error(400, {"detail": {"code": "insufficient_credits", "message": "Not enough"}}),
     "Out of ElevenLabs credits: Not enough. Check your usage at elevenlabs.io.", True),
    (api_error(400, {"detail": "Quota exceeded"}),
     "Out of ElevenLabs credits: Quota exceeded. Check your usage at elevenlabs.io.", True),
    # 404 unknown voice
    (api_error(404, detail("voice_not_found", "A voice with voice_id fake-voice was not found.")),
     "Voice not found: A voice with voice_id fake-voice was not found.. Check ELEVENLABS_VOICE_ID in .env.", True),
    # 422 validation list ("body" is dropped from loc; non-dict items are kept as text)
    (api_error(422, {"detail": [
        {"loc": ["body", "model_id"], "msg": "field required", "type": "value_error.missing"},
        {"loc": ["body"], "msg": "audio is empty"},
        {"loc": ["query", "seed"], "message": "not an int"},
        "plain item",
        {"loc": ["body", "x"]},  # no message: just the field name
    ]}),
     "ElevenLabs rejected the request (422): model_id: field required; audio is empty; query.seed: not an int; "
     "plain item; x:.", True),
    # 400 output format / tier problems get the pcm_24000 hint
    (api_error(400, detail("output_format_not_allowed", "pcm_44100 output_format requires the Pro tier.")),
     "ElevenLabs rejected the request (400): pcm_44100 output_format requires the Pro tier.. Your plan may not "
     f"allow this output format; {PCM_HINT}.", True),
    (api_error(422, {"detail": [{"loc": ["query", "output_format"], "msg": "invalid choice"}]}),
     "ElevenLabs rejected the request (422): query.output_format: invalid choice. Your plan may not allow this "
     f"output format; {PCM_HINT}.", True),
    (api_error(400, {"detail": "Your subscription does not include this"}),
     "ElevenLabs rejected the request (400): Your subscription does not include this. Your plan may not allow "
     f"this output format; {PCM_HINT}.", True),
    (api_error(400, {"detail": "bad audio"}), "ElevenLabs rejected the request (400): bad audio.", True),
    # string detail / bare string body / dict without "detail" / unknown status
    (api_error(418, {"detail": "I'm a teapot"}), "ElevenLabs API error (418): I'm a teapot.", False),
    (api_error(418, "  bare\n\n string   body "), "ElevenLabs API error (418): bare string body.", False),
    (api_error(418, {"message": "top-level message", "status": "odd"}),
     "ElevenLabs API error (418): top-level message.", False),
    (api_error(None, None), "ElevenLabs API error (None).", False),
    # rate limit and server errors are retryable
    (api_error(429, detail("too_many_concurrent_requests", "Too many concurrent requests")),
     "ElevenLabs rate/concurrency limit hit: Too many concurrent requests. Slow down or wait a moment.", False),
    (api_error(500, None), "ElevenLabs server error (500). Try again shortly.", False),
    (api_error(503, "<html><body>Service\nUnavailable</body></html>"),
     "ElevenLabs server error (503): <html><body>Service Unavailable</body></html>. Try again shortly.", False),
    (api_error(599, {"detail": {}}), "ElevenLabs server error (599). Try again shortly.", False),
    # the SDK's ParsingError carries the same fields
    (ParsingError(status_code=404, headers=dict(HEADERS), body=detail("voice_not_found", "nope")),
     "Voice not found: nope. Check ELEVENLABS_VOICE_ID in .env.", True),
    # network and anything else
    (httpx.ConnectError("connection refused"),
     "Network problem talking to ElevenLabs (ConnectError: connection refused). Check your connection.", False),
    (httpx.ReadTimeout("timed out"),
     "Network problem talking to ElevenLabs (ReadTimeout: timed out). Check your connection.", False),
    (ConnectionResetError("reset by peer"),
     "Network problem talking to ElevenLabs (ConnectionResetError: reset by peer). Check your connection.", False),
    (TimeoutError("slow"), "Network problem talking to ElevenLabs (TimeoutError: slow). Check your connection.",
     False),
    (ValueError("boom"), "Unexpected error (ValueError: boom).", False),
])
def test_describe_api_error(exc, expected, fatal):
    message, is_fatal = describe_api_error(exc)
    assert message == expected
    assert is_fatal is fatal
    assert_no_secrets(message)


def test_describe_api_error_truncates_huge_body():
    message, fatal = describe_api_error(api_error(502, "x" * 50_000))
    assert not fatal
    assert "x" * MAX_API_MESSAGE_CHARS + "..." in message
    assert "x" * (MAX_API_MESSAGE_CHARS + 1) not in message
    assert len(message) < MAX_API_MESSAGE_CHARS + 100


def test_describe_api_error_truncates_huge_list_detail():
    body = {"detail": [{"loc": ["body", f"f{i}"], "msg": "bad " * 20} for i in range(100)]}
    message, fatal = describe_api_error(api_error(422, body))
    assert fatal
    assert len(message) < MAX_API_MESSAGE_CHARS + 100
    assert message.endswith("....")  # truncation marker + the sentence's full stop


def test_api_error_str_would_leak_headers():
    """Why describe_api_error never uses str(exc): the SDK's __str__ includes the request headers."""
    exc = api_error(401, detail("invalid_api_key", "Invalid API key"))
    assert FAKE_KEY in str(exc)
    assert FAKE_KEY not in describe_api_error(exc)[0]


# --- check_account ---

class FakeVoices:
    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[str] = []

    def get(self, voice_id: str):
        self.calls.append(voice_id)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeSubscription:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = 0

    def get(self):
        self.calls += 1
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def fake_client(voice=None, sub=None) -> SimpleNamespace:
    voice = SimpleNamespace(name="My Test Voice") if voice is None else voice
    sub = SimpleNamespace(character_count=1234, character_limit=10_000, tier="starter") if sub is None else sub
    return SimpleNamespace(voices=FakeVoices(voice), user=SimpleNamespace(subscription=FakeSubscription(sub)))


MISSING_VOICES_READ = api_error(401, detail("missing_permissions", "missing the permission voices_read"))
MISSING_USER_READ = api_error(401, detail("missing_permissions", "missing the permission user_read"))


def test_check_account_success():
    client = fake_client()
    assert check_account(client, "fake-voice-id") == [
        "Voice: My Test Voice",
        "Credits: 1,234 of 10,000 used (8,766 left, starter tier)",
    ]
    assert client.voices.calls == ["fake-voice-id"]
    assert client.user.subscription.calls == 1


def test_check_account_voice_without_name_shows_id_and_overdrawn_credits_clamp_to_zero():
    sub = SimpleNamespace(character_count=12_000, character_limit=10_000, tier="free")
    lines = check_account(fake_client(voice=SimpleNamespace(name=""), sub=sub), "fake-voice-id")
    assert lines == ["Voice: fake-voice-id", "Credits: 12,000 of 10,000 used (0 left, free tier)"]


def test_check_account_missing_permissions_on_both_gives_one_warning():
    client = fake_client(voice=MISSING_VOICES_READ, sub=MISSING_USER_READ)
    lines = check_account(client, "fake-voice-id")
    assert len(lines) == 1
    assert lines[0].startswith("WARNING: can't verify the voice (the API key has no voices_read permission)")
    assert client.user.subscription.calls == 1


def test_check_account_missing_voice_permission_still_reads_credits():
    lines = check_account(fake_client(voice=MISSING_VOICES_READ), "fake-voice-id")
    assert len(lines) == 2
    assert lines[0].startswith("WARNING: can't verify the voice")
    assert lines[1].startswith("Credits: 1,234 of 10,000")


@pytest.mark.parametrize("exc, match", [
    (api_error(404, detail("voice_not_found", "A voice with voice_id fake-voice-id was not found.")),
     "Voice not found"),
    (api_error(401, detail("invalid_api_key", "Invalid API key")), "ElevenLabs rejected the API key"),
    (api_error(401, detail("quota_exceeded", "quota")), "Out of ElevenLabs credits"),
])
def test_check_account_fatal_voice_errors_raise(exc, match):
    client = fake_client(voice=exc)
    with pytest.raises(ApiFatalError, match=match) as info:
        check_account(client, "fake-voice-id")
    assert info.value.__cause__ is None and info.value.__suppress_context__  # `from None`: no header-laden chain
    assert_no_secrets(str(info.value))
    assert client.user.subscription.calls == 0


@pytest.mark.parametrize("exc", [httpx.ConnectError("no route"), ConnectionError("down"), TimeoutError("slow")])
def test_check_account_network_error_warns_and_skips_subscription(exc):
    client = fake_client(voice=exc)
    lines = check_account(client, "fake-voice-id")
    assert len(lines) == 1
    assert lines[0].startswith("WARNING: couldn't verify the voice: Network problem talking to ElevenLabs")
    assert client.user.subscription.calls == 0


def test_check_account_server_error_on_voice_warns_and_still_reads_credits():
    client = fake_client(voice=api_error(500, None))
    lines = check_account(client, "fake-voice-id")
    assert lines[0] == "WARNING: couldn't verify the voice: ElevenLabs server error (500). Try again shortly."
    assert lines[1].startswith("Credits:")


def test_check_account_subscription_error_is_a_warning():
    lines = check_account(fake_client(sub=api_error(503, None)), "fake-voice-id")
    assert lines == ["Voice: My Test Voice",
                     "WARNING: couldn't read the credit balance: ElevenLabs server error (503). Try again shortly."]


def test_check_account_subscription_fatal_error_is_only_a_warning():
    lines = check_account(fake_client(sub=api_error(401, detail("invalid_api_key", "Invalid API key"))), "v")
    assert lines[1].startswith("WARNING: couldn't read the credit balance: ElevenLabs rejected the API key")
    assert_no_secrets(lines[1])


# --- Pipeline.drain_events ---

class DummyPipeline(Pipeline):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def feed_audio(self, frame: bytes) -> None:
        pass


def test_pipeline_initial_state():
    out = []
    p = DummyPipeline(out.append)
    assert p.on_audio_out == out.append
    assert p.error is None and not p.finished.is_set()
    assert p.drain_events() == []


def test_drain_events_returns_in_order_and_empties_queue():
    p = DummyPipeline(lambda pcm: None)
    for line in ("one", "two", "three"):
        p._emit(line)
    p.events.put("four")  # main.run_sts's latency callback puts directly
    assert p.drain_events() == ["one", "two", "three", "four"]
    assert p.drain_events() == []
    p._emit("five")
    assert p.drain_events() == ["five"]


def test_pipeline_is_abstract():
    with pytest.raises(TypeError):
        Pipeline(lambda pcm: None)  # type: ignore[abstract]
