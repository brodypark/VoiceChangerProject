"""pipelines/word_filter.py tests: the filler list, has_real_words, and WordChecker. No network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from config import LANGUAGE_CODE, STT_BATCH_MODEL_ID
from pipelines.word_filter import WordChecker, WordVerdict, has_real_words, is_filler


def w(text: str, type_: str = "word") -> SimpleNamespace:
    """A Scribe word item, shaped like the SDK's SpeechToTextWordResponseModel."""
    return SimpleNamespace(text=text, type=type_)


# --- is_filler ---

@pytest.mark.parametrize("token", [
    "hm", "hmm", "Hmm", "Hmm.", "hmmmm", "HMMM!", "hmph",
    "mm", "mmm", "Mmm...", "mhm", "mmhmm", "mhmm",
    "uh", "Uh,", "uhh", "uhhh", "um", "Um.", "ummm", "uhm",
    "er", "err", "erm", "ah", "Ahh!", "aah", "oh", "Oh.", "ohh", "ooh", "eh", "Eh?",
    "huh", "Huh?", "ugh", "Ugh.", "phew", "ahem", "heh", "haha", "Hahaha!",
    "uh-huh", "Uh-huh.", "mm-hmm", "Mm-hmm,", "uh–huh", "uhhuh", "“Hmm”", "(hmm)",
])
def test_fillers(token):
    assert is_filler(token)


@pytest.mark.parametrize("token", [
    "a", "A", "I", "no", "No.", "yes", "ok", "okay", "hi", "Hi!", "hey", "ha",
    "um-brella", "umbrella", "uh-huh-yes", "mm-hmm-ok", "her", "hum", "human", "ohio", "huge", "m",
    "oh-no", "the", "error", "", "...", "-", "--",
])
def test_not_fillers(token):
    assert not is_filler(token)


# --- has_real_words ---

def test_real_sentence():
    words = [w("Hello"), w(" ", "spacing"), w("there.")]
    assert has_real_words(words)


def test_filler_only():
    assert not has_real_words([w("Hmm")])
    assert not has_real_words([w("Um,"), w(" ", "spacing"), w("uh..."), w(" ", "spacing"), w("mm-hmm")])


def test_oh_inside_a_sentence_counts():
    assert has_real_words([w("Oh"), w(" ", "spacing"), w("no")])
    assert has_real_words([w("Um"), w(" ", "spacing"), w("I")])


def test_audio_events_are_ignored():
    assert not has_real_words([w("(sighs)", "audio_event")])
    assert not has_real_words([w("(laughs)", "audio_event"), w(" ", "spacing"), w("Hmm")])
    assert has_real_words([w("(sighs)", "audio_event"), w(" ", "spacing"), w("fine")])


def test_dict_items():
    assert has_real_words([{"text": "Yes", "type": "word"}])
    assert not has_real_words([{"text": "uh", "type": "word"}, {"text": "(sighs)", "type": "audio_event"}])


def test_item_without_type_counts_as_word():
    assert has_real_words([SimpleNamespace(text="hello")])
    assert has_real_words([{"text": "hello"}])


def test_empty_word_items_are_not_words():
    assert not has_real_words([w(""), w("  "), w("..."), w("", "spacing")])


@pytest.mark.parametrize("words", [None, []])
def test_falls_back_to_text_when_no_words(words):
    assert has_real_words(words, "Hello there.")
    assert not has_real_words(words, "Hmm.")
    assert not has_real_words(words, "(sighs) Uh-huh. [laughter]")
    assert has_real_words(words, "(sighs) okay")
    assert not has_real_words(words, "")
    assert not has_real_words(words)


# --- WordChecker ---

class FakeStt:
    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[dict] = []

    def convert(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def test_checker_request_and_verdict():
    stt = FakeStt(SimpleNamespace(text=" Hello there. ", words=[w("Hello"), w(" ", "spacing"), w("there.")]))
    checker = WordChecker(SimpleNamespace(speech_to_text=stt))
    assert checker.check(b"\x00\x01" * 10) == WordVerdict(has_words=True, text="Hello there.")

    (call,) = stt.calls
    assert call["model_id"] == STT_BATCH_MODEL_ID == "scribe_v2"
    assert call["language_code"] == LANGUAGE_CODE == "en"
    assert call["file"] == ("phrase.pcm", b"\x00\x01" * 10, "application/octet-stream")
    assert call["file_format"] == "pcm_s16le_16"
    assert call["tag_audio_events"] is True
    assert call["request_options"]["max_retries"] == 0


def test_checker_filler_verdict():
    stt = FakeStt(SimpleNamespace(text="Hmm", words=[w("Hmm")]))
    verdict = WordChecker(SimpleNamespace(speech_to_text=stt), model_id="m", language_code="fr").check(b"")
    assert verdict == WordVerdict(has_words=False, text="Hmm")
    assert (stt.calls[0]["model_id"], stt.calls[0]["language_code"]) == ("m", "fr")


def test_checker_handles_missing_text_and_words():
    stt = FakeStt(SimpleNamespace(text=None, words=None))
    assert WordChecker(SimpleNamespace(speech_to_text=stt)).check(b"") == WordVerdict(False, "")


def test_checker_raises_sdk_errors():
    stt = FakeStt(ConnectionError("down"))
    with pytest.raises(ConnectionError):
        WordChecker(SimpleNamespace(speech_to_text=stt)).check(b"")
