"""pipelines/text_stream.py tests: turning Scribe partial/committed transcripts into TTS text."""

from __future__ import annotations

import pytest

from config import PARTIAL_HOLDBACK_WORDS
from pipelines.text_stream import TextStabilizer
from pipelines.word_filter import is_filler


def feed(stab: TextStabilizer, partials: list[str]) -> list[str]:
    return [stab.on_partial(p) for p in partials]


def assert_no_fillers(emitted: str) -> None:
    assert not any(is_filler(tok) for tok in emitted.split()), emitted


# --- realistic growing partials ---

def test_growing_partials_emit_in_order_then_commit_tail():
    stab = TextStabilizer(holdback_words=2)
    partials = [
        "I",
        "I think",
        "I think we",
        "I think we should",
        "I think we should go",
        "I think we should go to",
        "I think we should go to the park",
    ]
    emits = feed(stab, partials)
    assert emits == ["", "", "I ", "think ", "we ", "should ", "go to "]
    assert stab.sent_words == ("I", "think", "we", "should", "go", "to")

    committed = "I think we should go to the park."
    tail = stab.on_committed(committed)
    assert tail == "the park. "
    # Every word exactly once, in order, nothing missing.
    assert "".join(emits) + tail == committed + " "
    assert stab.sent_words == ()


def test_default_holdback_comes_from_config():
    assert TextStabilizer().holdback_words == PARTIAL_HOLDBACK_WORDS


def test_negative_holdback_rejected():
    with pytest.raises(ValueError):
        TextStabilizer(holdback_words=-1)


@pytest.mark.parametrize("holdback, partials, expected", [
    # holdback 0 still needs two partials to agree
    (0, ["hello", "hello there", "hello there friend"], ["", "hello ", "there "]),
    (0, ["hello", "hello"], ["", "hello "]),
    # holdback 1
    (1, ["a b", "a b c", "a b c d"], ["", "a b ", "c "]),
    # holdback larger than the partial: nothing yet
    (5, ["one two", "one two three"], ["", ""]),
    # empty/whitespace partials emit nothing
    (0, ["", "   ", "\n\t"], ["", "", ""]),
])
def test_holdback_and_agreement(holdback, partials, expected):
    assert feed(TextStabilizer(holdback_words=holdback), partials) == expected


# --- revisions ---

def test_revision_of_unsent_word_causes_no_glitch():
    stab = TextStabilizer(holdback_words=1)
    emits = feed(stab, [
        "put it",
        "put it their",
        "put it there now",     # "their" -> "there" before it was stable
        "put it there now please",
    ])
    assert emits == ["", "put it ", "", "there now "]
    tail = stab.on_committed("Put it there now, please.")
    assert tail == "please. "
    assert "their" not in "".join(emits) + tail


def test_revision_of_sent_word_blocks_until_commit():
    stab = TextStabilizer(holdback_words=0)
    assert feed(stab, ["I think so", "I think so"]) == ["", "I think so "]
    # The model now revises "think" (already spoken): nothing more until commit.
    assert feed(stab, ["I thought so too", "I thought so too today"]) == ["", ""]
    assert stab.sent_words == ("I", "think", "so")
    # Best effort: the tail after the common prefix ("I") is spoken, not the whole sentence.
    assert stab.on_committed("I thought so too today.") == "thought so too today. "


def test_revision_that_reverts_resumes_emitting():
    stab = TextStabilizer(holdback_words=0)
    assert feed(stab, ["go home", "go home"]) == ["", "go home "]
    assert stab.on_partial("go Rome now") == ""  # contradicts a sent word
    # Agrees with the sent words again; "now" sat at index 2 in both partials, so it is stable.
    assert stab.on_partial("go home now") == "now "
    assert stab.on_partial("go home now") == ""


def test_shorter_partial_never_resends():
    stab = TextStabilizer(holdback_words=0)
    assert feed(stab, ["one two three", "one two three"]) == ["", "one two three "]
    assert feed(stab, ["one two", "one two", "one two three"]) == ["", "", ""]
    assert stab.on_partial("one two three four") == ""
    assert stab.on_partial("one two three four") == "four "
    assert stab.sent_words == ("one", "two", "three", "four")


def test_commit_shorter_than_sent_returns_nothing():
    stab = TextStabilizer(holdback_words=0)
    feed(stab, ["one two three", "one two three"])
    assert stab.on_committed("One two.") == ""


@pytest.mark.parametrize("first, second, emitted", [
    ("hello world", "Hello, world.", "Hello, world. "),   # punctuation/casing only
    ("I don't know", "I don’t know", "I don’t know "),    # unicode apostrophe
    ("I don’t know", "I don't know", "I don't know "),
    ("“quote” me", "quote me", "quote me "),
])
def test_cosmetic_changes_count_as_agreement_and_use_latest_spelling(first, second, emitted):
    stab = TextStabilizer(holdback_words=0)
    assert feed(stab, [first, second]) == ["", emitted]


def test_commit_matching_sent_words_with_different_punctuation():
    stab = TextStabilizer(holdback_words=0)
    feed(stab, ["hello there", "hello there"])
    assert stab.on_committed("Hello, there! How are you?") == "How are you? "


# --- fillers and audio events ---

@pytest.mark.parametrize("partials, committed, expected_all", [
    (["um so", "um so I uh", "um so I uh think", "um so I uh think it works"],
     "Um, so I, uh, think it works.", "so I think it works."),
    (["(sighs) okay", "(sighs) okay let's", "(sighs) okay let's go now"],
     "(sighs) Okay, let's go now.", "okay let's go now."),
    (["well (clears", "well (clears throat) I", "well (clears throat) I mean it", "well (clears throat) I mean it"],
     "Well (clears throat) I mean it.", "well I mean it."),
    (["[laughter] that", "[laughter] that was hmm", "[laughter] that was hmm funny right"],
     "[laughter] That was, hmm, funny, right?", "that was funny, right?"),
])
def test_fillers_and_events_are_never_spoken(partials, committed, expected_all):
    # Words spoken early keep the partial's spelling; only the commit tail has the final casing.
    stab = TextStabilizer(holdback_words=1)
    emits = feed(stab, partials)
    tail = stab.on_committed(committed)
    spoken = "".join(emits) + tail
    for chunk in [*emits, tail]:
        assert_no_fillers(chunk)
        assert "(" not in chunk and "[" not in chunk
    assert spoken.strip() == expected_all


def test_um_so_i_uh_think_speaks_so_i_think():
    stab = TextStabilizer(holdback_words=0)
    assert feed(stab, ["um so I uh think", "um so I uh think"]) == ["", "so I think "]


@pytest.mark.parametrize("committed", ["Hmm.", "uh, um... mm-hmm", "(sighs)", "[laughter] haha", "", "  ", "..."])
def test_commit_with_no_real_words_returns_empty(committed):
    assert TextStabilizer().on_committed(committed) == ""


def test_filler_only_partials_emit_nothing():
    stab = TextStabilizer(holdback_words=0)
    assert feed(stab, ["hmm", "hmm uh", "hmm uh (sighs)"]) == ["", "", ""]


# --- formatting ---

@pytest.mark.parametrize("partials", [
    ["a  b\tc", "a b   c d e"],
    ["  spaced   out  words  ", "spaced out words here"],
])
def test_emits_are_single_spaced_with_one_trailing_space(partials):
    stab = TextStabilizer(holdback_words=0)
    for out in [*feed(stab, partials), stab.on_committed(partials[-1] + " end")]:
        if out:
            assert out.endswith(" ") and not out.endswith("  ")
            assert "  " not in out and not out.startswith(" ")


# --- segments ---

def test_consecutive_segments_are_independent():
    stab = TextStabilizer(holdback_words=0)
    assert feed(stab, ["first one", "first one"]) == ["", "first one "]
    assert stab.on_committed("First one.") == ""
    # The next segment starts fresh: no carried-over prefix or previous partial.
    assert stab.sent_words == ()
    assert stab.on_partial("second") == ""
    assert stab.on_partial("second") == "second "
    assert stab.on_committed("Second segment.") == "segment. "
    assert stab.on_committed("Third, never partial.") == "Third, never partial. "


def test_reset_clears_state():
    stab = TextStabilizer(holdback_words=0)
    feed(stab, ["hello world", "hello world"])
    assert stab.sent_words == ("hello", "world")
    stab.reset()
    assert stab.sent_words == ()
    assert stab.on_partial("hello world") == ""  # previous partial was forgotten too
    assert stab.on_partial("hello world") == "hello world "


def test_property_emitted_text_never_contains_filler():
    fillers = ["um", "uh", "hmm", "Mm-hmm,", "Uh.", "ahh", "(sighs)", "[laughter]"]
    real = ["so", "we", "should", "probably", "head", "out", "soon"]
    stab = TextStabilizer(holdback_words=1)
    partial: list[str] = []
    out: list[str] = []
    for i, word in enumerate(real):
        partial += [fillers[i % len(fillers)], word]
        text = " ".join(partial)
        out += [stab.on_partial(text), stab.on_partial(text)]
    out.append(stab.on_committed(" ".join(partial) + " uh"))
    for chunk in out:
        assert_no_fillers(chunk)
    assert "".join(out).strip() == " ".join(real)
