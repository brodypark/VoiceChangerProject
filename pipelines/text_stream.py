"""Turns Scribe realtime transcripts into append-only text for the TTS websocket (Pipeline A).

Scribe v2 Realtime sends, per segment, many `partial_transcript` messages (each the FULL text
of the segment so far, possibly revising earlier words) and then one `committed_transcript`
after ~0.5 s of silence. Text sent to TTS can't be taken back, so TextStabilizer only speaks
words that look settled, to start the voice before a long sentence ends:

    A word in a partial is stable when it is not among the newest `holdback_words` tokens AND
    the same normalised word sat at the same index in the previous partial (two consecutive
    partials agree). Words are only ever emitted by extending what was already sent; if a
    partial contradicts an already-sent word, nothing more is emitted until it agrees again
    or the segment commits.

Normalisation: lowercase, unicode apostrophes mapped to "'", surrounding punctuation stripped,
so "hello" == "Hello," and "don't" == "don’t". Emitted text keeps the latest partial's
original spelling/punctuation (it helps TTS prosody), joined by single spaces with exactly one
trailing space (TTS websocket text chunks should end with a space).

Fillers ("um", "hmm", see pipelines.word_filter.is_filler) and bracketed audio events such as
"(sighs)", "[laughter]" or "(clears throat)" are removed before tokenising, both in partials
and commits. A trailing unclosed "(clears" in a partial is treated as an event in progress.
"""

from __future__ import annotations

import re
import string

from config import PARTIAL_HOLDBACK_WORDS
from pipelines.word_filter import is_filler

# Closed "(...)"/"[...]" spans, plus an unclosed one running to the end (an event still being
# transcribed in a partial).
_EVENT_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]|[(\[][^)\]]*$")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_STRIP_CHARS = string.punctuation + string.whitespace + "“”…¿¡«»"


def _normalise(token: str) -> str:
    return token.translate(_APOSTROPHES).lower().strip(_STRIP_CHARS)


def _words(text: str) -> list[str]:
    """Real word tokens of `text` (original spelling), with events, fillers and bare punctuation removed."""
    tokens = _EVENT_RE.sub(" ", text or "").split()
    return [tok for tok in tokens if _normalise(tok) and not is_filler(tok)]


def _common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _join(words: list[str]) -> str:
    return " ".join(words) + " " if words else ""


class TextStabilizer:
    """Per-segment state machine: feed partials/commits in, get text safe to send to TTS out."""

    def __init__(self, holdback_words: int = PARTIAL_HOLDBACK_WORDS) -> None:
        if holdback_words < 0:
            raise ValueError("holdback_words must be >= 0")
        self.holdback_words = holdback_words
        self._sent: list[str] = []  # original spellings, as emitted
        self._sent_norms: list[str] = []
        self._prev_norms: list[str] = []  # previous partial, normalised

    @property
    def sent_words(self) -> tuple[str, ...]:
        """Words already sent to TTS for the current segment."""
        return tuple(self._sent)

    def reset(self) -> None:
        """Forget the current segment (call between segments or after a reconnect)."""
        self._sent.clear()
        self._sent_norms.clear()
        self._prev_norms = []

    def on_partial(self, text: str) -> str:
        """Return newly stable text to send now (ends with one space), or ""."""
        words = _words(text)
        norms = [_normalise(w) for w in words]
        prev, self._prev_norms = self._prev_norms, norms

        start = len(self._sent_norms)
        # Shorter than what was sent, or a sent word was revised: never re-send, just wait.
        if norms[:start] != self._sent_norms:
            return ""

        limit = len(words) - self.holdback_words
        end = start
        while end < limit and end < len(prev) and norms[end] == prev[end]:
            end += 1

        new = words[start:end]
        self._sent.extend(new)
        self._sent_norms.extend(norms[start:end])
        return _join(new)

    def on_committed(self, text: str) -> str:
        """Return the rest of the segment to speak (ends with one space), or "", then reset.

        The tail starts after the longest common normalised prefix of the committed words and
        the words already sent. If the transcript revised an already-spoken word, the speech
        can't be unsaid, so this is best effort: the listener may hear a word that was later
        corrected, or a word repeated, but never the whole sentence a second time.
        """
        words = _words(text)
        norms = [_normalise(w) for w in words]
        tail = words[_common_prefix_len(norms, self._sent_norms):]
        self.reset()
        return _join(tail)
