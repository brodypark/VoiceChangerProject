"""Word check for Pipeline B: does a phrase contain real words, or only "hmm", "uh", a sigh?

The phrase is transcribed with ElevenLabs batch speech-to-text (Scribe v2), uploaded as
raw 16 kHz mono int16 LE PCM (file_format="pcm_s16le_16", same bytes as the STS upload).
Scribe returns a list of words, each with a `type`:
    "word"         spoken words, INCLUDING fillers: a spoken "Hmmmm." comes back as ('Hmm', 'word')
    "spacing"      whitespace between words
    "audio_event"  non-speech sounds such as "(sighs)" or "(laughs)" (tag_audio_events=True)
So audio events and spacing are ignored, and "word" items are checked against a filler list.

Filler list (case-insensitive, after stripping surrounding punctuation; letters may repeat,
so "hmmmm", "uhhh", "ummm", "ahhh", "ooh" all match):
    hm/hmm, hmph, mm/mmm, mhm/mmhmm, uh, um, uhm, er/err, erm, ah, aah, oh, ooh, eh,
    huh, ugh, phew, ahem, heh, haha/hahaha (laughter), uhhuh.
Hyphenated tokens are filler only if every part is: "uh-huh", "mm-hmm" are fillers,
"um-brella" is not. Short real words such as "a", "I", "no", "yes", "ok", "hi" are never
fillers. "Oh" on its own is a filler, but "Oh no" has a real word, so it is played.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Iterable

from config import LANGUAGE_CODE, STT_BATCH_MODEL_ID

INPUT_FILE_FORMAT = "pcm_s16le_16"  # raw 16-bit 16 kHz mono LE PCM, as emitted by the segmenter
STT_REQUEST_TIMEOUT_S = 10  # HTTP timeout for one check (int: the SDK's RequestOptions wants an int)

# One alternative per filler; `+` allows elongated forms ("hmmmm", "uhhh").
_FILLER_PATTERNS = (
    r"h+m+",            # hm, hmm, hmmm
    r"h+m+p+h+",        # hmph
    r"m{2,}",           # mm, mmm (a lone "m" is left alone)
    r"m+h+m+",          # mhm, mmhmm, mhmm
    r"u+h+",            # uh, uhh
    r"u+m+",            # um, umm
    r"u+h+m+",          # uhm
    r"e+r+",            # er, err
    r"e+r+m+",          # erm
    r"a+h+",            # ah, ahh, aah
    r"o+h+",            # oh, ohh, ooh
    r"e+h+",            # eh
    r"h+u+h+",          # huh
    r"u+g+h+",          # ugh
    r"p+h+e+w+",        # phew
    r"a+h+e+m+",        # ahem
    r"h+e+h+",          # heh
    r"(?:h+a+){2,}h*",  # haha, hahaha (laughter transcribed as a word)
    r"u+h+h+u+h+",      # uhhuh (written without the hyphen)
)
_FILLER_RE = re.compile(r"(?:" + "|".join(_FILLER_PATTERNS) + r")")
_HYPHENS_RE = re.compile(r"[-‐‑‒–—]")  # ASCII hyphen and unicode dashes
_STRIP_CHARS = string.punctuation + string.whitespace + "‘’“”…¿¡"
_EVENT_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")  # "(sighs)", "[laughter]" in plain transcripts


def _normalise(token: str) -> str:
    return token.lower().strip(_STRIP_CHARS)


def is_filler(token: str) -> bool:
    """True if `token` is a non-word sound such as "Hmm." or "uh-huh" (see the module docstring)."""
    norm = _normalise(token)
    if not norm:
        return False
    parts = [_normalise(part) for part in _HYPHENS_RE.split(norm)]
    return all(part and _FILLER_RE.fullmatch(part) for part in parts)


def _field(item: Any, name: str) -> Any:
    """Read `name` from an SDK object or a dict."""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _is_real_token(token: str) -> bool:
    return bool(_normalise(token)) and not is_filler(token)


def has_real_words(words: Iterable[Any] | None, text: str = "") -> bool:
    """True if any "word" item is a real word, not a filler.

    `words` are Scribe word items (SDK objects or dicts with "text" and "type"); items with no
    type count as words. If `words` is empty or None, `text` (the plain transcript) is split
    on whitespace instead, with parenthesised/bracketed audio events like "(sighs)" removed.
    """
    items = list(words or [])
    if not items:
        return any(_is_real_token(tok) for tok in _EVENT_RE.sub(" ", text or "").split())
    for item in items:
        if (_field(item, "type") or "word") != "word":
            continue  # "spacing" or "audio_event"
        if any(_is_real_token(tok) for tok in str(_field(item, "text") or "").split()):
            return True
    return False


@dataclass(frozen=True)
class WordVerdict:
    has_words: bool
    text: str  # the transcript, for messages ("" if nothing was heard)


class WordChecker:
    """Transcribes one phrase with batch speech-to-text and decides whether it has real words.

    check() is synchronous and raises the SDK's errors (ApiError, httpx errors); the caller
    classifies them. No retries: a failed check simply lets the phrase play.
    """

    def __init__(self, client: Any, model_id: str = STT_BATCH_MODEL_ID,
                 language_code: str = LANGUAGE_CODE) -> None:
        self._client = client
        self.model_id = model_id
        self.language_code = language_code

    def check(self, phrase: bytes) -> WordVerdict:
        result = self._client.speech_to_text.convert(
            model_id=self.model_id,
            file=("phrase.pcm", phrase, "application/octet-stream"),
            file_format=INPUT_FILE_FORMAT,
            tag_audio_events=True,
            language_code=self.language_code,
            request_options={"timeout_in_seconds": STT_REQUEST_TIMEOUT_S, "max_retries": 0},
        )
        text = str(_field(result, "text") or "").strip()
        return WordVerdict(has_words=has_real_words(_field(result, "words"), text), text=text)
