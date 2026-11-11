from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
import re
import threading
import unicodedata

_LOCK = threading.Lock()
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")


@lru_cache(maxsize=2)
def _backend(locale):
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    import espeakng_loader
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    if hasattr(EspeakWrapper, "set_data_path"):
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    return EspeakBackend(language=locale, with_stress=True, preserve_punctuation=False,
                         language_switch="remove-flags")


@dataclass
class WordPhones:
    text: str
    phones: list[str]
    stressed_ipa: str


def tokenize_ipa(ipa, vocab):
    """Split each explicit eSpeak phone separately; never merge across boundaries."""
    tokens = sorted((s for s in vocab if not s.startswith("<") and s != "|"), key=len, reverse=True)
    result = []
    for chunk in unicodedata.normalize("NFC", ipa).replace("ˈ", "").replace("ˌ", "").split():
        chunk = chunk.replace("͡", "").replace("͜", "")
        while chunk:
            token = next((s for s in tokens if chunk.startswith(s)), None)
            if token is None:
                raise ValueError(f"Unsupported IPA sequence: {chunk!r}; add a G2P exception")
            result.append(token)
            chunk = chunk[len(token):]
    return result


def phonemize_words(text, vocab, locale="en-gb", exceptions=None):
    locale = locale.lower()
    if locale not in {"en-gb", "en-us"}:
        raise ValueError("Supported locales: en-gb, en-us")
    if not isinstance(text, str) or len(text) > 300:
        raise ValueError("Reference must be at most 300 characters")
    words = WORD.findall(text)
    residue = WORD.sub("", text)
    if not words or len(words) > 40 or any(c.isalnum() for c in residue):
        raise ValueError("Use 1–40 English words; spell out numbers")
    from phonemizer.separator import Separator
    with _LOCK:
        pronunciations = _backend(locale).phonemize(words, separator=Separator(phone=" ", word=" | ", syllable=""), strip=True)
    if exceptions is None:
        exceptions = json.loads(files("speakstar").joinpath("data/g2p_exceptions.json").read_text(encoding="utf-8"))
    result = []
    for word, ipa in zip(words, pronunciations, strict=True):
        ipa = exceptions.get(locale, {}).get(word.lower().replace("’", "'"), ipa)
        phones = tokenize_ipa(ipa.replace("|", " "), vocab)
        if not phones:
            raise ValueError(f"No pronunciation for {word}")
        result.append(WordPhones(word, phones, ipa))
    return result
