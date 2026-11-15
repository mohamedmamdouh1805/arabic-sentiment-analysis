import re
import unicodedata
from dataclasses import dataclass

ARABIC_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
ARABIC_MARKS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
LETTERS = str.maketrans("أإآٱى", "ااااي")
URL = re.compile(r"(?:https?://|www\.)[^\s<>«»،؛؟]+", re.IGNORECASE)
MENTION = re.compile(r"(?<![\w@])@\w+")
EMOJI = re.compile(r"[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f\u200d]")
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def has_arabic(text):
    return any(unicodedata.name(char, "").startswith("ARABIC ")
               for char in ARABIC_LETTER.findall(text))


@dataclass(frozen=True)
class ArabicNormalizer:
    unicode_form: str = "NFKC"
    normalize_letters: bool = True
    remove_tatweel: bool = True
    remove_diacritics: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    keep_punctuation: bool = True
    keep_emoji: bool = True

    def __call__(self, text):
        if not isinstance(text, str):
            raise TypeError("Expected a text string.")
        text = unicodedata.normalize(self.unicode_form, text).casefold()
        if self.remove_urls:
            text = URL.sub(" ", text)
        if self.remove_mentions:
            text = MENTION.sub(" ", text)
        if self.normalize_letters:
            text = text.translate(LETTERS)
        if self.remove_tatweel:
            text = text.replace("ـ", "")
        if self.remove_diacritics:
            text = ARABIC_MARKS.sub("", text)
        if not self.keep_emoji:
            text = EMOJI.sub(" ", text)
        if not self.keep_punctuation:
            text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
        return " ".join(text.split())


def tokenize(text):
    return TOKEN.findall(text)
