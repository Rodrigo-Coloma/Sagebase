"""Light text normalization shared across parsers."""

from __future__ import annotations

import re
import unicodedata

# Common ligatures and OCR artifacts seen in biomedical PDFs.
_LIGATURES = str.maketrans({
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
})

_HYPHEN_LINEBREAK = re.compile(r"-\n(?=\w)")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def normalize_text(text: str) -> str:
    """Repair line-break hyphenation, collapse whitespace, NFKC normalize."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_LIGATURES)
    text = _HYPHEN_LINEBREAK.sub("", text)        # de-hyphenate "rapamy-\ncin"
    text = _TRAILING_WS.sub("\n", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
