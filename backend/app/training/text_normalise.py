"""Tweet-style text normalisation shared by training and inference.

Invariant: this function is applied BEFORE the sentiment pipeline at train
time and BEFORE the sentiment pipeline at serve time. The bundle stores the
import path of this function so the two sides can never silently drift apart.

Conventions follow TweetEval (Barbiri et al., arXiv:2010.12421), the corpus
behind D1: URLs become the literal token `http`, mentions become `@user`,
hashtag markers are dropped while the hashtag word is kept, and leading `RT`
is removed.
"""

from __future__ import annotations

import re
from typing import Iterable

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@[\w]{1,40}")
_RT_RE = re.compile(r"^\s*rt[:\s]+", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#(\w+)")
_ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")
_WS_RE = re.compile(r"\s+")

PREPROCESS_ID = "app.training.text_normalise.normalise_tweet"


def normalise_tweet(text: str) -> str:
    """Normalise one raw tweet/message. Raises on non-string input."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")

    out = text.replace("\x00", " ")
    out = _RT_RE.sub("", out)
    out = _URL_RE.sub(" http ", out)
    out = _MENTION_RE.sub(" @user ", out)
    out = _HASHTAG_RE.sub(r" \1 ", out)
    out = _ZERO_WIDTH_RE.sub(" ", out)
    out = _WS_RE.sub(" ", out).strip()
    return out


def normalise_all(texts: Iterable[str]) -> list[str]:
    """Normalise a sequence of raw texts, preserving order."""
    return [normalise_tweet(t) for t in texts]
