"""Shared, deterministic text cleaning that preserves sentiment negation."""

from __future__ import annotations

import html
import re


PREPROCESSING_VERSION = "1.0"
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
_NOISE_RE = re.compile(r"[^a-z0-9\s'!?-]")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCTUATION_RE = re.compile(r"([!?])\1+")


def clean_text(value: object) -> str:
    """Normalize one text value without removing words such as not/no/never."""
    if value is None:
        return ""
    text = html.unescape(str(value)).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _NOISE_RE.sub(" ", text)
    text = _REPEATED_PUNCTUATION_RE.sub(r"\1", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_many(values: list[object]) -> list[str]:
    return [clean_text(value) for value in values]
