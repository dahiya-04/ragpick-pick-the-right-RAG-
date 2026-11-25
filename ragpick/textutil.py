"""Small text helpers shared by every stage."""

from __future__ import annotations

import hashlib
import json
import re

STOPWORDS = frozenset(
    """a an and are as at be been by for from has have in is it its of on or that the
    this to was were will with which what when who whom how shall should may can any
    all such under per not no do does did if than then there these those their they
    under into also other""".split()
)

_ALNUM = re.compile(r"[a-z0-9]+")
# Keeps identifiers like "4.2", "2023-24" and "rbi/2023-24/53" together.
_COMPOUND = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")

_ABBREV = {
    "no", "nos", "rs", "viz", "i.e", "e.g", "etc", "sr", "dr", "ltd", "co", "para",
    "paras", "vol", "art", "sec", "cl", "govt", "dept", "mr", "ms", "st", "w.e.f",
    "inr", "fig", "ref", "approx", "pvt", "u/s", "vs",
}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'‘“])")


def words(text: str) -> list[str]:
    """Lower-cased alphanumeric words."""
    return _ALNUM.findall(text.lower())


def normalize(text: str) -> str:
    return " ".join(words(text))


def count_words(text: str) -> int:
    return len(text.split())


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS]


def bm25_tokens(text: str) -> list[str]:
    """Tokens for keyword search: compound identifiers plus their parts."""
    out: list[str] = []
    for tok in _COMPOUND.findall(text.lower()):
        out.append(tok)
        if any(c in tok for c in "./-"):
            out.extend(p for p in re.split(r"[./-]", tok) if p)
    return [t for t in out if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter that tolerates legal abbreviations (No., Rs., para.)."""
    sentences: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        if not para:
            continue
        pieces = _SENT_SPLIT.split(para)
        buf = ""
        for piece in pieces:
            buf = f"{buf} {piece}".strip() if buf else piece
            last = buf.rstrip(".").split(" ")[-1].lower() if buf else ""
            # Do not end a sentence on an abbreviation, a bare number ("4.") or a single letter.
            if last in _ABBREV or re.fullmatch(r"\(?[0-9a-z]{1,2}\)?", last) or count_words(buf) < 3:
                continue
            sentences.append(buf)
            buf = ""
        if buf:
            if sentences and count_words(buf) < 3:
                sentences[-1] = f"{sentences[-1]} {buf}"
            else:
                sentences.append(buf)
    return sentences


def stable_hash(obj) -> str:
    data = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def shorten(text: str, n: int = 300) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"
