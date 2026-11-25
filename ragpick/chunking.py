"""Chunking strategies. Sizes are in words (1 word is roughly 1.3 tokens for English legal text)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .textutil import count_words, split_sentences


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str          # body only; used for gold-evidence matching
    index_text: str    # what retrievers see (body, optionally with a context header)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers

_SENT = "<sentence>"


def _window(text: str, size: int, overlap: int) -> list[str]:
    toks = text.split()
    if not toks:
        return []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(toks), step):
        out.append(" ".join(toks[start:start + size]))
        if start + size >= len(toks):
            break
    return out


def _split_rec(text: str, size: int, seps: tuple) -> list[str]:
    if count_words(text) <= size:
        return [text]
    if not seps:
        return _window(text, size, 0)
    sep, rest = seps[0], seps[1:]
    parts = split_sentences(text) if sep == _SENT else text.split(sep)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return _split_rec(text, size, rest)
    out = []
    for p in parts:
        out.extend(_split_rec(p, size, rest) if count_words(p) > size else [p])
    return out


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks, cur, n = [], [], 0
    overlap = min(overlap, size // 2)
    for p in pieces:
        w = count_words(p)
        if cur and n + w > size:
            chunks.append("\n".join(cur))
            tail, t = [], 0
            for q in reversed(cur):
                qw = count_words(q)
                if t + qw > overlap:
                    break
                tail.insert(0, q)
                t += qw
            cur, n = tail, t
        cur.append(p)
        n += w
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def recursive_split(text: str, size: int, overlap: int = 0) -> list[str]:
    return _merge(_split_rec(text, size, ("\n\n", "\n", _SENT, " ")), size, overlap)


# --------------------------------------------------------------------------- base class


class Chunker:
    def __init__(self, name: str, context_header: bool = False, **params):
        self.name = name
        self.context_header = context_header
        self.params = params

    def split(self, doc) -> list[tuple[str, dict]]:
        raise NotImplementedError

    def chunk_corpus(self, docs) -> list[Chunk]:
        chunks = []
        for doc in docs:
            for i, (text, extra) in enumerate(self.split(doc)):
                text = text.strip()
                if not text:
                    continue
                meta = {
                    "title": doc.title,
                    "date": doc.meta.get("date", ""),
                    "circular_no": doc.meta.get("circular_no") or doc.meta.get("ref_no", ""),
                    "superseded": bool(doc.meta.get("superseded", False)),
                    "position": i,
                    **extra,
                }
                index_text = text
                if self.context_header:
                    header = " | ".join(str(x) for x in (doc.title, meta["circular_no"], meta["date"],
                                                         meta.get("section", "")) if x)
                    index_text = f"{header}\n{text}"
                chunks.append(Chunk(f"{self.name}:{doc.doc_id}:{i}", doc.doc_id, text, index_text, meta))
        return chunks


class FixedChunker(Chunker):
    """Sliding window of `size` words with `overlap` words of overlap."""

    def split(self, doc):
        size, overlap = self.params.get("size", 400), self.params.get("overlap", 50)
        return [(t, {}) for t in _window(doc.text, size, overlap)]


class RecursiveChunker(Chunker):
    """Split on paragraphs, then lines, then sentences, then words; pack up to `size` words."""

    def split(self, doc):
        size, overlap = self.params.get("size", 400), self.params.get("overlap", 0)
        return [(t, {}) for t in recursive_split(doc.text, size, overlap)]


_HEADING = (
    r"^\s*(?:CHAPTER|Chapter|PART|Part|SECTION|Section|ANNEX(?:URE)?|Annex(?:ure)?|APPENDIX|Appendix|"
    r"SCHEDULE|Schedule)\b[^\n]{0,120}$"
)
_PARA = r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+[A-Z(\"'‘“]"


class StructureChunker(Chunker):
    """Split on the document's own numbered paragraphs and headings (para 4, 4.2, Chapter III, Annex).

    Tiny sections are merged forward; oversized ones are split recursively but keep their
    section label, so every chunk knows which para it came from.
    """

    def split(self, doc):
        max_size = self.params.get("max_size", 600)
        min_size = self.params.get("min_size", 40)
        heading_re = re.compile(self.params.get("heading_pattern", _HEADING))
        para_re = re.compile(self.params.get("para_pattern", _PARA))

        segments: list[list] = [["preamble", "", []]]  # [section, heading, lines]
        heading = ""
        for line in doc.text.splitlines():
            if heading_re.match(line):
                heading = line.strip()[:120]
                segments.append([heading.split()[0] + " " + " ".join(heading.split()[1:2]), heading, [line]])
                continue
            m = para_re.match(line)
            if m:
                segments.append([m.group(1), heading, [line]])
            else:
                segments[-1][2].append(line)

        merged: list[list] = []
        carry = None
        for sec, head, lines in segments:
            text = "\n".join(lines).strip()
            if not text:
                continue
            if carry:
                text = carry[2] + "\n" + text
                sec = carry[0]
                carry = None
            if count_words(text) < min_size:
                carry = [sec, head, text]
                continue
            merged.append([sec, head, text])
        if carry:
            if merged:
                merged[-1][2] += "\n" + carry[2]
            else:
                merged.append(carry)

        out = []
        for sec, head, text in merged:
            parts = recursive_split(text, max_size) if count_words(text) > max_size else [text]
            for p in parts:
                out.append((p, {"section": sec, "heading": head}))
        return out


class SemanticChunker(Chunker):
    """Break between sentences where embedding similarity drops (above the given distance percentile)."""

    def __init__(self, name, embedder=None, **params):
        super().__init__(name, **params)
        if embedder is None:
            raise ValueError("semantic chunker needs `embedder:` naming an entry in `embedders`")
        self.embedder = embedder

    def split(self, doc):
        pct = self.params.get("breakpoint_percentile", 90)
        max_size = self.params.get("max_size", 600)
        min_size = self.params.get("min_size", 50)
        sents = split_sentences(doc.text)
        if len(sents) < 3:
            return [(doc.text, {})]
        emb = self.embedder.encode_passages(sents)
        dist = 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)
        thr = float(np.percentile(dist, pct))
        out, cur, n = [], [sents[0]], count_words(sents[0])
        for i, s in enumerate(sents[1:]):
            w = count_words(s)
            if (dist[i] > thr and n >= min_size) or n + w > max_size:
                out.append((" ".join(cur), {}))
                cur, n = [], 0
            cur.append(s)
            n += w
        if cur:
            out.append((" ".join(cur), {}))
        return out


BUILTIN = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "structure": StructureChunker,
    "semantic": SemanticChunker,
}


def make_chunker(spec: dict, embedders: dict):
    spec = dict(spec)
    kind = spec.pop("type")
    name = spec.pop("name", kind)
    if "embedder" in spec:
        spec["embedder"] = embedders[spec["embedder"]]
    if kind in BUILTIN:
        return BUILTIN[kind](name, **spec)
    from .registry import load_object

    return load_object(kind)(name, **spec)
