"""Draft test questions from the corpus with an LLM, for a human to review.

    ragpick draft CONFIG --n 40     -> testset/draft_questions.csv  (review it: keep = y / n, edit text)
    ragpick accept DRAFT_CSV        -> appends the kept rows to testset/questions.jsonl

Every draft is checked automatically before you see it:
- the evidence must really be in the source passage (word-for-word, small differences allowed)
- near-duplicates of existing questions are dropped
- unanswerable drafts are searched for in the whole corpus and dropped if a passage answers them
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .chunking import Chunk, recursive_split
from .config import load_config
from .llm import LLMCache, make_llm
from .metrics import shingles
from .parsing import parse_corpus
from .retrieval import BM25Retriever
from .testset import _normalize, _read
from .textutil import content_words, normalize

INTRO = """You write test questions for checking a search-and-answer system over a document collection.
Rules for every question:
- It must make sense on its own, to someone who has not seen the passage. Never say "the passage", "this document" or "the above". If the topic is ambiguous across documents, name the document by its title or year.
- answer: a short reference answer (one or two sentences) using only facts in the passage(s).
- evidence: the exact sentence(s) from the passage that contain the answer, copied word for word.
- If no good question is possible (e.g. the passage is only an address, header or list of references), return an empty question."""

TYPE_HINT = {
    "lookup": "Ask about a definition, requirement, condition, or who / what / which.",
    "numeric": "Ask about a number, amount, limit, percentage, date or time period stated in the passage.",
    "paraphrase": ("Ask about a fact in the passage, but word the question the way a non-expert customer would: "
                   "avoid the passage's key terms and phrasing."),
}

SINGLE_SCHEMA = {"type": "object", "properties": {
    "question": {"type": "string"}, "answer": {"type": "string"}, "evidence": {"type": "string"}}}
MULTI_SCHEMA = {"type": "object", "properties": {
    "question": {"type": "string"}, "answer": {"type": "string"},
    "evidence_1": {"type": "string"}, "evidence_2": {"type": "string"}}}
UNANS_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}}
CHECK_SCHEMA = {"type": "object", "properties": {"answerable": {"type": "boolean"}, "reason": {"type": "string"}}}

CHECK_SYSTEM = ("Decide whether the passages below contain the answer to the question. answerable is true only if "
                "the passages state the specific fact asked for.")

DEFAULT_MIX = {"lookup": 0.25, "numeric": 0.25, "paraphrase": 0.2, "multi_doc": 0.15, "unanswerable": 0.15}
FIELDS = ["keep", "id", "type", "question", "answer", "evidence_1", "doc_1", "evidence_2", "doc_2", "notes"]


def _passages(docs, size=180, min_words=40):
    out = []
    for d in docs:
        parts = recursive_split(d.text, size)
        for i, t in enumerate(parts):
            if i == 0 and len(parts) >= 3:
                continue  # the opening passage is mostly letterhead: circular no., date, addressees
            if len(t.split()) >= min_words:
                out.append(Chunk(f"{d.doc_id}:{i}", d.doc_id, t, t, {"title": d.title,
                                                                    "superseded": bool(d.meta.get("superseded"))}))
    return out


def _spread(passages, rng):
    """Round-robin over documents so the drafts cover the whole corpus, not one long file."""
    by_doc = {}
    for i, p in enumerate(passages):
        by_doc.setdefault(p.doc_id, []).append(i)
    queues = list(by_doc.values())
    for q in queues:
        rng.shuffle(q)
    rng.shuffle(queues)
    order = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop())
    return order


def _plan(n, mix, rng):
    counts = {t: round(n * f) for t, f in mix.items()}
    counts["lookup"] += n - sum(counts.values())
    kinds = [t for t, c in counts.items() for _ in range(c)]
    rng.shuffle(kinds)
    return kinds


def _in_passage(evidence: str, passage: str) -> bool:
    if not evidence.strip():
        return False
    if normalize(evidence) in normalize(passage):
        return True
    ev = shingles(evidence)
    return bool(ev) and len(ev & shingles(passage)) / len(ev) >= 0.8


def _json(text):
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _block(p):
    return f"Document title: {p.meta['title']}\nPassage:\n{p.text}"


_NOT_STANDALONE = re.compile(r"\b(passage|the (two|first|second|above) (document|circular|direction)s?|"
                             r"both documents)\b", re.I)
_VAGUE = re.compile(r"\b(these|the) (directions?|rules|guidelines|norms|circular|instructions)\b", re.I)


def _similar(q1: str, q2: str) -> bool:
    a, b = set(content_words(q1)), set(content_words(q2))
    return bool(a and b) and len(a & b) / len(a | b) >= 0.7


class Drafter:
    def __init__(self, llm, passages, seed):
        self.llm, self.passages, self.rng = llm, passages, random.Random(seed)
        self.bm25 = BM25Retriever()
        self.bm25.index(passages)

    def single(self, p, kind):
        if kind == "numeric" and not any(c.isdigit() for c in p.text):
            kind = "lookup"
        r = self.llm.complete(f"{INTRO}\n\nQuestion style: {TYPE_HINT[kind]}", _block(p), schema=SINGLE_SCHEMA)
        d = _json(r.text)
        if not d.get("question"):
            return None, "no usable fact" if not r.error else r.error
        if not _in_passage(d.get("evidence", ""), p.text):
            return None, "evidence not found in passage"
        return {"type": kind, "question": d["question"], "answer": d.get("answer", ""),
                "evidence_1": d["evidence"], "doc_1": p.doc_id}, ""

    def multi(self, p):
        other = next((self.passages[i] for i, _ in self.bm25.search(p.text, 20)
                      if self.passages[i].doc_id != p.doc_id), None)
        if other is None:
            return None, "no related passage in another document"
        user = f"PASSAGE 1\n{_block(p)}\n\nPASSAGE 2\n{_block(other)}"
        system = (f"{INTRO}\n\nWrite ONE question that can only be answered by combining facts from BOTH passages "
                  f"about the SAME subject (for example: the same rule in both documents, or how a rule changed "
                  f"between them). Name both documents in the question by their title or year. Do not combine "
                  f"unrelated facts just because both contain a number. "
                  f"evidence_1 is copied from passage 1 and evidence_2 from passage 2. "
                  f"If the passages share no meaningful topic, return an empty question.")
        d = _json(self.llm.complete(system, user, schema=MULTI_SCHEMA).text)
        if not d.get("question"):
            return None, "passages not related"
        if not (_in_passage(d.get("evidence_1", ""), p.text) and _in_passage(d.get("evidence_2", ""), other.text)):
            return None, "evidence not found in passages"
        return {"type": "multi_doc", "question": d["question"], "answer": d.get("answer", ""),
                "evidence_1": d["evidence_1"], "doc_1": p.doc_id,
                "evidence_2": d["evidence_2"], "doc_2": other.doc_id}, ""

    def unanswerable(self, p):
        system = (f"{INTRO}\n\nThis time write ONE realistic question on the same topic as the passage that the "
                  f"passage does NOT answer: ask for a specific detail (a number, deadline, fee, condition) a reader "
                  f"might expect, but that is not stated. It must look answerable. Return only the question.")
        q = _json(self.llm.complete(system, _block(p), schema=UNANS_SCHEMA).text).get("question", "")
        if not q:
            return None, "no question"
        hits = [self.passages[i] for i, _ in self.bm25.search(q, 8)]
        ctx = "\n\n".join(f"[{n}] ({h.meta['title']}) {h.text}" for n, h in enumerate(hits, 1))
        check = _json(self.llm.complete(CHECK_SYSTEM, f"Question:\n{q}\n\nPassages:\n{ctx}", schema=CHECK_SCHEMA).text)
        if check.get("answerable") is not False:
            return None, "corpus seems to answer it"
        return {"type": "unanswerable", "question": q, "answer": "", "doc_1": p.doc_id}, ""

    def make(self, p, kind):
        if kind == "multi_doc":
            return self.multi(p)
        if kind == "unanswerable":
            return self.unanswerable(p)
        return self.single(p, kind)


def _notes(row, superseded):
    notes = []
    if row["type"] == "unanswerable":
        notes.append("checked against the top-8 search results; confirm no document answers it")
    if any(superseded.get(row.get(k)) for k in ("doc_1", "doc_2")):
        notes.append("source is a SUPERSEDED document: make sure the question names that document/year "
                     "or the answer is still the current rule")
    if _VAGUE.search(row["question"]):
        notes.append("VAGUE: says 'these Directions/rules'; add the document name or year")
    if row["type"] != "unanswerable":
        q, ev = set(content_words(row["question"])), set(content_words(row["evidence_1"]))
        if q and len(q & ev) / len(q) > 0.8:
            notes.append("copies the passage wording (easy for keyword search)")
    return "; ".join(notes)


def draft(config_path, n: int = 40, out: str | None = None, seed: int = 0, overrides=None, log=print) -> Path:
    cfg = load_config(config_path, overrides)
    base = Path(cfg["_base_dir"])
    cache_dir = (base / cfg["cache_dir"]).resolve()
    testset_path = base / cfg["testset"]["path"]
    out_path = Path(out) if out else testset_path.parent / "draft_questions.csv"

    docs = parse_corpus(cfg["corpus"], base, cache_dir, log)
    passages = _passages(docs)
    if not passages:
        raise ValueError("No passages of 40+ words found in the corpus")
    superseded = {d.doc_id: bool(d.meta.get("superseded")) for d in docs}
    existing = [r.get("question", "") for r in (_read(testset_path) if testset_path.exists() else [])]

    llm = make_llm(cfg["generation"]["llm"], LLMCache(cache_dir / "llm_cache.sqlite"))
    drafter = Drafter(llm, passages, seed)
    rng = random.Random(seed)
    kinds = _plan(n, DEFAULT_MIX, rng)
    order = _spread(passages, rng)
    # try ~1.5x as many passages as needed: some drafts fail the checks
    tasks = [(passages[order[i % len(order)]], kinds[i % n]) for i in range(int(n * 1.5) + 2)]
    log(f"[draft] {len(docs)} documents, {len(passages)} passages; drafting {n} questions "
        f"({', '.join(f'{t} {c}' for t, c in Counter(kinds).items())}) with {llm.label}")

    with ThreadPoolExecutor(cfg["generation"]["concurrency"]) as pool:
        results = list(pool.map(lambda t: (t[1], *drafter.make(*t)), tasks))

    rows, want, dropped = [], Counter(kinds), Counter()
    for kind, row, why in results:
        if row is not None and _NOT_STANDALONE.search(row["question"]):
            row, why = None, "refers to 'the passage' (not standalone)"
        if row is None:
            dropped[why] += 1
            continue
        if want[kind] <= 0:
            continue
        if any(_similar(row["question"], q) for q in existing + [r["question"] for r in rows]):
            dropped["duplicate"] += 1
            continue
        want[kind] -= 1
        row.update(keep="y", id=f"d{len(rows) + 1:03d}", notes=_notes(row, superseded))
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    got = Counter(r["type"] for r in rows)
    log(f"[draft] wrote {len(rows)} questions: {dict(got)}")
    if dropped:
        log(f"        dropped by automatic checks: {dict(dropped)}")
    if len(rows) < n:
        log(f"        {n - len(rows)} fewer than asked; for more, draft again with --seed 1 --out <another file>")
    log(f"Review {out_path}: set keep=n to reject, fix any wording, then run: ragpick accept {out_path}")
    return out_path


def accept(draft_csv, into: str | None = None, log=print) -> Path:
    draft_csv = Path(draft_csv)
    target = Path(into) if into else draft_csv.parent / "questions.jsonl"
    if target.suffix != ".jsonl":
        raise ValueError(f"accept appends to a .jsonl test set, got {target}")
    with open(draft_csv, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("keep") or "").strip().lower() in ("y", "yes", "1", "true")]
    existing = _read(target) if target.exists() else []
    used = {r.get("id") for r in existing}
    seen = {normalize(r.get("question", "")) for r in existing}
    new, skipped = [], 0
    for r in rows:
        if normalize(r["question"]) in seen:
            skipped += 1
            continue
        evidence = [{"text": r[f"evidence_{i}"].strip(), "doc": r[f"doc_{i}"].strip()}
                    for i in (1, 2) if (r.get(f"evidence_{i}") or "").strip()]
        q = {"id": r["id"].strip(), "type": r["type"].strip(), "question": r["question"].strip(),
             "answer": r["answer"].strip(), "evidence": evidence, "source": "drafted"}
        n = 1
        while q["id"] in used or not q["id"]:
            q["id"], n = f"d{n:03d}", n + 1
        _normalize(q, 0)  # raises a clear error if a row is incomplete
        used.add(q["id"])
        seen.add(normalize(q["question"]))
        new.append(q)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        if existing and not target.read_text(encoding="utf-8").endswith("\n"):
            f.write("\n")
        for q in new:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    log(f"added {len(new)} questions to {target} ({dict(Counter(q['type'] for q in new))})"
        + (f"; skipped {skipped} already present" if skipped else ""))
    log(f"Next: ragpick split {target}   (then ragpick run <config>)")
    return target
