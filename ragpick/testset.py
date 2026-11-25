"""Load, validate and split the hand-labelled test set.

One JSON object per line (.jsonl) or a JSON list. Fields:

    id          unique string
    question    the question
    answer      your reference answer ("" for unanswerable)
    evidence    gold passage text: a string, a list of strings, or a list of {"text": ..., "doc": ...}
                ("doc" is optional: file name or doc id; restricts matching to that document)
    type        lookup | numeric | multi_doc | latest_rule | unanswerable | anything you like
    split       dev | test   (fill it with `ragpick split`)
    answerable  optional; defaults to False when type == "unanswerable"
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e})") from e
    return rows


def _normalize(row: dict, n: int) -> dict:
    q = dict(row)
    q.setdefault("id", f"q{n:03d}")
    if not q.get("question"):
        raise ValueError(f"Question {q['id']} has no 'question'")
    q.setdefault("type", "lookup")
    q.setdefault("answer", "")
    ev = q.get("evidence") or []
    if isinstance(ev, (str, dict)):
        ev = [ev]
    q["evidence"] = [{"text": e} if isinstance(e, str) else dict(e) for e in ev if e]
    q.setdefault("answerable", q["type"] != "unanswerable")
    q["answerable"] = bool(q["answerable"])
    if q["answerable"] and not q["evidence"]:
        raise ValueError(f"Question {q['id']} is answerable but has no gold evidence text")
    return q


def load_testset(path: Path) -> list[dict]:
    rows = _read(path)
    qs = [_normalize(r, i) for i, r in enumerate(rows, 1)]
    ids = [q["id"] for q in qs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate question ids: {sorted(dupes)}")
    return qs


def select_split(qs: list[dict], split: str) -> list[dict]:
    if split in ("all", None):
        return qs
    missing = [q["id"] for q in qs if "split" not in q]
    if missing:
        raise ValueError(f"{len(missing)} questions have no 'split'. Run: ragpick split <testset>")
    return [q for q in qs if q["split"] == split]


def assign_splits(path: Path, dev_frac: float = 0.3, seed: int = 42, force: bool = False) -> dict:
    """Stratified dev/test split by question type, written back into the file."""
    rows = _read(path)
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for r in rows:
        if force or "split" not in r:
            by_type[r.get("type", "lookup")].append(r)
    for group in by_type.values():
        rng.shuffle(group)
        n_dev = round(len(group) * dev_frac)
        for i, r in enumerate(group):
            r["split"] = "dev" if i < n_dev else "test"
    with open(path, "w", encoding="utf-8") as f:
        if path.suffix == ".json":
            json.dump(rows, f, ensure_ascii=False, indent=2)
        else:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for r in rows:
        counts[(r["split"], r.get("type", "lookup"))] += 1
    return dict(counts)
