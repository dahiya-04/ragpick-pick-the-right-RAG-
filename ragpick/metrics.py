"""Matching retrieved chunks against gold evidence *text* (not chunk ids).

Each strategy cuts documents differently, so a chunk counts as a hit when it overlaps
the gold passage enough:

    gold_coverage  = share of the gold passage's word 3-grams found in the chunk
    chunk_purity   = share of the chunk's word 3-grams found in the gold passage

hit = gold_coverage >= threshold  OR  (chunk_purity >= purity_threshold and the chunk is not tiny)

The second rule catches small chunks that sit entirely inside a long gold passage.
"""

from __future__ import annotations

from .textutil import words


def shingles(text: str, n: int = 3) -> set:
    toks = words(text)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


class EvidenceMatcher:
    def __init__(self, chunks, threshold: float = 0.5, purity_threshold: float = 0.8):
        self.chunks = chunks
        self.threshold = threshold
        self.purity_threshold = purity_threshold
        self._sh = [None] * len(chunks)

    def _chunk_sh(self, i):
        if self._sh[i] is None:
            self._sh[i] = shingles(self.chunks[i].text)
        return self._sh[i]

    def is_hit(self, chunk_idx: int, ev: dict) -> bool:
        if ev.get("doc") and not _doc_matches(self.chunks[chunk_idx].doc_id, ev["doc"]):
            return False
        gold = ev["_sh"]
        if not gold:
            return False
        ch = self._chunk_sh(chunk_idx)
        inter = len(gold & ch)
        if inter / len(gold) >= self.threshold:
            return True
        return bool(ch) and inter / len(ch) >= self.purity_threshold and len(ch) >= min(15, 0.3 * len(gold))

    def findable(self, ev: dict) -> bool:
        """Does *any* chunk in the corpus contain this evidence? False means parsing or chunking lost it."""
        cands = range(len(self.chunks))
        if ev.get("doc"):
            cands = [i for i in cands if _doc_matches(self.chunks[i].doc_id, ev["doc"])] or cands
        return any(self.is_hit(i, ev) for i in cands)


def _doc_matches(doc_id: str, doc: str) -> bool:
    doc = doc.rsplit(".", 1)[0] if doc.lower().endswith((".pdf", ".txt", ".md", ".html")) else doc
    return doc_id == doc or doc_id.endswith("/" + doc) or doc_id.split("/")[-1] == doc


def prepare_evidence(questions):
    for q in questions:
        for ev in q["evidence"]:
            ev["_sh"] = shingles(ev["text"])


def retrieval_metrics(ranked: list[int], q: dict, matcher: EvidenceMatcher, ks: list[int]) -> dict:
    """Recall@k = share of gold passages found in the top k (averaged over passages).
    Hit-all@k = every gold passage found. MRR = 1 / rank of the first relevant chunk."""
    evs = q["evidence"]
    if not evs:
        return {}
    first_rank = []  # per evidence: first rank (1-based) that hits it, or None
    for ev in evs:
        fr = next((r for r, ci in enumerate(ranked, 1) if matcher.is_hit(ci, ev)), None)
        first_rank.append(fr)
    out = {}
    for k in ks:
        found = [fr is not None and fr <= k for fr in first_rank]
        out[f"recall@{k}"] = sum(found) / len(found)
        out[f"hit_all@{k}"] = float(all(found))
    ranks = [fr for fr in first_rank if fr is not None]
    out["mrr"] = 1.0 / min(ranks) if ranks else 0.0
    out["evidence_ranks"] = first_rank
    return out
