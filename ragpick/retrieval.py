"""Retrievers: BM25 (keyword), dense (embeddings) and hybrid (Reciprocal Rank Fusion)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .embeddings import cached_passage_embeddings, query_cache
from .textutil import bm25_tokens


class Retriever:
    name: str

    def index(self, chunks, cache_dir: Path) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (chunk_index, score) pairs, best first."""
        raise NotImplementedError


def _top(scores: np.ndarray, k: int, positive_only: bool = False) -> list[tuple[int, float]]:
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx if not positive_only or scores[i] > 0]


class BM25Retriever(Retriever):
    def __init__(self, name="bm25", k1: float = 1.5, b: float = 0.75, **_):
        self.name, self.k1, self.b = name, k1, b

    def index(self, chunks, cache_dir=None):
        self.n = len(chunks)
        postings = defaultdict(lambda: ([], []))
        lengths = np.zeros(self.n, dtype=np.float32)
        for i, c in enumerate(chunks):
            toks = bm25_tokens(c.index_text)
            lengths[i] = len(toks)
            for t, tf in Counter(toks).items():
                postings[t][0].append(i)
                postings[t][1].append(tf)
        self.avgdl = float(lengths.mean()) if self.n else 1.0
        self.lengths = lengths
        self.post = {t: (np.array(ix), np.array(tf, dtype=np.float32)) for t, (ix, tf) in postings.items()}
        self.idf = {t: math.log(1 + (self.n - len(ix) + 0.5) / (len(ix) + 0.5)) for t, (ix, _) in self.post.items()}

    def search(self, query, k):
        scores = np.zeros(self.n, dtype=np.float32)
        for t in set(bm25_tokens(query)):
            if t not in self.post:
                continue
            ix, tf = self.post[t]
            denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[ix] / self.avgdl)
            scores[ix] += self.idf[t] * tf * (self.k1 + 1) / denom
        return _top(scores, k, positive_only=True)


class DenseRetriever(Retriever):
    def __init__(self, name="dense", embedder=None, **_):
        if embedder is None:
            raise ValueError("dense retriever needs `embedder:`")
        self.name, self.embedder = name, embedder

    def index(self, chunks, cache_dir):
        self.matrix = cached_passage_embeddings(self.embedder, [c.index_text for c in chunks], cache_dir)
        self.qcache = query_cache(cache_dir)

    def search(self, query, k):
        q = self.qcache.encode(self.embedder, query)
        return _top(self.matrix @ q, k)


class HybridRetriever(Retriever):
    """BM25 + dense, fused with Reciprocal Rank Fusion: score = sum w / (rrf_k + rank)."""

    def __init__(self, name="hybrid", embedder=None, rrf_k: int = 60, depth: int = 100,
                 bm25_weight: float = 1.0, dense_weight: float = 1.0, k1: float = 1.5, b: float = 0.75, **_):
        self.name = name
        self.sparse = BM25Retriever(k1=k1, b=b)
        self.dense = DenseRetriever(embedder=embedder)
        self.rrf_k, self.depth = rrf_k, depth
        self.weights = (bm25_weight, dense_weight)

    def index(self, chunks, cache_dir):
        self.sparse.index(chunks)
        self.dense.index(chunks, cache_dir)

    def search(self, query, k):
        depth = max(self.depth, k)
        fused: dict[int, float] = defaultdict(float)
        for w, results in zip(self.weights, (self.sparse.search(query, depth), self.dense.search(query, depth))):
            for rank, (i, _) in enumerate(results, 1):
                fused[i] += w / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


BUILTIN = {"bm25": BM25Retriever, "dense": DenseRetriever, "hybrid": HybridRetriever}


def make_retriever(spec: dict, embedders: dict) -> Retriever:
    spec = dict(spec)
    kind = spec.pop("type")
    spec.setdefault("name", kind)
    if "embedder" in spec:
        spec["embedder"] = embedders[spec["embedder"]]
    if kind in BUILTIN:
        return BUILTIN[kind](**spec)
    from .registry import load_object

    return load_object(kind)(**spec)
