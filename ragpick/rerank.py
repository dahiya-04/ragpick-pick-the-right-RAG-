"""Rerankers re-score the top `top_n` retrieved chunks."""

from __future__ import annotations

from .textutil import content_words


class Reranker:
    name: str
    top_n: int

    def scores(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    def __init__(self, name="cross-encoder", model: str = "BAAI/bge-reranker-v2-m3", top_n: int = 30,
                 batch_size: int = 16, device: str | None = None, max_length: int = 512, **_):
        self.name, self.model_name, self.top_n = name, model, top_n
        self.batch_size, self.device, self.max_length = batch_size, device, max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError("Cross-encoder reranking needs: pip install 'ragpick[dense]'") from e
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
        return self._model

    def scores(self, query, texts):
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=self.batch_size)]


class LexicalOverlapReranker(Reranker):
    """Toy reranker (share of query words found in the chunk). Offline demo only."""

    def __init__(self, name="lexical", top_n: int = 20, **_):
        self.name, self.top_n = name, top_n

    def scores(self, query, texts):
        q = set(content_words(query))
        if not q:
            return [0.0] * len(texts)
        return [len(q & set(content_words(t))) / len(q) for t in texts]


BUILTIN = {"cross-encoder": CrossEncoderReranker, "lexical": LexicalOverlapReranker}


def make_reranker(spec: dict):
    spec = dict(spec)
    kind = spec.pop("type")
    spec.setdefault("name", kind)
    if kind == "none":
        return None
    if kind in BUILTIN:
        return BUILTIN[kind](**spec)
    from .registry import load_object

    return load_object(kind)(**spec)
