"""Embedding back-ends.

- `hashing`: dependency-free hashed bag of words + bigrams. Not a real semantic model;
  it exists so the whole pipeline runs offline (demo, CI, smoke tests).
- `sentence-transformers`: any HF model, e.g. BAAI/bge-m3, intfloat/multilingual-e5-base.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from .textutil import stable_hash, words


class HashingEmbedder:
    def __init__(self, dim: int = 4096, **_):
        self.dim = dim
        self.cache_id = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        toks = words(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            v[zlib.crc32(f.encode()) % self.dim] += 1.0
        v = np.sqrt(v)
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    encode_queries = encode_passages


class SentenceTransformerEmbedder:
    def __init__(self, model: str, query_prefix: str = "", passage_prefix: str = "",
                 batch_size: int = 32, device: str | None = None, max_seq_length: int | None = None, **_):
        self.model_name = model
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.device = device
        self.max_seq_length = max_seq_length
        self._model = None
        self.cache_id = f"st-{model}-{query_prefix}-{passage_prefix}-{max_seq_length}"

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("Dense models need: pip install 'ragpick[dense]'") from e
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length:
                self._model.max_seq_length = self.max_seq_length
        return self._model

    def _encode(self, texts, prefix, show):
        return self.model.encode([prefix + t for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=show,
                                 convert_to_numpy=True).astype(np.float32)

    def encode_passages(self, texts):
        return self._encode(texts, self.passage_prefix, len(texts) > 200)

    def encode_queries(self, texts):
        return self._encode(texts, self.query_prefix, False)


class AzureEmbedder:
    """Embedding models deployed in Azure AI Foundry / Azure OpenAI.

    api: openai     -> Azure OpenAI deployments (text-embedding-3-large / -small, ada-002)
                       POST {endpoint}/openai/deployments/{deployment}/embeddings?api-version=...
    api: inference  -> Foundry model inference endpoint (Cohere-embed-v3-*, embed-v-4-0, ...)
                       POST {endpoint}/models/embeddings?api-version=...  (sends input_type query/document)

    Credentials come from environment variables (or a .env file):
        AZURE_EMBEDDING_ENDPOINT, AZURE_EMBEDDING_API_KEY, AZURE_EMBEDDING_API_VERSION (optional)
    Use endpoint_env / api_key_env to point one embedder at different variables.
    """

    def __init__(self, deployment: str, api: str = "openai", dimensions: int | None = None,
                 batch_size: int = 64, endpoint_env: str = "AZURE_EMBEDDING_ENDPOINT",
                 api_key_env: str = "AZURE_EMBEDDING_API_KEY", api_version_env: str = "AZURE_EMBEDDING_API_VERSION",
                 api_version: str | None = None, max_retries: int = 6, timeout: float = 60.0, **_):
        import os

        if api not in ("openai", "inference"):
            raise ValueError("azure embedder `api` must be 'openai' or 'inference'")
        self.endpoint = (os.environ.get(endpoint_env) or "").rstrip("/")
        self.api_key = os.environ.get(api_key_env) or ""
        if not self.endpoint or not self.api_key:
            raise ValueError(f"Azure embedder '{deployment}' needs {endpoint_env} and {api_key_env} "
                             f"(set them in the environment or in a .env file)")
        default_version = "2024-10-21" if api == "openai" else "2024-05-01-preview"
        self.api_version = api_version or os.environ.get(api_version_env) or default_version
        self.deployment, self.api, self.dimensions = deployment, api, dimensions
        self.batch_size = min(batch_size, 96 if api == "inference" else 2048)
        self.max_retries, self.timeout = max_retries, timeout
        self.cache_id = f"azure-{api}-{deployment}-{dimensions}"

    def _url(self):
        if self.api == "openai":
            return f"{self.endpoint}/openai/deployments/{self.deployment}/embeddings?api-version={self.api_version}"
        return f"{self.endpoint}/models/embeddings?api-version={self.api_version}"

    def _post(self, body: dict) -> dict:
        import json
        import time
        import urllib.error
        import urllib.request

        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "api-key": self.api_key,
                   "Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self._url(), data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                # Azure answers DeploymentNotFound intermittently for a few minutes after a deployment is created
                transient = e.code in (429, 500, 502, 503, 504) or (e.code == 404 and "DeploymentNotFound" in detail)
                if transient and attempt < self.max_retries:
                    wait = float(e.headers.get("retry-after") or min(60, 5 * 2 ** attempt))
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Azure embeddings HTTP {e.code} for '{self.deployment}': {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                raise RuntimeError(f"Azure embeddings request failed for '{self.deployment}': {e}") from e

    def _encode(self, texts: list[str], input_type: str) -> np.ndarray:
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start:start + self.batch_size]]
            body: dict = {"input": batch}
            if self.dimensions:
                body["dimensions"] = self.dimensions
            if self.api == "inference":
                body["model"] = self.deployment
                body["input_type"] = input_type
            resp = self._post(body)
            rows = sorted(resp["data"], key=lambda d: d["index"])
            out.extend(r["embedding"] for r in rows)
        if not out:
            return np.zeros((0, self.dimensions or 1), np.float32)
        m = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    def encode_passages(self, texts):
        return self._encode(texts, "document")

    def encode_queries(self, texts):
        return self._encode(texts, "query")


def make_embedder(spec: dict):
    kind = spec.get("type", "sentence-transformers")
    params = {k: v for k, v in spec.items() if k != "type"}
    if kind == "hashing":
        return HashingEmbedder(**params)
    if kind in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(**params)
    if kind == "azure":
        return AzureEmbedder(**params)
    from .registry import load_object

    return load_object(kind)(**params)


class QueryEmbeddingCache:
    """Disk cache for question embeddings, so re-runs (and dense + hybrid) embed each question once."""

    def __init__(self, path: Path):
        import sqlite3
        import threading

        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS q (key TEXT PRIMARY KEY, vec BLOB)")
        self._lock = threading.Lock()

    def encode(self, embedder, query: str) -> np.ndarray:
        key = stable_hash({"emb": embedder.cache_id, "q": query})
        with self._lock:
            row = self._db.execute("SELECT vec FROM q WHERE key=?", (key,)).fetchone()
        if row:
            return np.frombuffer(row[0], dtype=np.float32)
        vec = embedder.encode_queries([query])[0].astype(np.float32)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO q VALUES (?, ?)", (key, vec.tobytes()))
            self._db.commit()
        return vec


_query_caches: dict[str, QueryEmbeddingCache] = {}


def query_cache(cache_dir: Path) -> QueryEmbeddingCache:
    k = str(cache_dir)
    if k not in _query_caches:
        _query_caches[k] = QueryEmbeddingCache(cache_dir / "query_embeddings.sqlite")
    return _query_caches[k]


_passage_caches: dict[str, object] = {}


def cached_passage_embeddings(embedder, texts: list[str], cache_dir: Path) -> np.ndarray:
    """Embed passages, caching each text separately: only new or changed texts are sent to the model."""
    import sqlite3

    path = cache_dir / "passage_embeddings.sqlite"
    if str(path) not in _passage_caches:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(path), check_same_thread=False)
        db.execute("CREATE TABLE IF NOT EXISTS p (key TEXT PRIMARY KEY, vec BLOB)")
        _passage_caches[str(path)] = db
    db = _passage_caches[str(path)]

    keys = [stable_hash({"emb": embedder.cache_id, "t": t}) for t in texts]
    found: dict[str, np.ndarray] = {}
    for start in range(0, len(keys), 500):
        part = keys[start:start + 500]
        rows = db.execute(f"SELECT key, vec FROM p WHERE key IN ({','.join('?' * len(part))})", part).fetchall()
        found.update((k, np.frombuffer(v, dtype=np.float32)) for k, v in rows)

    missing = [i for i, k in enumerate(keys) if k not in found]
    if missing:
        print(f"    embedding {len(missing)} new chunks ({len(texts) - len(missing)} cached)")
        vecs = embedder.encode_passages([texts[i] for i in missing]).astype(np.float32)
        db.executemany("INSERT OR REPLACE INTO p VALUES (?, ?)",
                       [(keys[i], v.tobytes()) for i, v in zip(missing, vecs)])
        db.commit()
        found.update((keys[i], v) for i, v in zip(missing, vecs))
    if not texts:
        return np.zeros((0, 1), np.float32)
    return np.stack([found[k] for k in keys])
