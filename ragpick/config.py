"""Config loading: YAML file + defaults + command-line overrides (key.sub=value)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "run_name": "rag-eval",
    "output_dir": "results",
    "cache_dir": ".ragpick_cache",
    "corpus": {
        "path": "data/corpus",
        "glob": "**/*",
        "parse": {"remove_headers": True, "drop_hindi": True, "tables": True},
        "metadata": {"extractor": "rbi", "csv": None, "patterns": {}},
    },
    "testset": {"path": "testset/questions.jsonl", "split": "test"},
    "embedders": {},
    "chunkers": [{"name": "fixed", "type": "fixed", "size": 400, "overlap": 50}],
    "retrievers": [{"name": "bm25", "type": "bm25"}],
    "rerankers": [{"name": "none", "type": "none"}],
    "superseded_filter": [False],
    "evaluation": {
        "k": [1, 5, 10],
        "overlap_threshold": 0.5,
        "purity_threshold": 0.8,
        "context_k": 5,
        "baseline": None,
        "failure_examples": 15,
    },
    "generation": {
        "enabled": True,
        "configs": "all",  # "all" or "top:N" (top N setups by Recall@context_k)
        "concurrency": 2,
        "llm": {"type": "goedel", "model": "deepseek/deepseek-v4.1-flash", "max_tokens": 2048, "thinking": False},
    },
    "judge": {
        "enabled": True,
        "llm": {"type": "goedel", "model": "deepseek/deepseek-v4.1-flash", "max_tokens": 2048, "thinking": False},
    },
    # USD per million tokens [input, output], used for the cost column. Key = LLM model name, or the
    # embedder's deployment/model name. Add your own, e.g. "deepseek/deepseek-v4.1-flash": [0.1, 0.4]
    "prices": {
        "claude-opus-5-5": [4.0, 20.0],
        "claude-sonnet-5-5": [2.0, 10.0],
        "claude-haiku-5-5": [0.10, 0.50],
        "claude-fable-5-2": [8.0, 40.0],
        "text-embedding-3-small": [0.02, 0.0],
        "text-embedding-3-large": [0.13, 0.0],
    },
    # How to choose among setups that are statistically tied on quality.
    "decision": {
        "max_latency_ms": None,    # limit on end-to-end time per query (p95), e.g. 3000
        "max_usd_per_1k": None,    # limit on cost per 1,000 queries (needs prices)
        "prefer": "cost",          # tie-break: "cost" (cheapest first) or "speed" (fastest first)
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_path(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    from .envfile import load_dotenv

    path = Path(path).resolve()
    load_dotenv(Path.cwd(), path.parent)  # .env next to the config or in the current folder
    user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = _merge(DEFAULTS, user)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_path(cfg, key.strip(), yaml.safe_load(raw))
    cfg["_base_dir"] = str(path.parent)
    cfg["_config_path"] = str(path)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    for section in ("chunkers", "retrievers", "rerankers"):
        names = [c.get("name", c.get("type")) for c in cfg[section]]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate names in `{section}`: {names}")
        for c in cfg[section]:
            if "type" not in c:
                raise ValueError(f"Every entry in `{section}` needs a `type`: {c}")
            emb = c.get("embedder")
            if emb and emb not in cfg["embedders"]:
                raise ValueError(f"{section} '{c.get('name')}' uses embedder '{emb}', which is not defined in `embedders`")
    gen = cfg["generation"]["configs"]
    if gen != "all" and not (isinstance(gen, str) and gen.startswith("top:")):
        raise ValueError("generation.configs must be 'all' or 'top:N'")
    if cfg["decision"]["prefer"] not in ("cost", "speed"):
        raise ValueError("decision.prefer must be 'cost' or 'speed'")
