"""Plug in your own components without editing the library.

In config.yaml use an import path as the `type`:

    chunkers:
      - name: mine
        type: my_package.chunkers:MyChunker   # class taking (name, **params)

The class must follow the same interface as the built-ins (see chunking.Chunker,
retrieval.Retriever, rerank.Reranker, generation.Generator).
"""

from __future__ import annotations

import importlib


def load_object(path: str):
    if ":" not in path:
        raise ValueError(f"Unknown component type '{path}'. Use a built-in name or 'module.path:ClassName'.")
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)
