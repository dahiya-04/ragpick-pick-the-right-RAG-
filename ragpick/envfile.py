"""Minimal .env loader (no extra dependency). Existing environment variables always win."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(*folders: str | Path) -> list[Path]:
    """Load .env from each folder and its parent folders; the nearest file wins for a repeated key."""
    loaded = []
    seen = set()
    candidates = [p / ".env" for folder in folders for p in (Path(folder).resolve(), *Path(folder).resolve().parents)]
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.removeprefix("export ").partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)
        loaded.append(path)
    return loaded
