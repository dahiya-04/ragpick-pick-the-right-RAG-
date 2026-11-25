"""LLM client with a persistent on-disk cache, so re-running an experiment costs nothing."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .textutil import stable_hash


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    error: str = ""


class LLMCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()


class AnthropicLLM:
    """Claude via the official Anthropic SDK.

    Credentials come from ANTHROPIC_API_KEY (or an `ant auth login` profile).
    `fallbacks: default` lets the API re-run a declined request on a fallback model
    instead of returning an empty answer; set `fallbacks: null` to turn it off.
    """

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 4096, effort: str | None = "medium",
                 fallbacks: str | list | None = "default", cache: LLMCache | None = None, max_retries: int = 5, **_):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("Generation with Claude needs: pip install 'ragpick[llm]'") from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.model, self.max_tokens, self.effort, self.fallbacks = model, max_tokens, effort, fallbacks
        self.cache = cache

    @property
    def label(self):
        return self.model

    def complete(self, system: str, user: str, schema: dict | None = None) -> LLMResult:
        key = stable_hash({"m": self.model, "s": system, "u": user, "schema": schema,
                           "e": self.effort, "t": self.max_tokens})
        if self.cache and (hit := self.cache.get(key)):
            return LLMResult(**{**hit, "cached": True})

        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        output_config = {}
        if self.effort:
            output_config["effort"] = self.effort
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            kwargs["output_config"] = output_config

        a = self._anthropic
        t0 = time.perf_counter()
        try:
            if self.fallbacks:
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks=self.fallbacks, **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except (a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError, a.BadRequestError):
            raise  # configuration problems: stop the run instead of recording 1000 failures
        except (a.RateLimitError, a.APIStatusError, a.APIConnectionError) as e:
            return LLMResult(text="", error=f"{type(e).__name__}: {e}")
        latency = (time.perf_counter() - t0) * 1000

        error = ""
        if resp.stop_reason == "refusal":
            error = "refusal"
        elif resp.stop_reason == "max_tokens":
            error = "max_tokens"
        text = "".join(b.text for b in resp.content if b.type == "text")
        result = LLMResult(text=text, input_tokens=resp.usage.input_tokens,
                           output_tokens=resp.usage.output_tokens, latency_ms=latency, error=error)
        if self.cache and not error:
            self.cache.put(key, {k: v for k, v in result.__dict__.items() if k != "cached"})
        return result


class OpenAICompatibleLLM:
    """Any server that speaks the OpenAI chat-completions API (e.g. Goedel Machines).

    The base URL and key are read from environment variables (or .env):
        base_url_env (default OPENAI_COMPAT_BASE_URL), api_key_env (default OPENAI_COMPAT_API_KEY)
    `thinking: false` sends chat_template_kwargs={"thinking": false} (turns reasoning off on
    servers that support it; reasoning tokens count as output and are billed/rate-limited).
    When a JSON schema is requested, the schema is described in the system prompt and the
    first JSON object in the reply is extracted (structured-output parameters vary by server).
    """

    _warned_retention = False
    _warn_lock = threading.Lock()

    def __init__(self, model: str, base_url: str | None = None, base_url_env: str = "OPENAI_COMPAT_BASE_URL",
                 api_key_env: str = "OPENAI_COMPAT_API_KEY", max_tokens: int = 2048, thinking: bool | None = None,
                 temperature: float | None = None, extra_body: dict | None = None, max_retries: int = 6,
                 timeout: float = 300.0, cache: LLMCache | None = None, **_):
        import os

        self.base_url = (os.environ.get(base_url_env) or base_url or "").rstrip("/")
        self.api_key = os.environ.get(api_key_env) or ""
        if not self.base_url or not self.api_key:
            raise ValueError(f"LLM '{model}' needs {api_key_env} (and {base_url_env} unless base_url is set); "
                             f"put them in the environment or a .env file")
        self.model, self.max_tokens, self.thinking, self.temperature = model, max_tokens, thinking, temperature
        self.extra_body = extra_body or {}
        self.max_retries, self.timeout, self.cache = max_retries, timeout, cache

    @property
    def label(self):
        return self.model

    def _post(self, body: dict) -> tuple[dict, dict]:
        import urllib.error
        import urllib.request

        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8")), dict(resp.headers)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(float(e.headers.get("retry-after") or min(60, 2 ** (attempt + 1))))
                    continue
                if e.code in (400, 401, 402, 403, 404, 413):
                    # configuration / billing problems: stop the run instead of recording 1000 failures
                    raise RuntimeError(f"{self.model}: HTTP {e.code}: {detail}") from e
                return {"_error": f"HTTP {e.code}: {detail}"}, {}
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(min(60, 2 ** (attempt + 1)))
                    continue
                return {"_error": f"connection: {e}"}, {}
        return {"_error": "retries exhausted"}, {}

    @staticmethod
    def _extract_json(text: str) -> str:
        start, end = text.find("{"), text.rfind("}")
        return text[start:end + 1] if start != -1 and end > start else text

    def complete(self, system: str, user: str, schema: dict | None = None) -> LLMResult:
        if schema:
            example = {k: (True if v.get("type") == "boolean" else 0 if v.get("type") in ("integer", "number")
                           else "...") for k, v in schema.get("properties", {}).items()}
            system = (f"{system}\n\nRespond with only a JSON object and no other text, in exactly this shape: "
                      f"{json.dumps(example)}\nUse the JSON literals true or false for yes/no fields "
                      f"(not objects, not strings), and a plain string for text fields.")
        key = stable_hash({"m": self.model, "u": self.base_url, "s": system, "q": user, "t": self.max_tokens,
                           "th": self.thinking, "temp": self.temperature, "x": self.extra_body})
        if self.cache and (hit := self.cache.get(key)):
            return LLMResult(**{**hit, "cached": True})

        body = {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.thinking is not None:
            body["chat_template_kwargs"] = {"thinking": bool(self.thinking)}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(self.extra_body)

        t0 = time.perf_counter()
        resp, headers = self._post(body)
        latency = (time.perf_counter() - t0) * 1000
        if "_error" in resp:
            return LLMResult(text="", error=resp["_error"])
        if headers.get("X-Data-Policy", headers.get("x-data-policy", "")).lower() == "retained":
            with OpenAICompatibleLLM._warn_lock:
                if not OpenAICompatibleLLM._warned_retention:
                    OpenAICompatibleLLM._warned_retention = True
                    print("  ! note: this API key's prompts are retained by the provider (X-Data-Policy: retained)")

        choice = (resp.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        error = "max_tokens" if choice.get("finish_reason") == "length" else ""
        if schema:
            text = self._extract_json(text)
        usage = resp.get("usage") or {}
        result = LLMResult(text=text.strip(), input_tokens=usage.get("prompt_tokens", 0) or 0,
                           output_tokens=usage.get("completion_tokens", 0) or 0, latency_ms=latency, error=error)
        if self.cache and not error and text:
            self.cache.put(key, {k: v for k, v in result.__dict__.items() if k != "cached"})
        return result


PRESETS = {
    "goedel": {"base_url": "https://api.goedelmachines.com/v1", "base_url_env": "GOEDEL_BASE_URL",
               "api_key_env": "GOEDEL_API_KEY", "model": "deepseek/deepseek-v4.1-flash"},
}


def make_llm(spec: dict, cache: LLMCache | None):
    spec = dict(spec)
    kind = spec.pop("type", "anthropic")
    if kind == "anthropic":
        return AnthropicLLM(cache=cache, **spec)
    if kind in PRESETS:
        return OpenAICompatibleLLM(cache=cache, **{**PRESETS[kind], **spec})
    if kind == "openai-compatible":
        return OpenAICompatibleLLM(cache=cache, **spec)
    from .registry import load_object

    return load_object(kind)(cache=cache, **spec)
