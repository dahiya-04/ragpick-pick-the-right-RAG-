"""Answer generation. One fixed prompt for every setup, so only retrieval changes between runs."""

from __future__ import annotations

import re
import time

from .textutil import content_words, split_sentences

NOT_FOUND = "NOT FOUND"

SYSTEM_PROMPT = """You answer questions about regulatory circulars using only the numbered context passages supplied with each question.

Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- Cite every passage you rely on with its number in square brackets, for example [2].
- If the passages do not contain the answer, reply with exactly: NOT FOUND
- If passages disagree, follow the most recent one and say that it replaced the older rule.
- Keep the answer short: one to three sentences, quoting numbers, limits and dates exactly."""


def format_context(chunks) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        bits = [c.meta.get("title"), c.meta.get("circular_no"), c.meta.get("date")]
        if c.meta.get("section"):
            bits.append(f"para {c.meta['section']}")
        if c.meta.get("superseded"):
            bits.append("SUPERSEDED")
        header = " | ".join(str(b) for b in bits if b)
        blocks.append(f"[{i}] ({header})\n{c.text}")
    return "\n\n".join(blocks)


def is_abstention(answer: str) -> bool:
    return NOT_FOUND.lower() in answer.lower()[:200] or not answer.strip()


class LLMGenerator:
    def __init__(self, llm, system_prompt: str | None = None):
        self.llm = llm
        self.system = system_prompt or SYSTEM_PROMPT
        self.label = getattr(llm, "label", "llm")

    def answer(self, question: str, chunks) -> dict:
        user = f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"
        r = self.llm.complete(self.system, user)
        return {"answer": r.text.strip(), "abstained": is_abstention(r.text), "gen_ms": r.latency_ms,
                "in_tok": r.input_tokens, "out_tok": r.output_tokens, "gen_error": r.error, "cached": r.cached}


class ExtractiveGenerator:
    """No-LLM baseline: returns the context sentence that best matches the question.

    Useful for offline demos and as a floor that any LLM setup should beat.
    """

    label = "extractive"

    def __init__(self, min_overlap: float = 0.4, **_):
        self.min_overlap = min_overlap

    def answer(self, question, chunks):
        t0 = time.perf_counter()
        q = set(content_words(question))
        best, best_score = "", 0.0
        for i, c in enumerate(chunks, 1):
            for s in split_sentences(c.text):
                score = len(q & set(content_words(s))) / max(1, len(q))
                if score > best_score:
                    best, best_score = f"{s} [{i}]", score
        if best_score < self.min_overlap:
            best = NOT_FOUND
        return {"answer": best, "abstained": best == NOT_FOUND, "gen_ms": (time.perf_counter() - t0) * 1000,
                "in_tok": 0, "out_tok": 0, "gen_error": "", "cached": False}


def make_generator(spec: dict, cache):
    spec = dict(spec)
    kind = spec.get("type", "anthropic")
    if kind == "extractive":
        return ExtractiveGenerator(**{k: v for k, v in spec.items() if k != "type"})
    from .llm import make_llm

    system_prompt = spec.pop("system_prompt", None)
    return LLMGenerator(make_llm(spec, cache), system_prompt)


def cited_passages(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)})
