"""Grading answers for correctness (vs. your reference) and faithfulness (vs. retrieved text).

Deterministic rules run first and skip the LLM where the answer is obvious:
- unanswerable question: correct iff the system said NOT FOUND
- answerable question + NOT FOUND: incorrect, but faithful
"""

from __future__ import annotations

import json

from .textutil import content_words

JUDGE_SYSTEM = """You grade answers produced by a retrieval-augmented question answering system over regulatory documents. Return two independent labels.

answer_correct: true if the system answer gives the same key facts as the reference answer. Numbers, limits, dates and time periods must match. Extra detail is fine if it does not contradict the reference.

faithful: true if every factual claim in the system answer is supported by the retrieved passages shown. Judge only against the passages, not against the reference answer or your own knowledge.

reason: one short sentence explaining the labels."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_correct": {"type": "boolean"},
        "faithful": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["answer_correct", "faithful", "reason"],
    "additionalProperties": False,
}


def parse_verdict(text: str) -> dict | None:
    """Read the judge's JSON; if it is malformed, recover the true/false labels with a regex."""
    import re

    try:
        data = json.loads(text)
        if isinstance(data.get("answer_correct"), bool) and isinstance(data.get("faithful"), bool):
            return data
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    out = {}
    for key in ("answer_correct", "faithful"):
        # plain `"key": true` or a nested object such as `"key": {"type": "boolean", "value": true}`
        m = re.search(rf'"{key}"\s*:\s*(?:(true|false)\b|\{{[^}}]*?\b(true|false)\b[^}}]*\}})', text or "", re.I)
        if not m:
            return None
        out[key] = (m.group(1) or m.group(2)).lower() == "true"
    m = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', text or "")
    out["reason"] = (m.group(1) if m else "") + " [recovered from malformed JSON]"
    return out


def _rule(q, gen) -> dict | None:
    if not q["answerable"]:
        if gen["abstained"]:
            return {"correct": True, "faithful": True, "judge_reason": "correctly abstained", "judge_method": "rule"}
        return None  # still need a faithfulness check; correctness is fixed below
    if gen["abstained"]:
        return {"correct": False, "faithful": True, "judge_reason": "abstained on an answerable question",
                "judge_method": "rule"}
    return None


class LLMJudge:
    def __init__(self, llm):
        self.llm = llm
        self.label = getattr(llm, "label", "llm")

    def judge(self, q, gen, context: str) -> dict:
        ruled = _rule(q, gen)
        if ruled:
            return ruled
        reference = q["answer"] if q["answerable"] else "(This question cannot be answered from the corpus.)"
        user = (f"Question:\n{q['question']}\n\nReference answer:\n{reference}\n\n"
                f"Retrieved passages:\n{context}\n\nSystem answer:\n{gen['answer']}")
        r = self.llm.complete(JUDGE_SYSTEM, user, schema=JUDGE_SCHEMA)
        data = parse_verdict(r.text)
        if data is None:
            return {"correct": None, "faithful": None, "judge_reason": f"judge error: {r.error or 'bad JSON'}",
                    "judge_method": "llm", "judge_in_tok": r.input_tokens, "judge_out_tok": r.output_tokens}
        correct = bool(data["answer_correct"]) if q["answerable"] else False  # answered an unanswerable question
        return {"correct": correct, "faithful": bool(data["faithful"]), "judge_reason": data.get("reason", ""),
                "judge_method": "llm", "judge_in_tok": r.input_tokens, "judge_out_tok": r.output_tokens}


def token_f1(a: str, b: str) -> float:
    ta, tb = content_words(a), content_words(b)
    if not ta or not tb:
        return 0.0
    common = 0
    pool = list(tb)
    for t in ta:
        if t in pool:
            pool.remove(t)
            common += 1
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


class LexicalJudge:
    """Heuristic judge for offline runs. Correct if token F1 with the reference >= threshold;
    faithful if most answer words appear in the retrieved passages. Not a substitute for an LLM judge."""

    label = "lexical-heuristic"

    def __init__(self, f1_threshold: float = 0.3, support_threshold: float = 0.8, **_):
        self.f1_threshold, self.support_threshold = f1_threshold, support_threshold

    def judge(self, q, gen, context):
        ruled = _rule(q, gen)
        if ruled:
            return ruled
        words = [w for w in content_words(gen["answer"]) if not w.isdigit() or len(w) > 1]
        ctx = set(content_words(context))
        support = sum(w in ctx for w in words) / max(1, len(words))
        correct = q["answerable"] and token_f1(gen["answer"], q["answer"]) >= self.f1_threshold
        return {"correct": bool(correct), "faithful": support >= self.support_threshold,
                "judge_reason": f"token-F1 / support={support:.2f}", "judge_method": "lexical"}


def make_judge(spec: dict, cache):
    spec = dict(spec)
    kind = spec.get("type", "anthropic")
    if kind == "lexical":
        return LexicalJudge(**{k: v for k, v in spec.items() if k != "type"})
    from .llm import make_llm

    return LLMJudge(make_llm(spec, cache))


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
