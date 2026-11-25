"""Confidence ranges, tie verdicts and the final recommendation."""

import copy

from ragpick.config import DEFAULTS
from ragpick.decide import recommend
from ragpick.stats import ci, compare, questions_needed


def _scores(pattern, n=40):
    return {f"q{i}": float(pattern(i)) for i in range(n)}


def test_ci_contains_the_mean():
    s = _scores(lambda i: i % 4 != 0)  # 0.75
    lo, hi = ci(s)
    assert lo < 0.75 < hi and 0 <= lo and hi <= 1


def test_small_gap_is_a_tie_and_big_gap_is_significant():
    best = _scores(lambda i: i % 10 != 0)            # 0.90
    close = _scores(lambda i: i % 10 not in (0, 1))  # 0.80, differs on 4 questions
    far = _scores(lambda i: i % 2 == 0)              # 0.50
    assert not compare(best, close)["significant"]
    assert compare(best, far)["significant"]
    assert questions_needed(compare(best, close)) > 40


def test_sign_flip_is_exact_mcnemar_for_right_wrong_scores():
    import numpy as np

    from ragpick.stats import sign_flip_p

    assert sign_flip_p(np.array([1.0, 1, 1, 1] + [0] * 36)) == 0.125   # 4 of 4 wins: 2 * (1/2)^4
    assert sign_flip_p(np.array([1.0] * 6 + [0] * 34)) < 0.05           # 6 of 6 wins: p = 0.031
    assert sign_flip_p(np.zeros(10)) == 1.0


def _row(config, acc, ms, tokens, retriever="bm25"):
    return {"config": config, "retriever": retriever, "accuracy": acc, "ci": "–", "total_p95_ms": ms,
            "tokens_per_query": tokens, "recall@3": acc, "mrr": acc}


def _cfg(**decision):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["retrievers"] = [{"name": "bm25", "type": "bm25"}, {"name": "dense", "type": "dense", "embedder": "e"}]
    cfg["decision"].update(decision)
    return cfg


def _rank(r):
    return (r["accuracy"], r["recall@3"], r["mrr"])


def test_recommends_cheaper_setup_when_tied():
    summary = [_row("top", 0.90, 4000, 800, "dense"), _row("cheap", 0.875, 2500, 500)]
    scores = {"top": _scores(lambda i: i % 10 != 0), "cheap": _scores(lambda i: i % 8 != 0)}
    text, pick, _ = recommend(summary, scores, "accuracy", _rank, _cfg())
    assert pick["config"] == "cheap"
    assert "needs no embedding model" in text


def test_keeps_the_top_setup_when_it_is_clearly_better():
    summary = [_row("top", 0.90, 4000, 800), _row("cheap", 0.50, 1000, 300)]
    scores = {"top": _scores(lambda i: i % 10 != 0), "cheap": _scores(lambda i: i % 2 == 0)}
    _, pick, _ = recommend(summary, scores, "accuracy", _rank, _cfg())
    assert pick["config"] == "top"


def test_differences_under_ten_percent_do_not_beat_quality():
    summary = [_row("a", 0.90, 3000, 510), _row("b", 0.875, 3000, 500)]
    scores = {"a": _scores(lambda i: i % 10 != 0), "b": _scores(lambda i: i % 8 != 0)}
    _, pick, _ = recommend(summary, scores, "accuracy", _rank, _cfg())
    assert pick["config"] == "a"


def test_latency_limit_excludes_slow_setups():
    summary = [_row("slow", 0.90, 5000, 500), _row("fast", 0.50, 1000, 500)]
    scores = {"slow": _scores(lambda i: i % 10 != 0), "fast": _scores(lambda i: i % 2 == 0)}
    text, pick, _ = recommend(summary, scores, "accuracy", _rank, _cfg(max_latency_ms=2000))
    assert pick["config"] == "fast" and "excluded" in text
    text, pick, _ = recommend(summary, scores, "accuracy", _rank, _cfg(max_latency_ms=10))
    assert pick is None and "No setup meets your limits" in text
