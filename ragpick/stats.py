"""How sure are we? Bootstrap confidence ranges and paired 'is it really better?' tests.

Scores are averages over questions, so with few questions they move a lot by chance.
- confidence range: re-sample the questions with replacement many times; the middle 95% of the
  resulting scores is the 95% range.
- paired comparison: both setups answer the SAME questions, so only the questions where they differ
  carry information. A sign-flip test asks: if the two setups were equally good, how often would
  random luck produce a gap this big? If that is 5% of the time or more (p >= 0.05), they are tied.
  (For right/wrong scores this is the exact McNemar test. The bootstrap is too eager when only a
  handful of questions differ, which is the usual case.)
"""

from __future__ import annotations

import math

import numpy as np

N_BOOT = 2000


def per_question(records, metric: str) -> dict[str, float]:
    """qid -> score for one setup. accuracy uses judged answers; recall metrics use answerable questions."""
    out = {}
    for r in records:
        if metric == "accuracy":
            if r.get("correct") is not None:
                out[r["qid"]] = float(r["correct"])
        elif r["answerable"] and r.get(metric) is not None:
            out[r["qid"]] = float(r[metric])
    return out


def _boot_means(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    return values[idx].mean(axis=1)


def ci(scores: dict[str, float], seed: int = 0) -> tuple[float, float] | None:
    if len(scores) < 2:
        return None
    m = _boot_means(np.array(list(scores.values())), seed)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(a: dict[str, float], b: dict[str, float], seed: int = 0) -> dict | None:
    """Paired difference a - b over the questions both have."""
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    d = np.array([a[q] - b[q] for q in common])
    m = _boot_means(d, seed)
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    p = sign_flip_p(d, seed)
    return {"diff": float(d.mean()), "lo": lo, "hi": hi, "n": len(common), "sd": float(d.std(ddof=1)),
            "differs": int((d != 0).sum()), "p": p, "significant": p < 0.05}


def sign_flip_p(d: np.ndarray, seed: int = 0) -> float:
    """Two-sided p-value: share of random +/- relabellings with a total gap at least as large."""
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0
    obs = abs(nz.sum())
    if len(nz) <= 16:  # exact: every sign pattern
        signs = 1 - 2 * ((np.arange(2 ** len(nz))[:, None] >> np.arange(len(nz))) & 1)
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(20000, len(nz)))
    return float((np.abs(signs @ nz) >= obs - 1e-9).mean())


def questions_needed(cmp: dict) -> int | None:
    """Rough number of questions for an observed gap to become clear (95% confidence, 80% power)."""
    if not cmp or cmp["diff"] == 0 or cmp["sd"] == 0:
        return None
    return math.ceil(((1.96 + 0.84) * cmp["sd"] / abs(cmp["diff"])) ** 2)


def fmt_range(r) -> str:
    return f"{r[0]:.2f}–{r[1]:.2f}" if r else "–"
