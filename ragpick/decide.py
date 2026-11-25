"""The final answer: which setup to use, given quality, speed, cost and your limits.

1. Drop setups that break your limits (decision.max_latency_ms on p95 end-to-end time,
   decision.max_usd_per_1k on cost per 1,000 queries).
2. Take the highest-quality setup that is left, plus every setup statistically tied with it.
3. Among those, pick the cheapest (decision.prefer: cost) or the fastest (prefer: speed).
   Cost is $ per 1k queries when prices are known, otherwise LLM tokens per query (a proxy).
"""

from __future__ import annotations

from .stats import compare


def _cost(r):
    return r.get("usd_per_1k_queries") if r.get("usd_per_1k_queries") is not None else r.get("tokens_per_query")


def _ms(r):
    return r.get("total_p95_ms") if r.get("total_p95_ms") is not None else r.get("retrieval_p95_ms")


def _secs(ms):
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def _uses_embeddings(r, cfg):
    spec = next((x for x in cfg["retrievers"] if x["name"] == r["retriever"]), {})
    return bool(spec.get("embedder"))


def recommend(summary, scores, primary, rank, cfg) -> tuple[str, dict | None, list[dict]]:
    """Returns (markdown text, recommended row, contender rows with a 'decision' note)."""
    dec = cfg["decision"]
    pool = [r for r in summary if r.get(primary) is not None]
    if not pool:
        return "", None, []
    max_ms, max_usd = dec.get("max_latency_ms"), dec.get("max_usd_per_1k")
    ok, why_out = [], {}
    for r in pool:
        if max_ms and (_ms(r) or 0) > max_ms:
            why_out[r["config"]] = f"p95 {_secs(_ms(r))} > limit {_secs(max_ms)}"
        elif max_usd and r.get("usd_per_1k_queries") is not None and r["usd_per_1k_queries"] > max_usd:
            why_out[r["config"]] = f"${r['usd_per_1k_queries']:.3f} per 1k > limit ${max_usd}"
        else:
            ok.append(r)
    limits = " and ".join(x for x in (max_ms and f"p95 time at most {_secs(max_ms)}",
                                      max_usd and f"cost at most ${max_usd} per 1k queries") if x)
    if not ok:
        fastest = min(pool, key=lambda r: _ms(r) or 0)
        return (f"**No setup meets your limits ({limits}).** The fastest is **{fastest['config']}** "
                f"at {_secs(_ms(fastest))} (p95). Loosen the limits or try smaller models / fewer passages.",
                None, [])

    top = max(ok, key=rank)
    contenders = [top]
    for r in ok:
        if r is not top and (c := compare(scores[top["config"]], scores[r["config"]])) and not c["significant"]:
            contenders.append(r)
    by_cost = lambda r: (_cost(r) if _cost(r) is not None else float("inf"), _ms(r) or 0)  # noqa: E731
    by_speed = lambda r: (_ms(r) or 0, _cost(r) if _cost(r) is not None else float("inf"))  # noqa: E731
    key = by_cost if dec["prefer"] == "cost" else by_speed
    contenders.sort(key=key)
    # Differences under 10% are noise, not a reason to give up quality: among setups within 10% of the
    # cheapest (or fastest), take the highest-quality one.
    best_val = key(contenders[0])[0]
    near = [r for r in contenders if key(r)[0] <= best_val * 1.1]
    pick = max(near, key=rank)
    contenders.remove(pick)
    contenders.insert(0, pick)
    cost_unit = "$ per 1k queries" if pick.get("usd_per_1k_queries") is not None else "LLM tokens per query"

    for r in contenders:
        r["decision"] = "recommended" if r is pick else ("top score" if r is top else "tied with top")
    for r in pool:
        if r["config"] in why_out:
            r["decision"] = f"excluded: {why_out[r['config']]}"

    lines = [f"**Use {pick['config']}.**"]
    if pick is top:
        lines.append(f"It has the highest {primary} ({top[primary]:.2f}, 95% range {top['ci']})"
                     + (f" among setups within your limits ({limits})" if limits else "")
                     + (", and none of the setups tied with it is cheaper or faster."
                        if len(contenders) > 1 else ", and no other setup comes close."))
    else:
        reasons = []
        if _ms(pick) and _ms(top) and _ms(pick) < _ms(top):
            reasons.append(f"{_ms(top) / _ms(pick):.1f}× faster ({_secs(_ms(pick))} vs {_secs(_ms(top))} p95 per query)")
        if _cost(pick) is not None and _cost(top) and _cost(pick) < _cost(top):
            reasons.append(f"{100 * (1 - _cost(pick) / _cost(top)):.0f}% cheaper ({cost_unit})")
        if _uses_embeddings(top, cfg) and not _uses_embeddings(pick, cfg):
            reasons.append("needs no embedding model or vector index")
        lines.append(f"Its {primary} ({pick[primary]:.2f}) is statistically tied with the top score "
                     f"({top[primary]:.2f}, {top['config']}), and it is " + ", ".join(reasons or ["simpler"]) + ".")
    if len(contenders) > 1:
        lines.append(f"{len(contenders)} setups are tied on quality; the choice among them is by "
                     f"{'cost' if dec['prefer'] == 'cost' else 'speed'}, treating differences under 10% as equal "
                     f"(set `decision.prefer` to change).")
    if why_out:
        lines.append(f"{len(why_out)} setup(s) were excluded by your limits ({limits}).")
    if pick.get("usd_per_1k_queries") is None and pick.get("tokens_per_query"):
        lines.append("Cost is compared by token count because no price is set for the generation model "
                     "(add it under `prices` in the config to see $).")
    return " ".join(lines), pick, contenders
