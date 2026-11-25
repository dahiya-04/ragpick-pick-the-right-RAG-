"""Write every result file: tables (CSV + Markdown), charts, failure analysis, HTML report."""

from __future__ import annotations

import base64
import csv
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .stats import ci, compare, fmt_range, per_question, questions_needed

# Reference palette (validated): one hue for magnitude, muted ink for text.
BLUE = "#2a78d6"
MUTED = "#898781"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[i]


def _fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(rows: list[dict], cols: list[str], headers: dict | None = None) -> str:
    headers = headers or {}
    out = ["| " + " | ".join(headers.get(c, c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def write_csv(rows, cols, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: (json.dumps(r[c], ensure_ascii=False) if isinstance(r.get(c), (list, dict)) else r.get(c))
                        for c in cols})


# --------------------------------------------------------------------------- aggregation


def summarize(setups, records, ks, context_k, prices, gen_model):
    price = prices.get(gen_model) if gen_model else None
    rows = []
    for s in setups:
        recs = records[s["config"]]
        ans = [r for r in recs if r["answerable"]]
        row = dict(s)
        row["n"] = len(recs)
        for k in ks:
            row[f"recall@{k}"] = _mean([r.get(f"recall@{k}") for r in ans])
        row[f"hit_all@{context_k}"] = _mean([r.get(f"hit_all@{context_k}") for r in ans])
        row["mrr"] = _mean([r.get("mrr") for r in ans])
        search_ms = [r["retrieval_ms"] + (r.get("embed_ms") or 0) for r in recs]
        row["retrieval_p50_ms"] = _pct(search_ms, 50)
        row["retrieval_p95_ms"] = _pct(search_ms, 95)
        emb_price = prices.get(recs[0].get("embed_model")) if recs and recs[0].get("embed_tok") else [0.0, 0.0]
        if s.get("generated"):
            total = [m + (r.get("gen_ms") or 0) for m, r in zip(search_ms, recs)]
            row["total_p50_ms"], row["total_p95_ms"] = round(_pct(total, 50)), round(_pct(total, 95))
            row["tokens_per_query"] = round(_mean([(r.get("in_tok") or 0) + (r.get("out_tok") or 0) for r in recs]))
            if price and emb_price:
                costs = [((r.get("in_tok") or 0) * price[0] + (r.get("out_tok") or 0) * price[1]
                          + (r.get("embed_tok") or 0) * emb_price[0]) / 1e6 for r in recs]
                row["usd_per_1k_queries"] = 1000 * statistics.mean(costs)
            judged = [r for r in recs if r.get("correct") is not None]
            row["accuracy"] = _mean([float(r["correct"]) for r in judged])
            answered = [r for r in recs if not r.get("abstained") and r.get("faithful") is not None]
            row["faithfulness"] = _mean([float(r["faithful"]) for r in answered])
            unans = [r for r in recs if not r["answerable"]]
            row["abstain_correct"] = _mean([float(bool(r.get("abstained"))) for r in unans])
            row["false_abstain"] = _mean([float(bool(r.get("abstained"))) for r in ans])
            row["gen_p50_ms"] = _pct([r.get("gen_ms") for r in recs], 50)
        rows.append(row)
    return rows


def by_type(setups, records, context_k):
    rows = []
    for s in setups:
        groups = defaultdict(list)
        for r in records[s["config"]]:
            groups[r["type"]].append(r)
        for t, recs in sorted(groups.items()):
            row = {"config": s["config"], "type": t, "n": len(recs),
                   f"recall@{context_k}": _mean([r.get(f"recall@{context_k}") for r in recs if r["answerable"]])}
            if s.get("generated"):
                row["accuracy"] = _mean([float(r["correct"]) for r in recs if r.get("correct") is not None])
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- charts


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    return plt


def bar_chart(rows, metric, title, path, highlight: set[str], ranges: dict | None = None):
    rows = [r for r in rows if r.get(metric) is not None]
    if not rows:
        return None
    plt = _plt()
    rows = sorted(rows, key=lambda r: r[metric])
    h = max(2.2, 0.32 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(8.5, h))
    labels = [r["config"] for r in rows]
    vals = [r[metric] for r in rows]
    ax.barh(range(len(rows)), vals, height=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2)
    if ranges:  # 95% range as a thin whisker on each bar
        for i, r in enumerate(rows):
            if ranges.get(r["config"]):
                lo, hi = ranges[r["config"]]
                ax.plot([lo, hi], [i, i], color=INK_2, linewidth=1.2, solid_capstyle="butt")
    ax.set_yticks(range(len(rows)), labels, fontsize=8.5)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for i, r in enumerate(rows):  # label only the setups worth reading, clear of the range line
        if r["config"] in highlight:
            x = max(r[metric], (ranges or {}).get(r["config"], (0, 0))[1] or 0) + 0.012
            ax.text(x, i, f"{r[metric]:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def heatmap(rows, metric, title, path):
    rows = [r for r in rows if r.get(metric) is not None]
    if not rows:
        return None
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    configs = list(dict.fromkeys(r["config"] for r in rows))
    types = sorted({r["type"] for r in rows})
    grid = [[None] * len(types) for _ in configs]
    for r in rows:
        grid[configs.index(r["config"])][types.index(r["type"])] = r[metric]
    import numpy as np

    data = np.array([[np.nan if v is None else v for v in row] for row in grid], dtype=float)
    fig, ax = plt.subplots(figsize=(1.4 * len(types) + 4.5, 0.36 * len(configs) + 1.4))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ).with_extremes(bad="#f0efec")
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(types)), types, fontsize=9)
    ax.set_yticks(range(len(configs)), configs, fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(types)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(configs)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(configs)):
        for j in range(len(types)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 0.55 else INK)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- narrative


def _find(summary, spec):
    if not spec:
        return summary[0]
    for r in summary:
        if all(r.get(k) == v for k, v in spec.items()):
            return r
    return summary[0]


def add_confidence(summary, records, primary, best):
    """95% range for each setup's primary score, and a paired verdict against the best setup."""
    scores = {r["config"]: per_question(records[r["config"]], primary) for r in summary}
    cmps = {}
    for r in summary:
        r["range"] = ci(scores[r["config"]])
        r["ci"] = fmt_range(r["range"])
        if r.get(primary) is None:
            continue
        if r["config"] == best["config"]:
            r["vs_best"] = "best"
            continue
        c = compare(scores[best["config"]], scores[r["config"]])
        cmps[r["config"]] = c
        if c:
            r["vs_best"] = f"{'worse' if c['significant'] else 'tied'} ({-c['diff']:+.2f})"
    return scores, cmps


def confidence_text(summary, best, base, cmps, scores, primary, n_q):
    tied = [r for r in summary if (c := cmps.get(r["config"])) and not c["significant"]]
    out = [f"Best {primary}: {best[primary]:.2f} (95% range {best['ci']})."]
    if tied:
        ordered = sorted(tied, key=lambda r: -r[primary])
        names = ", ".join(f"**{r['config']}**" for r in ordered[:5])
        if len(ordered) > 5:
            names += f" and {len(ordered) - 5} more (see the 'vs best' column)"
        out.append(f"Statistically tied with {len(tied)} other setup(s): {names}. "
                   f"With {n_q} questions the data cannot tell them apart.")
    elif cmps:
        out.append("It is clearly better than every other setup.")
    if base["config"] != best["config"] and (c := compare(scores[best["config"]], scores[base["config"]])):
        verdict = "a real improvement" if c["significant"] else "could be chance"
        out.append(f"Gain over the baseline: {c['diff']:+.2f} (95% range {c['lo']:+.2f} to {c['hi']:+.2f}), {verdict}.")
    return " ".join(out), tied


def headline(summary, baseline_spec, context_k, n_q):
    base = _find(summary, baseline_spec)
    gen = [r for r in summary if r.get("accuracy") is not None]
    primary = "accuracy" if gen else f"recall@{context_k}"
    pool = gen or summary
    best = max(pool, key=lambda r: rank_key(r, context_k))
    rk = f"recall@{context_k}"
    if best["config"] == base["config"]:
        text = f"The baseline setup ({base['config']}) was also the best: {primary} {base[primary]:.2f} on {n_q} questions."
    else:
        def change(name, a, b):
            verb = "raised" if b > a else "lowered" if b < a else "kept"
            return f"{verb} {name} from {a:.2f} to {b:.2f}" if verb != "kept" else f"kept {name} at {a:.2f}"

        parts = []
        if gen and base.get("accuracy") is not None:
            parts.append(change("answer accuracy", base["accuracy"], best["accuracy"]))
        parts.append(change(f"Recall@{context_k}", base[rk], best[rk]))
        text = (f"Best setup **{best['config']}** " + " and ".join(parts)
                + f" compared with the baseline ({base['config']}) on {n_q} questions.")
    return text, best, base


def rank_key(r, context_k):
    """Same ordering everywhere: accuracy first (if generated), then Recall@context_k, then MRR."""
    return (r.get("accuracy") or 0, r.get(f"recall@{context_k}") or 0, r.get("mrr") or 0)


def insights(summary, type_rows, setups, records, findability, context_k, best):
    out = []
    metric = "accuracy" if any(r.get("accuracy") is not None for r in type_rows) else f"recall@{context_k}"
    by_t = defaultdict(list)
    for r in type_rows:
        if r.get(metric) is not None:
            by_t[r["type"]].append(r)
    for t, rows in sorted(by_t.items()):
        top = max(rows, key=lambda r: r[metric])
        worst = min(rows, key=lambda r: r[metric])
        if top[metric] == worst[metric]:
            out.append(f"**{t}**: all setups tie at {metric} {top[metric]:.2f}.")
        elif top[metric] < 0.5:
            out.append(f"**{t}**: every setup struggles; the best ({top['config']}) reaches only {metric} {top[metric]:.2f}.")
        else:
            out.append(f"**{t}**: best is {top['config']} ({metric} {top[metric]:.2f}); worst is "
                       f"{worst['config']} ({worst[metric]:.2f}).")

    # Reranking effect: pair each reranked setup with the same setup without reranking.
    rk = f"recall@{context_k}"
    plain = {(r["chunker"], r["retriever"], r["superseded_filter"]): r for r in summary if r["reranker"] == "none"}
    deltas = defaultdict(list)
    for r in summary:
        if r["reranker"] != "none":
            p = plain.get((r["chunker"], r["retriever"], r["superseded_filter"]))
            if p:
                deltas[r["reranker"]].append((r[rk] - p[rk], (r["retrieval_p50_ms"] or 0) - (p["retrieval_p50_ms"] or 0)))
    for name, ds in deltas.items():
        dr = statistics.mean(d[0] for d in ds)
        dl = statistics.mean(d[1] for d in ds)
        out.append(f"Reranking with **{name}** changed Recall@{context_k} by {dr:+.3f} on average "
                   f"and added {dl:.0f} ms median latency per query.")

    # Superseded filtering effect on latest-rule style questions.
    filt = [r for r in summary if r["superseded_filter"]]
    if filt:
        unf = {(r["chunker"], r["retriever"], r["reranker"]): r for r in summary if not r["superseded_filter"]}
        m = "accuracy" if filt[0].get("accuracy") is not None else rk
        ds = [r[m] - unf[k][m] for r in filt if (k := (r["chunker"], r["retriever"], r["reranker"])) in unf
              and r.get(m) is not None and unf[k].get(m) is not None]
        if ds:
            out.append(f"Dropping superseded circulars from results changed {m} by {statistics.mean(ds):+.3f} on average.")

    for ch, fmap in findability.items():
        vals = [v for v in fmap.values() if v is not None]
        lost = sum(not v for v in vals)
        if lost:
            out.append(f"Chunker **{ch}** lost the gold evidence entirely for {lost}/{len(vals)} answerable "
                       f"questions (no chunk overlaps it enough): a parsing or chunk-boundary problem.")

    diag = Counter(r["diagnosis"] for r in records[best["config"]] if r["diagnosis"])
    if diag:
        top = ", ".join(f"{k} ({v})" for k, v in diag.most_common(4))
        out.append(f"Failures of the best setup, by automatic diagnosis: {top}.")
    return out


# --------------------------------------------------------------------------- main entry


def build_report(out_dir: Path, cfg, setups, records, questions, chunk_stats, findability, ks, context_k, docs, log):
    charts = out_dir / "charts"
    charts.mkdir(exist_ok=True)
    gen_model = cfg["generation"]["llm"].get("model") if cfg["generation"]["enabled"] else None
    summary = summarize(setups, records, ks, context_k, cfg["prices"], gen_model)
    type_rows = by_type(setups, records, context_k)
    generated = any(r.get("accuracy") is not None for r in summary)
    n_q = len(questions)
    primary = "accuracy" if generated else f"recall@{context_k}"

    # ---- best setup and how sure we are about it
    text, best, base = headline(summary, cfg["evaluation"].get("baseline"), context_k, n_q)
    scores, cmps = add_confidence(summary, records, primary, best)
    conf_text, tied = confidence_text(summary, best, base, cmps, scores, primary, n_q)
    text = f"{text} {conf_text}"
    from .decide import recommend

    rec_text, pick, contenders = recommend(summary, scores, primary, lambda r: rank_key(r, context_k), cfg)

    # ---- tables
    ret_cols = ["chunker", "retriever", "reranker"] + (["superseded_filter"] if len(cfg["superseded_filter"]) > 1 else [])
    metric_cols = [f"recall@{k}" for k in ks] + ["mrr", f"hit_all@{context_k}"]
    gen_cols = ["accuracy", "faithfulness", "abstain_correct", "false_abstain"] if generated else []
    sure_cols = ["ci", "vs_best"]
    if generated:
        gen_cols[1:1] = sure_cols
    else:
        metric_cols[metric_cols.index(primary) + 1:metric_cols.index(primary) + 1] = sure_cols
    lat_cols = ["retrieval_p50_ms"] + (["gen_p50_ms", "total_p95_ms", "tokens_per_query"] if generated else []) + \
               (["usd_per_1k_queries"] if any("usd_per_1k_queries" in r for r in summary) else [])
    cols = ret_cols + metric_cols + gen_cols + lat_cols
    summary_sorted = sorted(summary, key=lambda r: rank_key(r, context_k), reverse=True)
    write_csv(summary_sorted, ["config"] + cols + ["retrieval_p95_ms", "n"], out_dir / "summary.csv")
    headers = {"superseded_filter": "drop superseded", "retrieval_p50_ms": "retrieval ms (p50)",
               "gen_p50_ms": "generation ms (p50)", "usd_per_1k_queries": "$ / 1k queries",
               "abstain_correct": "abstains on unanswerable", "false_abstain": "false NOT FOUND",
               "ci": f"{primary} 95% range", "vs_best": "vs best", "total_p95_ms": "end-to-end ms (p95)",
               "tokens_per_query": "LLM tokens / query", "decision": "decision"}
    rec_cols = ["config", primary, "ci", "total_p95_ms" if generated else "retrieval_p95_ms", "tokens_per_query"] + \
               (["usd_per_1k_queries"] if any("usd_per_1k_queries" in r for r in summary) else []) + ["decision"]
    rec_cols = [c for c in rec_cols if generated or c != "tokens_per_query"]
    rec_rows = contenders + [r for r in summary if str(r.get("decision", "")).startswith("excluded")]
    rec_md = md_table(rec_rows, rec_cols, headers | {"retrieval_p95_ms": "search ms (p95)"}) if rec_rows else ""
    summary_md = md_table(summary_sorted, cols, headers)
    (out_dir / "summary.md").write_text(summary_md + "\n", encoding="utf-8")

    type_cols = ["config", "type", "n", f"recall@{context_k}"] + (["accuracy"] if generated else [])
    write_csv(type_rows, type_cols, out_dir / "by_type.csv")
    # Pivot for readability: one row per setup, one column per question type.
    pivot_metric = "accuracy" if generated else f"recall@{context_k}"
    types = sorted({r["type"] for r in type_rows})
    pivot = []
    for s in summary_sorted:
        row = {"config": s["config"]}
        for r in type_rows:
            if r["config"] == s["config"]:
                row[r["type"]] = r.get(pivot_metric)
        pivot.append(row)
    type_md = md_table(pivot, ["config"] + types)
    (out_dir / "by_type.md").write_text(f"{pivot_metric} by question type\n\n{type_md}\n", encoding="utf-8")

    find_rows = []
    qtype = {q["id"]: q["type"] for q in questions}
    for ch, fmap in findability.items():
        row = {"chunker": ch}
        per = defaultdict(list)
        for qid, v in fmap.items():
            if v is not None:
                per[qtype[qid]].append(v)
                per["all"].append(v)
        for t, vs in per.items():
            row[t] = sum(vs) / len(vs)
        find_rows.append(row)
    find_types = ["all"] + sorted({t for r in find_rows for t in r if t not in ("chunker", "all")})
    write_csv(find_rows, ["chunker"] + find_types, out_dir / "evidence_findability.csv")
    write_csv(chunk_stats, list(chunk_stats[0].keys()), out_dir / "chunk_stats.csv")

    # ---- narrative
    notes = insights(summary, type_rows, setups, records, findability, context_k, best)
    if tied:
        runner = max(tied, key=lambda r: r[primary])
        need = questions_needed(cmps[runner["config"]])
        c = cmps[runner["config"]]
        gap = (f"they differ on only {c['differs']} of {c['n']} questions"
               + (f"; separating a gap this size ({c['diff']:+.2f}) needs roughly {need} questions" if need else ""))
        notes.insert(0, f"**How sure?** The best and the runner-up ({runner['config']}) are tied: {gap}. "
                        f"Per-type results below are based on even fewer questions; read them as hints.")

    # ---- failures (best setup): everything wrong, with a blank column for your own diagnosis
    fail_cols = ["qid", "type", "question", "gold_answer", "answer", "diagnosis", "human_diagnosis",
                 "evidence_findable", f"recall@{context_k}", "superseded_in_context", "judge_reason", "top_snippets"]
    fails = [r for r in records[best["config"]] if r["diagnosis"]]
    write_csv(fails, fail_cols, out_dir / "failures.csv")

    # ---- charts
    hl = {best["config"], base["config"]}
    ranges = {r["config"]: r["range"] for r in summary}
    c1 = bar_chart(summary, f"recall@{context_k}", f"Recall@{context_k} by setup (gold passage in the top {context_k})",
                   charts / f"recall_at_{context_k}.png", hl, None if generated else ranges)
    c2 = bar_chart(summary, "accuracy", "Answer accuracy by setup (line = 95% range)", charts / "accuracy.png", hl,
                   ranges) if generated else None
    c3 = heatmap(type_rows, pivot_metric, f"{pivot_metric.capitalize()} by question type", charts / "by_type_heatmap.png")
    chart_files = [c for c in (c2 or c1, c3, c1 if c2 else None) if c]

    agreement_md = ""
    agp = out_dir / "judge_agreement.md"
    if agp.exists():
        agreement_md = agp.read_text(encoding="utf-8")

    # ---- report.md (README-ready)
    lines = [f"# RAG evaluation: {cfg['run_name']}", "", text, ""]
    if rec_text:
        lines += ["## Recommendation", "", rec_text, "", rec_md, ""]
    if chart_files:
        lines += [f"![chart](charts/{Path(chart_files[0]).name})", ""]
    lines += ["## Main results", "", summary_md, "",
              f"*{n_q} questions from split `{cfg['testset']['split']}`; retrieval metrics use the "
              f"{sum(q['answerable'] for q in questions)} answerable ones. Recall@k = share of gold passages "
              f"found in the top k. Generator: {cfg.get('_generator_label', '–')}. "
              f"Judge: {cfg.get('_judge_label', '–')}.*", "",
              "## What the numbers say", ""] + [f"- {n}" for n in notes] + ["",
              f"## By question type ({pivot_metric})", "", type_md, "",
              "## Did the gold evidence survive parsing and chunking?", "",
              md_table(find_rows, ["chunker"] + find_types), "",
              "## Chunk statistics", "", md_table(chunk_stats, list(chunk_stats[0].keys())), ""]
    if agreement_md:
        lines += ["## Judge vs. human agreement", "", agreement_md, ""]
    ex = cfg["evaluation"].get("failure_examples", 15)
    if fails:
        lines += [f"## Failure examples ({best['config']})", "",
                  "Automatic diagnosis is a first guess. Review `failures.csv` and fill `human_diagnosis`.", ""]
        for r in fails[:ex]:
            lines += [f"**{r['qid']}** ({r['type']}): {r['question']}  ",
                      f"Gold: {r['gold_answer'] or '(unanswerable)'}  ",
                      f"System: {r.get('answer', '(retrieval only)')}  ",
                      f"Diagnosis: `{r['diagnosis']}`", ""]
    lines += ["## Reproduce", "", "```bash", f"ragpick run {Path(cfg['_config_path']).name}", "```", ""]
    report_md = "\n".join(lines)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    rec_html = (f"<h2>Recommendation</h2><p class=\"lead\">{_md_inline(rec_text)}</p>"
                f"{_html_table(rec_rows, rec_cols, headers | {'retrieval_p95_ms': 'search ms (p95)'})}") if rec_text else ""
    _write_html(out_dir, cfg, text, notes, summary_sorted, cols, headers, pivot, types, find_rows, find_types,
                chunk_stats, fails[:ex], chart_files, agreement_md, n_q, pivot_metric, rec_html)

    (out_dir / "run_info.json").write_text(json.dumps({
        "run_name": cfg["run_name"], "questions": n_q, "documents": len(docs), "setups": len(setups),
        "generator": cfg.get("_generator_label"), "judge": cfg.get("_judge_label"),
        "best": best["config"], "best_range": best.get("range"), "tied_with_best": [r["config"] for r in tied],
        "baseline": base["config"], "headline": text,
        "recommended": pick["config"] if pick else None, "recommendation": rec_text,
    }, indent=2), encoding="utf-8")
    log(f"[report] {text.replace('**', '')}")
    if rec_text:
        log(f"[recommend] {rec_text.replace('**', '')}")


def _html_table(rows, cols, headers=None):
    headers = headers or {}
    th = "".join(f"<th>{html.escape(headers.get(c, c))}</th>" for c in cols)
    body = []
    for r in rows:
        tds = []
        for c in cols:
            v = r.get(c)
            cls = ' class="num"' if isinstance(v, (int, float)) and not isinstance(v, bool) else ""
            tds.append(f"<td{cls}>{html.escape(_fmt(v))}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _md_inline(s):
    s = html.escape(s)
    while "**" in s:
        s = s.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
    return s


def _write_html(out_dir, cfg, text, notes, summary, cols, headers, pivot, types, find_rows, find_types,
                chunk_stats, fails, chart_files, agreement_md, n_q, pivot_metric, rec_html=""):
    imgs = "".join(
        f'<img alt="{html.escape(Path(c).stem)}" src="data:image/png;base64,'
        f'{base64.b64encode(Path(c).read_bytes()).decode()}">' for c in chart_files)
    fail_html = "".join(
        f"<details><summary><b>{html.escape(r['qid'])}</b> · {html.escape(r['type'])} · "
        f"<code>{html.escape(r['diagnosis'])}</code> — {html.escape(r['question'])}</summary>"
        f"<p><b>Gold:</b> {html.escape(r['gold_answer'] or '(unanswerable)')}</p>"
        f"<p><b>System:</b> {html.escape(str(r.get('answer', '(retrieval only)')))}</p>"
        f"<p><b>Top retrieved:</b></p><ol>" + "".join(f"<li>{html.escape(s)}</li>" for s in r["top_snippets"]) +
        "</ol></details>" for r in fails)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RAG Evaluation Report</title>
<style>
:root{{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--accent:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--accent:#3987e5}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:26px;margin:0 0 8px}} h2{{font-size:18px;margin:36px 0 10px}}
.lead{{font-size:17px;background:var(--surface);border-left:4px solid var(--accent);padding:14px 18px;border-radius:6px}}
.scroll{{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{color:var(--ink2);font-weight:600}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}
img{{max-width:100%;background:#fcfcfb;border-radius:8px;margin:8px 0;border:1px solid var(--line)}}
li{{margin:4px 0}} .meta{{color:var(--ink2);font-size:13px}}
details{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:8px 0}}
summary{{cursor:pointer}} code{{font-size:12px}} pre{{white-space:pre-wrap}}
</style></head><body><main>
<h1>RAG evaluation: {html.escape(cfg['run_name'])}</h1>
<p class="meta">{n_q} questions · split <code>{html.escape(str(cfg['testset']['split']))}</code> ·
generator {html.escape(str(cfg.get('_generator_label', '–')))} · judge {html.escape(str(cfg.get('_judge_label', '–')))}</p>
<p class="lead">{_md_inline(text)}</p>
{rec_html}
{imgs}
<h2>Main results</h2>{_html_table(summary, ["config"] + cols[len([c for c in cols if c in ("chunker", "retriever", "reranker", "superseded_filter")]):], headers)}
<h2>What the numbers say</h2><ul>{"".join(f"<li>{_md_inline(n)}</li>" for n in notes)}</ul>
<h2>By question type ({html.escape(pivot_metric)})</h2>{_html_table(pivot, ["config"] + types)}
<h2>Gold evidence findable after chunking</h2>{_html_table(find_rows, ["chunker"] + find_types)}
<h2>Chunk statistics</h2>{_html_table(chunk_stats, list(chunk_stats[0].keys()))}
{"<h2>Judge vs. human agreement</h2><pre>" + html.escape(agreement_md) + "</pre>" if agreement_md else ""}
<h2>Failure examples</h2>{fail_html or "<p>No failures.</p>"}
</main></body></html>"""
    (out_dir / "report.html").write_text(doc, encoding="utf-8")
