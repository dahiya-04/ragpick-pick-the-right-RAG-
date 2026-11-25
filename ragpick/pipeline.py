"""The whole experiment: parse -> chunk -> index -> retrieve -> rerank -> generate -> judge -> report."""

from __future__ import annotations

import json
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .chunking import Chunk, make_chunker
from .config import load_config
from .embeddings import make_embedder
from .generation import format_context, make_generator
from .judge import make_judge
from .llm import LLMCache
from .metrics import EvidenceMatcher, prepare_evidence, retrieval_metrics
from .parsing import parse_corpus, write_parse_report
from .rerank import make_reranker
from .retrieval import make_retriever
from .testset import load_testset, select_split
from .textutil import shorten, stable_hash


def _config_id(ch, rt, rr, filt):
    return f"{ch} | {rt} | {rr}" + (" | no-superseded" if filt else "")


def _chunk_corpus(chunker, spec, embedders, docs, cache_dir: Path, log) -> list[Chunk]:
    emb_id = embedders[spec["embedder"]].cache_id if spec.get("embedder") else None
    key = stable_hash({"docs": stable_hash([(d.doc_id, d.text, d.title, d.meta) for d in docs]),
                       "spec": spec, "emb": emb_id, "v": 1})
    path = cache_dir / "chunks" / f"{key}.json"
    if path.exists():
        return [Chunk(**c) for c in json.loads(path.read_text(encoding="utf-8"))]
    t0 = time.perf_counter()
    chunks = chunker.chunk_corpus(docs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False), encoding="utf-8")
    log(f"  chunked in {time.perf_counter() - t0:.1f}s")
    return chunks


def _probe_latency(embedder, n: int = 3) -> float:
    """Median time of a few uncached single-question embedding calls."""
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        embedder.encode_queries([f"latency probe {i} {time.time_ns()}: what is the time limit for this request?"])
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _diagnose(q, rec, context_k) -> str:
    """Automatic first guess at why an answer failed. Check it by hand in failures.csv."""
    if rec.get("correct") is True:
        return ""
    if not q["answerable"]:
        return "answered_unanswerable" if rec.get("correct") is False else ""
    if not rec["evidence_findable"]:
        return "parsing_or_chunking_lost_evidence"
    ctx_recall = rec.get(f"recall@{context_k}", 0.0)
    if ctx_recall == 0:
        return "retrieval_miss_superseded_retrieved" if rec["superseded_in_context"] else "retrieval_miss"
    if rec.get("correct") is None:
        return ""  # not generated/judged, and the evidence was retrieved
    if ctx_recall < 1:
        return "partial_retrieval"
    if rec.get("abstained"):
        return "false_abstain"
    if rec.get("faithful") is False:
        return "unfaithful_generation"
    if rec.get("superseded_in_context"):
        return "generation_followed_superseded_rule"
    return "generation_error"


def run(config_path, overrides=None, limit: int | None = None, split: str | None = None,
        dry_run: bool = False, log=print) -> Path | None:
    cfg = load_config(config_path, overrides)
    base = Path(cfg["_base_dir"])
    cache_dir = (base / cfg["cache_dir"]).resolve()
    ev_cfg, gen_cfg, judge_cfg = cfg["evaluation"], cfg["generation"], cfg["judge"]
    context_k = ev_cfg["context_k"]
    ks = sorted(set(ev_cfg["k"]) | {context_k})

    # ---------------------------------------------------------------- questions
    questions = load_testset(base / cfg["testset"]["path"])
    questions = select_split(questions, split or cfg["testset"]["split"])
    if limit:
        questions = questions[:limit]
    if not questions:
        raise ValueError("No questions selected. Check testset.split.")
    prepare_evidence(questions)

    grid = [(c, r, rr, f) for c in cfg["chunkers"] for r in cfg["retrievers"]
            for rr in cfg["rerankers"] for f in cfg["superseded_filter"]]
    n_answerable = sum(q["answerable"] for q in questions)
    gen_sel = gen_cfg["configs"]
    n_gen_cfgs = len(grid) if gen_sel == "all" else min(len(grid), int(gen_sel.split(":")[1]))
    log(f"[plan] {len(questions)} questions ({n_answerable} answerable), {len(grid)} setups "
        f"= {len(cfg['chunkers'])} chunkers x {len(cfg['retrievers'])} retrievers x "
        f"{len(cfg['rerankers'])} rerankers x {len(cfg['superseded_filter'])} filters")
    if gen_cfg["enabled"]:
        log(f"[plan] generation for {n_gen_cfgs} setups -> up to {n_gen_cfgs * len(questions)} answer calls"
            + (f" + up to {n_gen_cfgs * len(questions)} judge calls" if judge_cfg["enabled"] else "")
            + " (cached calls are free on re-runs)")
    if dry_run:
        for c, r, rr, f in grid:
            log("   - " + _config_id(c["name"], r["name"], rr["name"], f))
        return None

    out_dir = (base / cfg["output_dir"] / f"{cfg['run_name']}_{datetime.now():%Y%m%d-%H%M%S}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg["_config_path"], out_dir / "config.yaml")

    # ---------------------------------------------------------------- parse
    docs = parse_corpus(cfg["corpus"], base, cache_dir, log)
    write_parse_report(docs, out_dir / "parse_report.csv")
    n_sup = sum(bool(d.meta.get("superseded")) for d in docs)
    log(f"[parse] {len(docs)} documents, {n_sup} marked superseded -> parse_report.csv")

    embedders = {name: make_embedder(spec) for name, spec in cfg["embedders"].items()}
    # Question embeddings are cached, so timed searches skip the embedding call a real system makes.
    # Measure it live once per embedder and add it to every query of the setups that use it.
    embed_ms = {n: _probe_latency(embedders[n]) for n in {r.get("embedder") for r in cfg["retrievers"]} if n}
    for n, ms in embed_ms.items():
        log(f"[latency] query embedding with {n}: {ms:.0f} ms (median of 3 live calls)")
    rerankers = {spec["name"]: make_reranker(spec) for spec in cfg["rerankers"]}
    depth = max([max(ks)] + [rr.top_n for rr in rerankers.values() if rr])

    setups: list[dict] = []           # one per grid cell
    records: dict[str, list] = {}     # config_id -> per-question records
    chunk_stats, findability = [], {}
    chunk_store: dict[str, list[Chunk]] = {}

    # ---------------------------------------------------------------- retrieval phase
    for ch_spec in cfg["chunkers"]:
        log(f"[chunk] {ch_spec['name']} ({ch_spec['type']})")
        chunker = make_chunker(ch_spec, embedders)
        chunks = _chunk_corpus(chunker, ch_spec, embedders, docs, cache_dir, log)
        chunk_store[ch_spec["name"]] = chunks
        sizes = [len(c.text.split()) for c in chunks]
        chunk_stats.append({"chunker": ch_spec["name"], "chunks": len(chunks),
                            "mean_words": round(statistics.mean(sizes), 1) if sizes else 0,
                            "median_words": statistics.median(sizes) if sizes else 0,
                            "max_words": max(sizes) if sizes else 0})
        matcher = EvidenceMatcher(chunks, ev_cfg["overlap_threshold"], ev_cfg["purity_threshold"])
        findable = {q["id"]: (all(matcher.findable(ev) for ev in q["evidence"]) if q["evidence"] else None)
                    for q in questions}
        findability[ch_spec["name"]] = findable
        lost = [qid for qid, v in findable.items() if v is False]
        log(f"  {len(chunks)} chunks; gold evidence findable for "
            f"{n_answerable - len(lost)}/{n_answerable} answerable questions")

        for r_spec in cfg["retrievers"]:
            retriever = make_retriever(r_spec, embedders)
            emb_name = r_spec.get("embedder")
            emb_spec = cfg["embedders"].get(emb_name, {}) if emb_name else {}
            emb_model = emb_spec.get("deployment") or emb_spec.get("model") or emb_spec.get("type")
            t0 = time.perf_counter()
            retriever.index(chunks, cache_dir)
            log(f"  [index] {r_spec['name']} built in {time.perf_counter() - t0:.1f}s")
            raw = {}
            for q in questions:
                t0 = time.perf_counter()
                hits = retriever.search(q["question"], depth * 3 if any(cfg["superseded_filter"]) else depth)
                raw[q["id"]] = ([i for i, _ in hits], (time.perf_counter() - t0) * 1000)

            for rr_spec in cfg["rerankers"]:
                reranker = rerankers[rr_spec["name"]]
                for filt in cfg["superseded_filter"]:
                    cid = _config_id(ch_spec["name"], r_spec["name"], rr_spec["name"], filt)
                    recs = []
                    for q in questions:
                        ranked, ret_ms = raw[q["id"]]
                        if filt:
                            ranked = [i for i in ranked if not chunks[i].meta.get("superseded")]
                        ranked = ranked[:depth]
                        rr_ms = 0.0
                        if reranker:
                            t0 = time.perf_counter()
                            head = ranked[:reranker.top_n]
                            sc = reranker.scores(q["question"], [chunks[i].index_text for i in head])
                            head = [i for _, i in sorted(zip(sc, head), key=lambda x: -x[0])]
                            ranked = head + ranked[reranker.top_n:]
                            rr_ms = (time.perf_counter() - t0) * 1000
                        rec = {"config": cid, "chunker": ch_spec["name"], "retriever": r_spec["name"],
                               "reranker": rr_spec["name"], "superseded_filter": filt, "qid": q["id"],
                               "type": q["type"], "answerable": q["answerable"], "question": q["question"],
                               "gold_answer": q["answer"], "evidence_findable": findable[q["id"]],
                               "retrieval_ms": ret_ms + rr_ms, "rerank_ms": rr_ms,
                               "embed_ms": embed_ms.get(emb_name, 0.0), "embed_model": emb_model,
                               "embed_tok": round(1.3 * len(q["question"].split())) if emb_name else 0,
                               "context_ids": [chunks[i].chunk_id for i in ranked[:context_k]],
                               "_ranked": ranked,
                               "superseded_in_context": any(chunks[i].meta.get("superseded")
                                                            for i in ranked[:context_k])}
                        if q["evidence"]:
                            rec.update(retrieval_metrics(ranked, q, matcher, ks))
                        recs.append(rec)
                    records[cid] = recs
                    setups.append({"config": cid, "chunker": ch_spec["name"], "retriever": r_spec["name"],
                                   "reranker": rr_spec["name"], "superseded_filter": filt})
                    ans = [r for r in recs if q_answerable(r)]
                    r_at = statistics.mean(r[f"recall@{context_k}"] for r in ans) if ans else 0
                    log(f"    {cid:<55} Recall@{context_k}={r_at:.3f}")

    # ---------------------------------------------------------------- generation + judging phase
    if gen_cfg["enabled"]:
        llm_cache = LLMCache(cache_dir / "llm_cache.sqlite")
        generator = make_generator(gen_cfg["llm"], llm_cache)
        judge = make_judge(judge_cfg["llm"], llm_cache) if judge_cfg["enabled"] else None

        def score(cid):
            ans = [r for r in records[cid] if r["answerable"]]
            return statistics.mean(r[f"recall@{context_k}"] for r in ans) if ans else 0

        chosen = [s["config"] for s in setups]
        if gen_sel != "all":
            chosen = sorted(chosen, key=score, reverse=True)[:int(gen_sel.split(":")[1])]
        qmap = {q["id"]: q for q in questions}
        tasks = [(cid, rec) for cid in chosen for rec in records[cid]]
        log(f"[generate] {generator.label}: {len(tasks)} answers across {len(chosen)} setups"
            + (f"; judge: {judge.label}" if judge else ""))

        def work(task):
            cid, rec = task
            chunks = chunk_store[rec["chunker"]]
            ctx = [chunks[i] for i in rec["_ranked"][:context_k]]
            out = generator.answer(rec["question"], ctx)
            rec.update(out)
            if judge:
                rec.update(judge.judge(qmap[rec["qid"]], out, format_context(ctx)))
            return rec

        done = 0
        with ThreadPoolExecutor(max_workers=gen_cfg.get("concurrency", 4)) as pool:
            for _ in pool.map(work, tasks):
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    log(f"  {done}/{len(tasks)}")
        for s in setups:
            s["generated"] = s["config"] in chosen
        cfg["_generator_label"] = generator.label
        cfg["_judge_label"] = judge.label if judge else None

    # ---------------------------------------------------------------- diagnosis + save
    qmap = {q["id"]: q for q in questions}
    used: dict[str, str] = {}
    for cid, recs in records.items():
        chunks = chunk_store[recs[0]["chunker"]] if recs else []
        for rec in recs:
            for i in rec["_ranked"][:context_k]:
                used[chunks[i].chunk_id] = format_context([chunks[i]])[4:]  # header + text, without "[1] "
            rec["diagnosis"] = _diagnose(qmap[rec["qid"]], rec, context_k)
            rec["top_snippets"] = [shorten(chunks[i].text, 240) for i in rec["_ranked"][:3]]
            del rec["_ranked"]

    with open(out_dir / "context_chunks.jsonl", "w", encoding="utf-8") as f:
        for cid_, text in used.items():
            f.write(json.dumps({"chunk_id": cid_, "text": text}, ensure_ascii=False) + "\n")
    with open(out_dir / "per_question.jsonl", "w", encoding="utf-8") as f:
        for recs in records.values():
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from .report import build_report

    build_report(out_dir, cfg, setups, records, questions, chunk_stats, findability, ks, context_k, docs, log)
    log(f"[done] results in {out_dir}")
    return out_dir


def q_answerable(rec) -> bool:
    return bool(rec["answerable"])
