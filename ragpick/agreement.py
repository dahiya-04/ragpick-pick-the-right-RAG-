"""Check the LLM judge against your own grades.

1. `ragpick judge-sample RUN_DIR --n 50`  -> human_grading.csv (judge labels hidden)
2. Fill human_correct / human_faithful with y or n
3. `ragpick agreement RUN_DIR`           -> judge_agreement.md (+ appended to report.md)
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .judge import cohen_kappa


def _load(run_dir: Path):
    recs = [json.loads(l) for l in (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines() if l]
    ctx = {}
    p = run_dir / "context_chunks.jsonl"
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            if l:
                d = json.loads(l)
                ctx[d["chunk_id"]] = d["text"]
    return recs, ctx


def make_sample(run_dir: Path, n: int = 50, seed: int = 0) -> Path:
    recs, ctx = _load(run_dir)
    pool = [r for r in recs if r.get("judge_method") in ("llm", "lexical")]
    if not pool:
        raise ValueError("No model-judged answers in this run (generation or judge was disabled).")
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))
    out = run_dir / "human_grading.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "qid", "type", "question", "gold_answer", "system_answer", "retrieved_context",
                    "human_correct", "human_faithful", "notes"])
        for i, r in enumerate(sample):
            context = "\n\n".join(f"[{j}] {ctx.get(c, c)}" for j, c in enumerate(r["context_ids"], 1))
            w.writerow([i, r["qid"], r["type"], r["question"], r["gold_answer"], r.get("answer", ""), context, "", "", ""])
    with open(run_dir / "judge_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "config", "judge_correct", "judge_faithful"])
        for i, r in enumerate(sample):
            w.writerow([i, r["config"], r.get("correct"), r.get("faithful")])
    return out


def _yn(v: str):
    v = (v or "").strip().lower()
    if v in ("y", "yes", "1", "true", "t"):
        return True
    if v in ("n", "no", "0", "false", "f"):
        return False
    return None


def compute(run_dir: Path) -> str:
    human = {r["sample_id"]: r for r in csv.DictReader(open(run_dir / "human_grading.csv", encoding="utf-8"))}
    judge = {r["sample_id"]: r for r in csv.DictReader(open(run_dir / "judge_labels.csv", encoding="utf-8"))}
    lines = ["| label | graded | agreement | Cohen's kappa |", "|---|---|---|---|"]
    disagreements = []
    for label in ("correct", "faithful"):
        pairs = []
        for sid, h in human.items():
            hv, jv = _yn(h[f"human_{label}"]), _yn(judge.get(sid, {}).get(f"judge_{label}", ""))
            if hv is None or jv is None:
                continue
            pairs.append((hv, jv))
            if hv != jv:
                disagreements.append(f"- sample {sid} ({h['qid']}): {label} human={hv} judge={jv}")
        if not pairs:
            lines.append(f"| {label} | 0 | – | – |")
            continue
        agree = sum(a == b for a, b in pairs) / len(pairs)
        kappa = cohen_kappa([a for a, _ in pairs], [b for _, b in pairs])
        lines.append(f"| {label} | {len(pairs)} | {agree:.1%} | {kappa:.2f} |")
    md = "\n".join(lines)
    md += ("\n\nKappa above 0.6 is usually read as substantial agreement; below 0.4, do not trust the judge's numbers."
           "\n\nDisagreements:\n" + ("\n".join(disagreements) if disagreements else "- none") + "\n")
    (run_dir / "judge_agreement.md").write_text(md, encoding="utf-8")
    rep = run_dir / "report.md"
    if rep.exists():
        text = rep.read_text(encoding="utf-8")
        marker = "## Judge vs. human agreement"
        if marker in text:
            head, _, rest = text.partition(marker)
            nxt = rest.find("\n## ")
            text = head + marker + "\n\n" + md + (rest[nxt:] if nxt != -1 else "")
        else:
            text = text.replace("## Reproduce", f"{marker}\n\n{md}\n## Reproduce")
        rep.write_text(text, encoding="utf-8")
    return md
