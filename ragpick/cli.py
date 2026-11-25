"""Command line: `ragpick <command>`.

  init DIR            scaffold a project (config.yaml, testset template, folders)
  demo                run the bundled offline demo end to end
  draft CONFIG        LLM-drafted test questions -> testset/draft_questions.csv for review
  accept DRAFT_CSV    add the reviewed (keep=y) drafts to testset/questions.jsonl
  split TESTSET       stratified 30/70 dev/test split, written into the file
  parse CONFIG        parse the corpus only and write parse_report.csv
  run CONFIG          the full experiment grid -> results/<run_name>_<time>/
  judge-sample RUN    sample answers for hand grading (judge labels hidden)
  agreement RUN       judge-vs-human agreement and Cohen's kappa
"""

from __future__ import annotations

import argparse
import shutil
import sys
from importlib import resources
from pathlib import Path


def _copy_pkg_dir(sub: str, dest: Path):
    src = resources.files("ragpick") / sub
    with resources.as_file(src) as p:
        shutil.copytree(p, dest, dirs_exist_ok=True)


def cmd_init(a):
    dest = Path(a.dir)
    dest.mkdir(parents=True, exist_ok=True)
    tpl = resources.files("ragpick") / "templates"
    for name, target in (("config.yaml", "config.yaml"), ("questions.jsonl", "testset/questions.jsonl"),
                         ("metadata.csv", "data/metadata.csv")):
        out = dest / target
        if out.exists():
            print(f"  skip {out} (exists)")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text((tpl / name).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  wrote {out}")
    (dest / "data" / "corpus").mkdir(parents=True, exist_ok=True)
    print(f"\nNext:\n  1. put your PDFs in {dest / 'data' / 'corpus'}\n"
          f"  2. write questions in {dest / 'testset' / 'questions.jsonl'}, or draft them:\n"
          f"     ragpick draft {dest / 'config.yaml'}  ->  review  ->  ragpick accept {dest / 'testset' / 'draft_questions.csv'}\n"
          f"  3. ragpick split {dest / 'testset' / 'questions.jsonl'}\n"
          f"  4. ragpick run {dest / 'config.yaml'}")


def cmd_demo(a):
    from .pipeline import run

    dest = Path(a.out)
    _copy_pkg_dir("demo", dest)
    print(f"[demo] copied the synthetic sample project to {dest}")
    run(dest / "config.yaml", overrides=a.set)


def cmd_draft(a):
    from .drafting import draft

    draft(a.config, n=a.n, out=a.out, seed=a.seed, overrides=a.set)


def cmd_accept(a):
    from .drafting import accept

    accept(a.draft_csv, into=a.into)


def cmd_split(a):
    from .testset import assign_splits

    counts = assign_splits(Path(a.testset), a.dev, a.seed, a.force)
    for (split, t), n in sorted(counts.items()):
        print(f"  {split:5} {t:20} {n}")


def cmd_parse(a):
    from .config import load_config
    from .parsing import parse_corpus, write_parse_report

    cfg = load_config(a.config, a.set)
    base = Path(cfg["_base_dir"])
    docs = parse_corpus(cfg["corpus"], base, (base / cfg["cache_dir"]).resolve())
    out = base / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    write_parse_report(docs, out / "parse_report.csv")
    flags = {}
    for d in docs:
        for f in filter(None, d.issues.get("flags", "").split(";")):
            flags[f] = flags.get(f, 0) + 1
    print(f"{len(docs)} documents; superseded: {sum(bool(d.meta.get('superseded')) for d in docs)}; "
          f"with circular no.: {sum(bool(d.meta.get('circular_no')) for d in docs)}; flags: {flags}")
    print(f"-> {out / 'parse_report.csv'}")


def cmd_run(a):
    from .pipeline import run

    run(a.config, overrides=a.set, limit=a.limit, split=a.split, dry_run=a.dry_run)


def cmd_judge_sample(a):
    from .agreement import make_sample

    out = make_sample(Path(a.run_dir), a.n, a.seed)
    print(f"Fill human_correct and human_faithful (y/n) in {out}, then run: ragpick agreement {a.run_dir}")


def cmd_agreement(a):
    from .agreement import compute

    print(compute(Path(a.run_dir)))


def main(argv=None):
    p = argparse.ArgumentParser(prog="ragpick", description="One-command RAG evaluation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="scaffold a new project")
    s.add_argument("dir", nargs="?", default=".")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("demo", help="run the offline demo")
    s.add_argument("--out", default="ragpick_demo")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("draft", help="draft test questions from the corpus for review")
    s.add_argument("config")
    s.add_argument("--n", type=int, default=40, help="how many questions (default 40)")
    s.add_argument("--out", help="default: testset/draft_questions.csv")
    s.add_argument("--seed", type=int, default=0, help="change it to draft from different passages")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    s.set_defaults(fn=cmd_draft)

    s = sub.add_parser("accept", help="add reviewed drafts (keep=y) to the test set")
    s.add_argument("draft_csv")
    s.add_argument("--into", help="default: questions.jsonl next to the draft file")
    s.set_defaults(fn=cmd_accept)

    s = sub.add_parser("split", help="assign dev/test splits")
    s.add_argument("testset")
    s.add_argument("--dev", type=float, default=0.3)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--force", action="store_true", help="re-split questions that already have a split")
    s.set_defaults(fn=cmd_split)

    for name, fn, hlp in (("parse", cmd_parse, "parse corpus only"), ("run", cmd_run, "run the full evaluation")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("config")
        s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override config, e.g. --set generation.enabled=false")
        if name == "run":
            s.add_argument("--split", choices=["dev", "test", "all"])
            s.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
            s.add_argument("--dry-run", action="store_true", help="show the grid and call counts, run nothing")
        s.set_defaults(fn=fn)

    s = sub.add_parser("judge-sample", help="sample answers for human grading")
    s.add_argument("run_dir")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_judge_sample)

    s = sub.add_parser("agreement", help="judge vs human agreement")
    s.add_argument("run_dir")
    s.set_defaults(fn=cmd_agreement)

    a = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles can't print every character; never crash on it
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        a.fn(a)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
