"""The bundled offline demo, end to end, on a small grid (no network, no API keys)."""

import json
import shutil
from importlib import resources

from ragpick.pipeline import run


def test_offline_demo_produces_a_full_report(tmp_path):
    with resources.as_file(resources.files("ragpick") / "demo") as demo:
        shutil.copytree(demo, tmp_path / "demo")
    out = run(tmp_path / "demo" / "config.yaml", log=lambda *_: None, overrides=[
        "chunkers=[{name: fixed, type: fixed, size: 120, overlap: 20}, {name: structure, type: structure}]",
        "rerankers=[{name: none, type: none}]",
        "superseded_filter=[false]",
    ])
    for name in ("report.md", "report.html", "summary.csv", "by_type.csv", "failures.csv", "per_question.jsonl"):
        assert (out / name).exists(), name
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert info["setups"] == 6 and info["recommended"]
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Recommendation" in report and "95% range" in report
