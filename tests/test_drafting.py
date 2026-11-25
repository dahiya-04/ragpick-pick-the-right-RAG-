"""Automatic checks on drafted questions, and `accept`."""

import csv
import json

from ragpick.drafting import FIELDS, _in_passage, _NOT_STANDALONE, _plan, _VAGUE, accept
from ragpick.testset import load_testset

PASSAGE = "2.1 Any request for closure of a credit card shall be honoured within seven working days by the card-issuer."


def test_evidence_must_come_from_the_passage():
    assert _in_passage("Any request for closure of a credit card shall be honoured within seven working days", PASSAGE)
    assert not _in_passage("Closure requests must be honoured within thirty days of receipt.", PASSAGE)
    assert not _in_passage("", PASSAGE)


def test_question_style_checks():
    assert _NOT_STANDALONE.search("How does the limit in passage 1 compare?")
    assert _NOT_STANDALONE.search("How do the two documents differ on fees?")
    assert not _NOT_STANDALONE.search("Is a fee charged on loans above Rs 5 lakh?")
    assert _VAGUE.search("How often must the policy under these Directions be reviewed?")


def test_plan_has_the_requested_size_and_mix():
    import random

    kinds = _plan(40, {"lookup": 0.5, "unanswerable": 0.5}, random.Random(0))
    assert len(kinds) == 40 and kinds.count("unanswerable") == 20


def _write_draft(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def test_accept_appends_kept_rows_renumbers_and_skips_duplicates(tmp_path):
    target = tmp_path / "questions.jsonl"
    target.write_text(json.dumps({"id": "d001", "question": "Old question?", "answer": "a",
                                  "evidence": ["some text"]}) + "\n", encoding="utf-8")
    draft = tmp_path / "draft.csv"
    _write_draft(draft, [
        {"keep": "y", "id": "d001", "type": "lookup", "question": "How fast must card closure happen?",
         "answer": "Seven working days.", "evidence_1": PASSAGE, "doc_1": "cards"},
        {"keep": "n", "id": "d002", "type": "lookup", "question": "Rejected?", "answer": "x", "evidence_1": "y"},
        {"keep": "Y", "id": "d003", "type": "unanswerable", "question": "What is the fee for a gold card?"},
    ])
    accept(draft, log=lambda *_: None)
    qs = load_testset(target)
    assert [q["question"] for q in qs][1:] == ["How fast must card closure happen?", "What is the fee for a gold card?"]
    assert len({q["id"] for q in qs}) == 3 and qs[2]["answerable"] is False
    accept(draft, log=lambda *_: None)  # accepting the same file again adds nothing
    assert len(load_testset(target)) == 3
