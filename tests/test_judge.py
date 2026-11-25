from ragpick.judge import _rule, cohen_kappa, parse_verdict


def test_parse_valid_json():
    assert parse_verdict('{"answer_correct": true, "faithful": false, "reason": "r"}') == {
        "answer_correct": True, "faithful": False, "reason": "r"}


def test_recovers_labels_from_schema_shaped_reply():
    # a model that copied the schema description instead of filling it in
    text = '{"answer_correct": {"type": "boolean", "value": false}, "faithful": {"type": "boolean", "value": true}, "reason": "ok"}'
    v = parse_verdict(text)
    assert v["answer_correct"] is False and v["faithful"] is True


def test_unreadable_reply_returns_none():
    assert parse_verdict("I think the answer is fine.") is None


def test_rules_for_abstention():
    unans = {"answerable": False}
    ans = {"answerable": True}
    assert _rule(unans, {"abstained": True})["correct"] is True
    assert _rule(ans, {"abstained": True})["correct"] is False
    assert _rule(ans, {"abstained": False}) is None


def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
