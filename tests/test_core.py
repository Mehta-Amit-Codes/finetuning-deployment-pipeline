import json

from app import data_prep
from app.config import build_messages, parse_triage
from app.metrics import bootstrap_ci, paired_bootstrap


def test_parse_triage_valid_and_wrapped():
    good = '{"category": "billing", "urgency": "high", "summary": "Double charge."}'
    assert parse_triage(good)["category"] == "billing"
    assert parse_triage("Sure! " + good + " hope that helps")["urgency"] == "high"


def test_parse_triage_rejects_bad_output():
    assert parse_triage("") is None
    assert parse_triage("no json here") is None
    assert parse_triage('{"category": "nonsense", "urgency": "high", "summary": "x"}') is None
    assert parse_triage('{"category": "billing", "urgency": "high"}') is None
    assert parse_triage('{"category": "billing", "urgency": "high", "summary": "x"') is None  # truncated


def test_build_messages_fewshot_adds_turns():
    assert len(build_messages("hi")) == 2
    assert len(build_messages("hi", fewshot=True)) == 2 + 2 * 3


def test_bootstrap_ci_contains_mean_and_handles_empty():
    r = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1])
    assert r["ci_low"] <= r["mean"] <= r["ci_high"]
    assert bootstrap_ci([])["n"] == 0


def test_paired_bootstrap_detects_clear_win_and_noise():
    win = paired_bootstrap([1] * 40 + [0] * 10)
    assert win["ci_low"] > 0 and win["p_better"] > 0.99
    noise = paired_bootstrap([1, -1] * 25)
    assert noise["ci_low"] < 0 < noise["ci_high"]


def test_parse_batch_filters_invalid_items():
    text = ('Here you go:\n[{"email": "my package never arrived, where is it please", "category": "shipping", '
            '"urgency": "medium", "summary": "Package missing."}, {"email": "short", "category": "x", '
            '"urgency": "low", "summary": "s"}]')
    out = data_prep._parse_batch(text)
    assert len(out) == 1 and out[0]["category"] == "shipping"
    assert data_prep._parse_batch("not json") == []


def test_split_is_stratified_and_leak_free(tmp_path):
    cats = ["billing", "technical", "account", "shipping", "other"]
    examples = [{"email": f"unique customer email number {i} about {cats[i % 5]}", "category": cats[i % 5],
                 "urgency": "low", "summary": "s"} for i in range(100)]
    m = data_prep.split_and_save(examples, out_dir=tmp_path)
    assert m["counts"]["total"] == 100
    assert m["counts"]["train"] + m["counts"]["val"] + m["counts"]["test"] == 100
    test_rows = [json.loads(l) for l in (tmp_path / "test.jsonl").read_text().splitlines()]
    assert {r["label"]["category"] for r in test_rows} == set(cats)  # every class held out
    train_rec = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in train_rec["messages"]] == ["system", "user", "assistant"]


def test_load_csv_validates_and_dedups():
    csv_text = ("email,category,urgency,summary\n"
                "my invoice shows the wrong amount again,billing,medium,Invoice amount wrong.\n"
                "my invoice shows the wrong amount again,billing,medium,Invoice amount wrong.\n"
                "tiny,billing,medium,x\n")
    assert len(data_prep.load_csv(csv_text)) == 1
