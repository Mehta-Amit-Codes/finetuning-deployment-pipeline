"""
A/B evaluation: identical held-out emails through every arm, scored the same way.

Default arms:
  base          zero-shot prompt on the untuned model
  base_fewshot  same model, 3-shot prompt  <- the honest "just prompt it better" baseline
  finetuned     LoRA model, zero-shot prompt

Metrics per example:
  json_valid, category_correct, urgency_correct, exact_match  (deterministic)
  judge                                                      (Grok, 0..1, optional)
  latency_ms, prompt_tokens, completion_tokens               (cost proxies)

Reporting: percentile-bootstrap CI per arm, and a *paired* bootstrap on
(finetuned - baseline) per example, with p_better = share of resamples where
the fine-tuned model wins. If the CI excludes 0, the lift is real at this
sample size; if it straddles 0, say so rather than claiming a win.

    python -m app.ab_test --limit 100
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from app.config import (BASE_ENDPOINT, BASE_MODEL_NAME, DATA_DIR, EVAL_DIR, FT_ENDPOINT,
                        FT_MODEL_NAME, build_messages, parse_triage)
from app.metrics import bootstrap_ci, paired_bootstrap, percentile

METRICS = ["json_valid", "category_correct", "urgency_correct", "exact_match", "judge"]


def default_arms() -> list[dict]:
    return [
        {"name": "base", "endpoint": BASE_ENDPOINT, "model": BASE_MODEL_NAME, "fewshot": False},
        {"name": "base_fewshot", "endpoint": BASE_ENDPOINT, "model": BASE_MODEL_NAME, "fewshot": True},
        {"name": "finetuned", "endpoint": FT_ENDPOINT, "model": FT_MODEL_NAME, "fewshot": False},
    ]


@lru_cache(maxsize=8)
def _client(endpoint: str) -> OpenAI:
    return OpenAI(base_url=endpoint, api_key="EMPTY", timeout=60)  # vLLM ignores the key


def load_test(path: Path | None = None, limit: int | None = None) -> list[dict]:
    path = Path(path or DATA_DIR / "test.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _eval_one(arm: dict, example: dict, use_judge: bool) -> dict:
    row = {"email": example["email"], "error": None}
    t0 = time.perf_counter()
    try:
        resp = _client(arm["endpoint"]).chat.completions.create(
            model=arm["model"], messages=build_messages(example["email"], arm["fewshot"]),
            temperature=0, max_tokens=150)
        text = resp.choices[0].message.content or ""
        row["latency_ms"] = (time.perf_counter() - t0) * 1000
        row["prompt_tokens"] = resp.usage.prompt_tokens
        row["completion_tokens"] = resp.usage.completion_tokens
    except Exception as e:
        # A failed call counts as a failed example, not a skipped one.
        row.update(error=str(e), latency_ms=None, prompt_tokens=None, completion_tokens=None)
        text = ""

    parsed = parse_triage(text)
    label = example["label"]
    row["output"] = text
    row["json_valid"] = 1 if parsed else 0
    row["category_correct"] = 1 if parsed and parsed["category"] == label["category"] else 0
    row["urgency_correct"] = 1 if parsed and parsed["urgency"] == label["urgency"] else 0
    row["exact_match"] = row["category_correct"] * row["urgency_correct"]
    row["judge"] = None
    if use_judge and not row["error"]:
        from app.judge import judge_triage
        row["judge"] = judge_triage(example["email"], text)
    return row


def run_arm(arm: dict, examples: list[dict], use_judge: bool = True, workers: int = 8) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda ex: _eval_one(arm, ex, use_judge), examples))  # order preserved


def summarize(rows_by_arm: dict[str, list[dict]], candidate: str = "finetuned") -> dict:
    arms = {}
    for name, rows in rows_by_arm.items():
        s = {"n": len(rows), "errors": sum(1 for r in rows if r["error"])}
        for m in METRICS:
            s[m] = bootstrap_ci(r[m] for r in rows if r.get(m) is not None)
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        s["latency_ms"] = {"mean": (sum(lat) / len(lat)) if lat else None,
                           "p50": percentile(lat, 50), "p95": percentile(lat, 95)}
        for tok in ("prompt_tokens", "completion_tokens"):
            vals = [r[tok] for r in rows if r.get(tok) is not None]
            s[f"{tok}_mean"] = (sum(vals) / len(vals)) if vals else None
        arms[name] = s

    comparisons = {}
    if candidate in rows_by_arm:
        cand = rows_by_arm[candidate]
        for name, rows in rows_by_arm.items():
            if name == candidate:
                continue
            comparisons[f"{candidate}_vs_{name}"] = {
                m: paired_bootstrap(
                    c[m] - b[m] for c, b in zip(cand, rows)
                    if c.get(m) is not None and b.get(m) is not None)
                for m in METRICS
            }
    return {"arms": arms, "comparisons": comparisons, "candidate": candidate}


def evaluate(arms: list[dict] | None = None, limit: int | None = None, use_judge: bool = True,
             test_path: Path | None = None, candidate: str = "finetuned",
             out_dir: Path | None = None, progress=None) -> dict:
    arms = arms or default_arms()
    examples = load_test(test_path, limit)
    out_dir = Path(out_dir or EVAL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_arm = {}
    for i, arm in enumerate(arms):
        if progress:
            progress(i / len(arms), f"Running arm '{arm['name']}' on {len(examples)} examples")
        rows = run_arm(arm, examples, use_judge)
        rows_by_arm[arm["name"]] = rows
        with open(out_dir / f"{arm['name']}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(rows_by_arm, candidate)
    summary["n_examples"] = len(examples)
    summary["judge_used"] = use_judge
    summary["generated_at"] = time.time()
    (out_dir / "ab_summary.json").write_text(json.dumps(summary, indent=2))
    if progress:
        progress(1.0, "Done")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--arms", help="JSON file overriding the default arms (see configs/arms.example.json)")
    ap.add_argument("--candidate", default="finetuned")
    args = ap.parse_args()

    arms = json.loads(Path(args.arms).read_text()) if args.arms else None
    summary = evaluate(arms, args.limit, not args.no_judge, candidate=args.candidate)

    for name, s in summary["arms"].items():
        em = s["exact_match"]
        print(f"{name:14s} exact_match={em['mean']:.3f} [{em['ci_low']:.3f}, {em['ci_high']:.3f}]  "
              f"json_valid={s['json_valid']['mean']:.3f}  p95={s['latency_ms']['p95'] or 0:.0f}ms")
    for name, comp in summary["comparisons"].items():
        d = comp["exact_match"]
        print(f"{name}: exact_match lift {d['mean_diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] "
              f"p_better={d['p_better']:.2f}")


if __name__ == "__main__":
    main()
