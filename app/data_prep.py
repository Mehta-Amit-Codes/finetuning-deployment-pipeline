"""
Dataset preparation.

Two ways to get data:
  1. generate_dataset(): Grok writes realistic labelled support emails.
  2. load_csv(): bring your own labelled data (columns: email, category, urgency, summary).

Either way the result goes through the same validation, dedup, and a
*stratified* split (so every category appears in train, val, and test).

Outputs in DATA_DIR:
  train.jsonl / val.jsonl : chat format {"messages": [system, user, assistant]} for SFT
  test.jsonl              : raw {"email", "label"} held out for the A/B test
  manifest.json           : counts + category distribution

Caveat to state in the README/interview: synthetic data from one LLM has a
single "voice" and can be easier than real traffic. Treat A/B numbers on it as
an upper bound and re-check on real (even small) labelled data.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.config import CATEGORIES, DATA_DIR, SYSTEM_PROMPT, URGENCIES

GEN_PROMPT = """Generate <N> realistic, diverse customer support emails for an online electronics retailer, each with ground-truth triage labels.

Requirements:
- Vary tone (angry, polite, confused, terse), length (1-8 sentences) and writing quality (some typos, some all-lowercase, some rambling).
- Some emails mention several issues; label the PRIMARY issue.
- About half of the emails in this batch should be category "<FOCUS>"; spread the rest across the other categories.
- category must be one of: billing, technical, account, shipping, other.
- urgency: high = outage / money lost / time-critical; medium = real problem but workable; low = question or minor request.
- summary: one neutral sentence, at most 20 words.

Return ONLY a JSON array of <N> objects, no prose, no code fences. Each object:
{"email": "...", "category": "...", "urgency": "...", "summary": "..."}"""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _valid(item) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("email"), str) and len(item["email"].strip()) >= 20
        and item.get("category") in CATEGORIES
        and item.get("urgency") in URGENCIES
        and isinstance(item.get("summary"), str) and item["summary"].strip()
    )


def _parse_batch(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        arr = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [
        {"email": i["email"].strip(), "category": i["category"], "urgency": i["urgency"],
         "summary": i["summary"].strip()}
        for i in arr if _valid(i)
    ]


def _gen_batch(batch_idx: int, n: int) -> list[dict]:
    from app.grok_client import chat  # imported lazily so tests/CSV flow don't need the key
    focus = CATEGORIES[batch_idx % len(CATEGORIES)]
    prompt = GEN_PROMPT.replace("<N>", str(n)).replace("<FOCUS>", focus)
    try:
        return _parse_batch(chat([{"role": "user", "content": prompt}], temperature=1.0, max_tokens=4000))
    except Exception:
        return []  # one failed batch shouldn't sink the whole generation run


def generate_dataset(n_total: int = 600, batch_size: int = 10, workers: int = 4,
                     progress=None, seed: int = 42) -> list[dict]:
    """Over-generates ~20% to survive validation/dedup losses, then truncates.
    progress(fraction_done, unique_count) is called from the calling thread."""
    n_batches = math.ceil(n_total * 1.2 / batch_size)
    seen: set[str] = set()
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_gen_batch, i, batch_size) for i in range(n_batches)]
        for done, fut in enumerate(as_completed(futures), 1):
            for ex in fut.result():
                key = _norm(ex["email"])
                if key not in seen:
                    seen.add(key)
                    out.append(ex)
            if progress:
                progress(done / n_batches, len(out))
    random.Random(seed).shuffle(out)
    return out[:n_total]


def load_csv(text: str) -> list[dict]:
    """Parse your own labelled data. Rows failing validation are dropped; duplicates removed."""
    seen, out = set(), []
    for row in csv.DictReader(io.StringIO(text)):
        item = {
            "email": (row.get("email") or "").strip(),
            "category": (row.get("category") or "").strip().lower(),
            "urgency": (row.get("urgency") or "").strip().lower(),
            "summary": (row.get("summary") or "").strip(),
        }
        key = _norm(item["email"])
        if _valid(item) and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _chat_record(ex: dict) -> dict:
    label = {"category": ex["category"], "urgency": ex["urgency"], "summary": ex["summary"]}
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex["email"]},
        {"role": "assistant", "content": json.dumps(label)},
    ]}


def split_and_save(examples: list[dict], seed: int = 42, train_frac: float = 0.8,
                   val_frac: float = 0.1, out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_cat[ex["category"]].append(ex)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for items in by_cat.values():
        rng.shuffle(items)
        n = len(items)
        n_tr, n_va = int(n * train_frac), int(n * val_frac)
        train += items[:n_tr]
        val += items[n_tr:n_tr + n_va]
        test += items[n_tr + n_va:]
    for split in (train, val, test):
        rng.shuffle(split)

    def write(name, rows, fmt):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(fmt(r), ensure_ascii=False) + "\n")

    write("train.jsonl", train, _chat_record)
    write("val.jsonl", val, _chat_record)
    write("test.jsonl", test, lambda ex: {
        "email": ex["email"],
        "label": {"category": ex["category"], "urgency": ex["urgency"], "summary": ex["summary"]},
    })

    # Leakage guard: no email may appear in more than one split.
    keys = [hashlib.sha256(_norm(e["email"]).encode()).hexdigest() for e in train + val + test]
    assert len(keys) == len(set(keys)), "duplicate emails across splits"

    manifest = {
        "counts": {"train": len(train), "val": len(val), "test": len(test), "total": len(examples)},
        "category_distribution": dict(Counter(e["category"] for e in examples)),
        "urgency_distribution": dict(Counter(e["urgency"] for e in examples)),
        "seed": seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Generate (Grok) or import (CSV) the triage dataset.")
    ap.add_argument("--n", type=int, default=600, help="examples to generate with Grok")
    ap.add_argument("--csv", type=str, help="import your own labelled CSV instead of generating")
    args = ap.parse_args()

    if args.csv:
        examples = load_csv(Path(args.csv).read_text(encoding="utf-8"))
    else:
        examples = generate_dataset(
            args.n, progress=lambda f, c: print(f"\r  {f:5.0%}  {c} unique examples", end="", flush=True))
        print()
    manifest = split_and_save(examples)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
