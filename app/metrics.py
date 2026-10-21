"""
Statistics for the A/B test. Everything is a percentile bootstrap: no
distributional assumptions, works for 0/1 metrics (accuracy) and 0..1
judge scores alike, and is easy to explain in an interview.

Paired bootstrap matters here: both models answer the *same* test emails, so
comparing per-example differences removes the "some emails are just harder"
noise that an unpaired comparison would leave in.
"""
import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(arr.size)}


def paired_bootstrap(diffs, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """diffs = candidate_score - baseline_score per example (same examples)."""
    arr = np.asarray(list(diffs), dtype=float)
    if arr.size == 0:
        return {"mean_diff": None, "ci_low": None, "ci_high": None, "p_better": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_diff": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi),
        # share of bootstrap resamples where the candidate beats the baseline
        "p_better": float((means > 0).mean()), "n": int(arr.size),
    }


def percentile(values, q: float):
    arr = np.asarray(list(values), dtype=float)
    return float(np.percentile(arr, q)) if arr.size else None
