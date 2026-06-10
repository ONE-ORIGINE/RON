from __future__ import annotations
import math
from statistics import mean, pstdev

def mean_std(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": mean(xs), "std": pstdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}

def bootstrap_ci(xs, samples: int = 1000, alpha: float = 0.05, seed: int = 123):
    """Deterministic bootstrap CI without numpy/scipy."""
    import random
    xs = [float(x) for x in xs]
    if len(xs) <= 1:
        v = xs[0] if xs else 0.0
        return [v, v]
    rng = random.Random(seed)
    vals = []
    for _ in range(samples):
        vals.append(mean(rng.choice(xs) for _ in xs))
    vals.sort()
    lo = vals[int((alpha/2) * (len(vals)-1))]
    hi = vals[int((1-alpha/2) * (len(vals)-1))]
    return [lo, hi]

def paired_differences(a, b):
    return [float(x) - float(y) for x, y in zip(a, b)]

def cohen_d(xs):
    xs = [float(x) for x in xs]
    if len(xs) <= 1:
        return 0.0
    sd = pstdev(xs)
    return 0.0 if sd == 0 else mean(xs) / sd

def sign_test_successes(xs, target: float = 0.0, prefer_lower: bool = True):
    if prefer_lower:
        return sum(1 for x in xs if x < target)
    return sum(1 for x in xs if x > target)

def summarize_metric_by_variant(rows, metric):
    variants = sorted(set(r["variant"] for r in rows))
    out = {}
    for v in variants:
        vals = [float(r[metric]) for r in rows if r["variant"] == v]
        s = mean_std(vals)
        s["ci95"] = bootstrap_ci(vals, seed=99)
        out[v] = s
    return out
