"""Deterministic IR metrics with explicit definitions."""
from __future__ import annotations

import math
import statistics


def retrieval_case(ranked: list[str], relevant: set[str], ks=(1, 3, 5, 10)) -> dict:
    if not relevant:
        raise ValueError("retrieval metrics require at least one relevance label")
    result = {}
    for k in ks:
        selected = ranked[:k]
        found = set(selected) & relevant
        result[f"hit_at_{k}"] = float(bool(found))
        result[f"recall_at_{k}"] = len(found) / len(relevant)
        gains = [1.0 if item in relevant else 0.0 for item in selected]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        result[f"ndcg_at_{k}"] = dcg / ideal if ideal else 0.0
    result["mrr"] = next((1 / (i + 1) for i, item in enumerate(ranked) if item in relevant), 0.0)
    return result


def aggregate(cases: list[dict]) -> dict:
    if not cases:
        return {}
    keys = sorted(set.intersection(*(set(row) for row in cases)))
    return {key: statistics.fmean(float(row[key]) for row in cases) for key in keys}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)
