"""Deterministic citation and performance evaluation helpers."""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import percentile
from .store import Store, normalize


def citation_metrics(report: dict, store: Store) -> dict:
    claims = report.get("claims", [])
    total = len(claims)
    valid, complete = 0, 0
    cases = []
    for index, claim in enumerate(claims):
        has_fields = all(claim.get(field) not in (None, "") for field in ("paper_id", "page", "quote", "evidence_id"))
        page = store.page(str(claim.get("paper_id", "")), int(claim.get("page", 0))) if has_fields else None
        on_page = bool(page and normalize(str(claim.get("quote", ""))) in normalize(page))
        complete += int(has_fields)
        valid += int(has_fields and on_page)
        cases.append({"index": index, "citation_complete": has_fields, "quote_on_page": on_page})
    return {
        "claims": total,
        "citation_precision_structural": valid / total if total else None,
        "citation_coverage": complete / total if total else None,
        "citation_correctness_structural": valid / total if total else None,
        "evidence_grounded_rate_structural": valid / total if total else None,
        "unsupported_claim_rate_structural": (total - valid) / total if total else None,
        "semantic_entailment": "NOT RUN",
        "cases": cases,
        "definition": "Structural metrics require paper id, page, evidence id and an exact quote present on that page; they do not prove semantic entailment.",
    }


def performance_metrics(trace_path: str | Path) -> dict:
    rows = [json.loads(line) for line in Path(trace_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    elapsed = [float(row.get("elapsed_s", 0)) * 1000 for row in rows]
    return {
        "events": len(rows),
        "total_latency_ms": max(elapsed, default=0.0),
        "event_elapsed_p50_ms": percentile(elapsed, .5),
        "event_elapsed_p95_ms": percentile(elapsed, .95),
        "retries": sum(int(row.get("retry_count", 0)) for row in rows),
        "timeouts": sum(int(row.get("timeout_count", 0)) for row in rows),
        "failures": sum(bool(row.get("failure_reason")) or row.get("event") == "failed" for row in rows),
    }
