"""Evaluate deterministic citation properties for saved reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_agent.evaluation import citation_metrics
from paper_agent.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="results/citation/summary.json")
    args = parser.parse_args()
    store = Store(ROOT / args.data)
    reports = sorted((ROOT / args.runs).glob("*/report.json"))
    rows = []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"run_id": report["run_id"], **citation_metrics(report, store)})
    claims = sum(row["claims"] for row in rows)
    valid = sum(sum(case["quote_on_page"] and case["citation_complete"] for case in row["cases"]) for row in rows)
    output = {
        "status": "COMPLETED" if rows else "REQUIRES DATA",
        "runs": len(rows),
        "claims": claims,
        "evidence_grounded_rate_structural": valid / claims if claims else None,
        "semantic_entailment": "NOT RUN",
        "scope": "Deterministic exact-quote provenance only; no LLM judge and no human semantic labels.",
        "cases": rows,
    }
    target = ROOT / args.output; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
