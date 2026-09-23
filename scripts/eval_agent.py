"""Agent benchmark with honest separation of structural and semantic metrics."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_agent.agents import Pipeline, Provider
from paper_agent.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["extractive", "llm"], default="extractive")
    parser.add_argument("--strategy", choices=["single", "multi"], default="multi")
    parser.add_argument("--data", default="data")
    parser.add_argument("--runs", default="runs/agent-benchmark")
    parser.add_argument("--output", default="results/agent/summary.json")
    parser.add_argument("--no-evidence-gate", action="store_true")
    args = parser.parse_args()
    if args.mode == "llm" and not all(__import__("os").getenv(key) for key in ("PAPER_AGENT_API_KEY", "PAPER_AGENT_BASE_URL", "PAPER_AGENT_MODEL")):
        output = {"status": "REQUIRES API KEY", "semantic_metrics": "NOT RUN"}
    else:
        store = Store(ROOT / args.data)
        cases = [json.loads(line) for line in (ROOT / "benchmark" / "agent_cases.jsonl").read_text(encoding="utf-8").splitlines()]
        provider = Provider() if args.mode == "llm" else None
        rows = []
        for case in cases:
            start = time.perf_counter()
            report = Pipeline(store, ROOT / args.runs, provider, checkpoints=False).run(case["question"], args.mode, 3, strategy=args.strategy, evidence_gate=not args.no_evidence_gate)
            present = {claim["paper_id"] for claim in report["claims"]}
            target = set(case["target_papers"])
            rows.append({
                "id": case["id"], "answerable": case["answerable"], "status": report["status"],
                "latency_s": time.perf_counter() - start, "claims": len(report["claims"]),
                "llm_calls": report["usage"].get("calls", 0),
                "target_paper_coverage": len(present & target) / len(target) if target else None,
                "abstained": not report["claims"],
                "quotes_on_page": all(claim["quote"] in (store.page(claim["paper_id"], claim["page"]) or "") for claim in report["claims"]),
            })
        answerable = [row for row in rows if row["answerable"]]
        unanswerable = [row for row in rows if not row["answerable"]]
        output = {
            "status": "COMPLETED", "mode": args.mode, "strategy": args.strategy, "evidence_gate": not args.no_evidence_gate, "cases": len(rows),
            "target_paper_coverage": statistics.fmean(row["target_paper_coverage"] for row in answerable),
            "refusal_accuracy": statistics.fmean(row["abstained"] for row in unanswerable),
            "evidence_grounded_rate_structural": statistics.fmean(row["quotes_on_page"] for row in rows),
            "latency_s": {"mean": statistics.fmean(row["latency_s"] for row in rows), "median": statistics.median(row["latency_s"] for row in rows)},
            "average_llm_calls": statistics.fmean(row["llm_calls"] for row in rows),
            "answer_correctness": "NOT RUN",
            "task_success_rate": "NOT RUN",
            "scope": "Target-paper coverage and exact-quote provenance are deterministic diagnostics; semantic correctness requires human review.",
            "details": rows,
        }
    target = ROOT / args.output; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
