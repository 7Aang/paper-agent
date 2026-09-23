"""Export standardized run performance without inventing unavailable cost data."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/agent-benchmark")
    parser.add_argument("--output", default="results/performance.csv")
    args = parser.parse_args()
    rows = []
    for report_path in sorted((ROOT / args.runs).glob("*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trace_path = report_path.with_name("trace.jsonl")
        events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        stage_ms = {event["stage"]: event.get("latency_ms") for event in events if event.get("latency_ms") is not None}
        usage = report.get("usage", {})
        rows.append({
            "run_id": report["run_id"], "mode": report["mode"], "strategy": report["strategy"],
            "status": report["status"], "total_latency_ms": report["elapsed_s"] * 1000,
            "retrieval_latency_ms": stage_ms.get("retriever"), "planning_latency_ms": stage_ms.get("planner"),
            "research_latency_ms": stage_ms.get("researcher"), "verification_latency_ms": stage_ms.get("reviewer"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)) if usage else None,
            "llm_calls": usage.get("calls", 0), "estimated_cost": "unavailable",
            "retry_count": sum(event.get("retry_count", 0) for event in events),
            "timeout_count": sum(event.get("timeout_count", 0) for event in events),
            "failure_reason": ";".join(error.get("error", "") for error in report.get("errors", [])),
        })
    target = ROOT / args.output; target.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": "COMPLETED" if rows else "REQUIRES DATA", "runs": len(rows),
        "average_latency_ms": statistics.fmean(row["total_latency_ms"] for row in rows) if rows else None,
        "failure_rate": statistics.fmean(bool(row["failure_reason"]) for row in rows) if rows else None,
        "cost_per_query": "unavailable",
    }
    (target.parent / "performance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
