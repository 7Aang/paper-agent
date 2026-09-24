"""Run the reproducible benchmark suite and record every command status."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFLINE_CONFIGS = (
    "bm25.yaml", "bm25_field.yaml", "bm25_field_qe.yaml", "dense_lsa.yaml", "hybrid_lsa.yaml",
)
MODEL_CONFIGS = ("dense.yaml", "hybrid.yaml", "hybrid_reranker.yaml")


def run(label: str, arguments: list[str], required: bool = True) -> dict:
    started = time.perf_counter()
    print(f"\n== {label} ==", flush=True)
    result = subprocess.run([sys.executable, *arguments], cwd=ROOT)
    row = {
        "label": label,
        "command": ["python", *arguments],
        "returncode": result.returncode,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "required": required,
        "status": "COMPLETED" if result.returncode == 0 else "FAILED",
    }
    if required and result.returncode:
        raise SystemExit(f"required benchmark step failed: {label}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["offline", "full"], default="full")
    parser.add_argument("--include-llm", action="store_true")
    args = parser.parse_args()
    rows: list[dict] = [run("validate frozen benchmark", ["scripts/validate_benchmark.py"])]

    configs = OFFLINE_CONFIGS + (MODEL_CONFIGS if args.profile == "full" else ())
    for name in configs:
        rows.append(run(
            f"retrieval {name.removesuffix('.yaml')}",
            ["scripts/eval_retrieval.py", "--config", f"configs/retrieval/{name}"],
            required=name in OFFLINE_CONFIGS,
        ))

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = f"runs/agent-benchmark-v3/{batch_id}"
    rows.append(run("extractive agent gate off", [
        "scripts/eval_agent.py", "--mode", "extractive", "--no-evidence-gate",
        "--runs", run_root, "--output", "results/agent/extractive_gate_off.json",
    ]))
    rows.append(run("extractive agent gate on", [
        "scripts/eval_agent.py", "--mode", "extractive", "--runs", run_root,
        "--output", "results/agent/extractive_gate_on.json",
    ]))
    if args.include_llm:
        key_ready = all(os.getenv(key) for key in ("PAPER_AGENT_API_KEY", "PAPER_AGENT_BASE_URL", "PAPER_AGENT_MODEL"))
        if key_ready:
            for strategy in ("single", "multi"):
                rows.append(run(f"llm agent {strategy}", [
                    "scripts/eval_agent.py", "--mode", "llm", "--strategy", strategy,
                    "--runs", "runs/llm-benchmark-v3",
                    "--output", f"results/agent/llm_{strategy}.json",
                ]))
        else:
            rows.append({
                "label": "llm agent ablation", "command": [], "returncode": None,
                "elapsed_s": 0, "required": False, "status": "REQUIRES API KEY",
            })

    rows.append(run("citation provenance", ["scripts/eval_citations.py", "--runs", run_root]))
    rows.append(run("fault injection", ["scripts/eval_failures.py"]))
    rows.append(run("performance export", ["scripts/export_performance.py", "--runs", run_root]))
    rows.append(run("benchmark report", ["scripts/build_report.py"]))
    summary = {
        "status": "COMPLETED" if all(row["status"] == "COMPLETED" for row in rows if row["required"]) else "FAILED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "steps": rows,
    }
    target = ROOT / "results" / "benchmark_run.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
