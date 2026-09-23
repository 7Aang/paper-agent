"""Reproducible paper- or chunk-level retrieval benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_agent.config import canonical_hash, load_yaml
from paper_agent.metrics import aggregate, percentile, retrieval_case
from paper_agent.retrieval import build_retriever, paper_results
from paper_agent.schemas import BenchmarkQuestion, ExperimentMetadata, RetrievalConfig
from paper_agent.store import Store


def load_questions(path: Path, split: str | None):
    rows = [BenchmarkQuestion.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if split is None or row.split == split]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--benchmark", default="benchmark/questions.jsonl")
    parser.add_argument("--split", choices=["dev", "test"])
    parser.add_argument("--output-root", default="results/retrieval")
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config, RetrievalConfig)
    questions = load_questions(ROOT / args.benchmark, args.split)
    store = Store(ROOT / args.data)
    if not store.papers() or not questions:
        raise RuntimeError("REQUIRES DATA: ingest corpus and provide benchmark questions")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    started = time.perf_counter()
    retriever = build_retriever(store.chunks(), config, ROOT / ".model_cache")
    build_ms = (time.perf_counter() - started) * 1000
    details, latencies = [], []
    max_k = max(config.top_k)
    for question in questions:
        start = time.perf_counter()
        if config.level == "paper":
            hits = paper_results(retriever, question.question, max_k)
            ranked = [row["paper_id"] for row in hits]
            relevant = set(question.ground_truth_papers)
        else:
            hits = retriever.search(question.question, max_k)
            ranked = [row["id"] for row in hits]
            relevant = set(question.ground_truth_chunks)
            if not relevant:
                continue
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        values = retrieval_case(ranked, relevant, config.top_k)
        details.append({"question_id": question.id, "ranked": ranked, "relevant": sorted(relevant), "latency_ms": latency, **values})
    status = "COMPLETED" if details else "REQUIRES DATA"
    metadata = ExperimentMetadata(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit(),
        config=config.model_dump(),
        corpus_hash=store.signature(),
        benchmark_hash=canonical_hash([q.model_dump() for q in questions]),
        status=status,
        limitation="Labels are assistant-authored and not independently human verified.",
    )
    summary = {
        "metadata": metadata.model_dump(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "papers": len(store.papers()),
        "chunks": len(store.chunks()),
        "questions": len(details),
        "verified_questions": sum(q.verified for q in questions),
        "index_build_ms": build_ms,
        "latency_ms": {"mean": sum(latencies) / len(latencies), "p50": percentile(latencies, .5), "p95": percentile(latencies, .95)},
        "metrics": aggregate([{k: v for k, v in row.items() if k.startswith(("hit_", "recall_", "ndcg_")) or k == "mrr"} for row in details]),
    }
    out = ROOT / args.output_root / config.name / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "cases.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=details[0].keys())
        writer.writeheader(); writer.writerows(details)
    lines = [f"# Retrieval benchmark: {config.name}", "", metadata.limitation or "", "", f"- Corpus: {summary['papers']} papers / {summary['chunks']} chunks", f"- Questions: {summary['questions']} ({summary['verified_questions']} independently verified)", f"- Index build: {build_ms:.2f} ms", f"- Query latency: mean {summary['latency_ms']['mean']:.2f} ms, P95 {summary['latency_ms']['p95']:.2f} ms", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value:.4f} |" for key, value in summary["metrics"].items())
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
