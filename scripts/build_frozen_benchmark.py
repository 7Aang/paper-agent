"""Create the 60-question frozen diagnostic set for the 50-paper corpus."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_agent.config import canonical_hash
from paper_agent.schemas import BenchmarkQuestion


UNANSWERABLE = (
    "What exact p99 latency was reported for a production deployment at Hangzhou Airport in January 2027?",
    "How many NVIDIA B300 GPUs were used in the authors' February 2027 commercial rollout?",
    "What exact annual enterprise license price in Chinese yuan is reported for the system?",
    "What percentage of battery power did the method save in a real PX4 flight test in March 2027?",
    "Which paper reports a 2027 randomized clinical trial with exactly 10,000 enrolled patients?",
    "What exact carbon-emission reduction was measured in the authors' 2027 data-center deployment?",
    "How many paying users did the system have on 1 April 2027?",
    "What exact service-level agreement did the authors sign with a Fortune 50 customer in 2027?",
    "Which paper reports audited 2027 revenue attributable to its proposed agent architecture?",
    "What was the exact incident rate after twelve months of an autonomous production rollout ending in 2027?",
)


def abstract_snippet(record: dict) -> str:
    abstract = " ".join(record.get("abstract", "").split())
    if not abstract:
        return f"the method and experiments described by the work titled {record['title']}"
    sentence = re.split(r"(?<=[.!?])\s+", abstract)[0]
    words = sentence.split()
    if len(words) < 12 and len(re.split(r"(?<=[.!?])\s+", abstract)) > 1:
        sentence += " " + re.split(r"(?<=[.!?])\s+", abstract)[1]
        words = sentence.split()
    snippet = " ".join(words[:42]).strip()
    return snippet.rstrip(".?!")


def main() -> None:
    benchmark_dir = ROOT / "benchmark"
    question_path = benchmark_dir / "questions.jsonl"
    manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != 50:
        raise SystemExit(f"expected 50 corpus records, found {len(manifest)}")

    prior = [
        BenchmarkQuestion.model_validate_json(line)
        for line in question_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seed = [row for row in prior if re.fullmatch(r"h\d{2}", row.id)][:24]
    if len(seed) != 24:
        raise SystemExit("expected the 24 reviewed seed diagnostics h01-h24")

    generated: list[BenchmarkQuestion] = []
    for number, record in enumerate(manifest[12:38], start=25):
        snippet = abstract_snippet(record)
        generated.append(
            BenchmarkQuestion(
                id=f"a{number:02d}",
                question=f"Which paper should I read for this contribution: {snippet}?",
                type="single_paper_abstract_diagnostic",
                difficulty="easy" if number % 3 else "medium",
                answerable=True,
                ground_truth_papers=[record["id"]],
                reference_answer=record["title"],
                supporting_evidence=[snippet],
                generation_source="arxiv_abstract_template_v3",
                verified=False,
                split="dev" if number % 5 == 0 else "test",
            )
        )

    negative = [
        BenchmarkQuestion(
            id=f"n{number:02d}",
            question=question,
            type="unanswerable_out_of_corpus",
            difficulty="hard",
            answerable=False,
            generation_source="constructed_future_detail_v3",
            verified=False,
            split="test",
        )
        for number, question in enumerate(UNANSWERABLE, start=51)
    ]
    questions = seed + generated + negative
    if len(questions) != 60:
        raise AssertionError(len(questions))
    records = [row.model_dump() for row in questions]
    question_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    frozen = {
        "version": "v3.0",
        "status": "frozen",
        "count": len(records),
        "answerable_count": sum(row["answerable"] for row in records),
        "unanswerable_count": sum(not row["answerable"] for row in records),
        "verified_count": sum(row["verified"] for row in records),
        "scope": (
            "Assistant-authored diagnostics on a fixed 50-paper corpus: 24 retained seed questions, "
            "26 arXiv-abstract-derived questions and 10 constructed unanswerable questions; "
            "independent human verification pending."
        ),
        "corpus_count": len(manifest),
        "corpus_sha256": canonical_hash(manifest),
        "sha256": canonical_hash(records),
    }
    (benchmark_dir / "manifest.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    answerable_cases = [row for row in questions if row.answerable][:8]
    unanswerable_cases = negative[:8]
    cases = [
        {
            "id": f"agent-{row.id}",
            "question": row.question,
            "answerable": row.answerable,
            "target_papers": row.ground_truth_papers,
            "type": row.type,
        }
        for row in answerable_cases + unanswerable_cases
    ]
    (benchmark_dir / "agent_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases), encoding="utf-8"
    )
    print(json.dumps({"status": "frozen", **frozen, "agent_cases": len(cases)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
