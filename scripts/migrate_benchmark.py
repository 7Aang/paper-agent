"""One-time migration from the legacy frozen JSON to the versioned JSONL schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_agent.config import canonical_hash
from paper_agent.schemas import BenchmarkQuestion


def main():
    source = json.loads((ROOT / "evaluation" / "heldout_v2.json").read_text(encoding="utf-8"))
    destination = ROOT / "benchmark" / "questions.jsonl"
    destination.parent.mkdir(exist_ok=True)
    records = []
    for index, row in enumerate(source):
        record = BenchmarkQuestion(
            id=row["id"],
            question=row["query"],
            ground_truth_papers=row["relevant"],
            generation_source="assistant_authored_diagnostic_v2",
            verified=False,
            split="dev" if index % 4 == 0 else "test",
        )
        records.append(record.model_dump())
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    manifest = {
        "version": "v2.1",
        "status": "frozen",
        "count": len(records),
        "verified_count": sum(row["verified"] for row in records),
        "scope": "Assistant-authored diagnostic labels on a fixed 12-paper corpus; independent human verification pending.",
        "sha256": canonical_hash(records),
    }
    (destination.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
