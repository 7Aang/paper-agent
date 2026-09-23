"""Fail CI when a frozen benchmark changes without updating its manifest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_agent.config import canonical_hash
from paper_agent.schemas import BenchmarkQuestion


def main():
    path = ROOT / "benchmark" / "questions.jsonl"
    records = [BenchmarkQuestion.model_validate_json(line).model_dump() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    actual = canonical_hash(records)
    if manifest["status"] != "frozen" or manifest["count"] != len(records) or manifest["sha256"] != actual:
        raise SystemExit("frozen benchmark manifest mismatch")
    print(json.dumps({"status": "valid", "count": len(records), "sha256": actual}))


if __name__ == "__main__":
    main()
