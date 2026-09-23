"""Deterministic fault-injection benchmark; no external API is used."""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_agent.agents import Pipeline
from paper_agent.checkpoints import CheckpointProvider, StepCache
from paper_agent.network import request_bytes
from paper_agent.store import Store


def make_store(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.pdf"
    doc = fitz.open(); page = doc.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 550, 800), "Dense retrieval uses a dual encoder to match questions and passages. Exact evidence is linked to a source page so every generated statement can be checked against the original document.")
    doc.save(source); doc.close()
    store = Store(root / "data")
    store.ingest(source, "Dense retrieval", "p1", 2024, "https://example.org/p1")
    return store, source


class FixtureProvider:
    base = "https://fixture.invalid"
    model = "fixture"

    def __init__(self, fail_role=None, malformed_role=None):
        self.fail_role, self.malformed_role = fail_role, malformed_role
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def call(self, role, _instruction, payload):
        self.usage["calls"] += 1
        if role == self.fail_role:
            raise TimeoutError("injected secret outage")
        if role == self.malformed_role:
            return {}
        if role == "Planner": return {"queries": [payload["question"]], "dimensions": ["method"]}
        if role == "Researcher":
            evidence = payload["evidence"][0]
            return {"claims": [{"claim": evidence["text"], "quote": evidence["text"], "evidence_id": evidence["id"], "kind": "method"}]}
        if role == "Synthesizer": return {"order": list(range(len(payload["claims"])))}
        return {"verdicts": [{"index": i, "supported": True, "relevant": True, "unsupported_details": []} for i in range(len(payload["claims"]))]}


def main():
    results = []

    def check(name, function):
        try:
            function()
            results.append({"case": name, "recovered": True})
        except Exception as exc:
            results.append({"case": name, "recovered": False, "error": type(exc).__name__})

    with tempfile.TemporaryDirectory(prefix="paper-agent-faults-") as temp:
        root = Path(temp)

        def rollback():
            store, _ = make_store(root / "rollback")
            try:
                with store.connect() as conn:
                    conn.execute("DELETE FROM papers")
                    raise RuntimeError("injected")
            except RuntimeError:
                pass
            assert len(store.papers()) == 1

        def corrupt_pdf():
            store, _ = make_store(root / "corrupt")
            bad = root / "corrupt" / "bad.pdf"; bad.write_bytes(b"%PDF corrupt")
            try: store.ingest(bad, paper_id="p1")
            except Exception: pass
            assert store.page("p1", 1)

        def concurrent_ingest():
            store, source = make_store(root / "concurrent")
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(lambda _: store.ingest(source, paper_id="p1"), range(8)))
            assert all(row["cached"] for row in rows) and len(store.papers()) == 1

        def empty_abstain():
            report = Pipeline(Store(root / "empty"), root / "empty-runs").run("retrieval")
            assert report["status"] == "insufficient_evidence"

        def malformed_detected():
            store, _ = make_store(root / "malformed")
            try: Pipeline(store, root / "malformed-runs", FixtureProvider(malformed_role="Planner")).run("retrieval", "llm")
            except ValueError: return
            raise AssertionError("malformed output was accepted")

        def worker_failure_contained():
            store, _ = make_store(root / "worker")
            report = Pipeline(store, root / "worker-runs", FixtureProvider(fail_role="Researcher")).run("retrieval", "llm")
            assert report["status"] == "insufficient_evidence" and report["errors"]
            assert "secret" not in json.dumps(report)

        def checkpoint_restart():
            store, _ = make_store(root / "checkpoint")
            failing = FixtureProvider(fail_role="Reviewer")
            try: Pipeline(store, root / "checkpoint-runs", failing).run("retrieval", "llm")
            except TimeoutError: pass
            recovered = FixtureProvider()
            report = Pipeline(store, root / "checkpoint-runs", recovered).run("retrieval", "llm")
            assert report["status"] == "completed" and report["checkpoints"]["hits"] == 3

        def corrupt_checkpoint():
            path = root / "cache" / "steps.db"
            provider = FixtureProvider(); cached = CheckpointProvider(provider, StepCache(path))
            cached.call("Planner", "plan", {"question": "q"})
            with cached.cache.connect() as conn: conn.execute("UPDATE steps SET response='{}'")
            cached.call("Planner", "plan", {"question": "q"})
            assert provider.usage["calls"] == 2

        def retry_status(code):
            calls = {"count": 0}
            class Response:
                def __enter__(self): return self
                def __exit__(self, *_): return False
                def read(self, _): return b"ok"
            def open_(*_args, **_kwargs):
                calls["count"] += 1
                if calls["count"] < 3: raise urllib.error.HTTPError("https://example.org", code, "retry", {}, None)
                return Response()
            with patch("paper_agent.network.urllib.request.urlopen", open_), patch("paper_agent.network.time.sleep", lambda _: None):
                assert request_bytes("https://example.org") == b"ok" and calls["count"] == 3

        for name, fn in [
            ("database_rollback", rollback), ("corrupt_pdf_preserves_record", corrupt_pdf),
            ("concurrent_ingest_idempotent", concurrent_ingest), ("empty_corpus_abstains", empty_abstain),
            ("malformed_llm_output_detected", malformed_detected), ("worker_timeout_contained", worker_failure_contained),
            ("checkpoint_restart_recovers", checkpoint_restart), ("corrupt_checkpoint_recomputed", corrupt_checkpoint),
            ("http_429_retry_recovers", lambda: retry_status(429)), ("http_500_retry_recovers", lambda: retry_status(500)),
        ]:
            check(name, fn)

    recovered = sum(row["recovered"] for row in results)
    output = {"status": "COMPLETED", "cases": len(results), "recovered": recovered, "recovery_success_rate": recovered / len(results), "details": results}
    target = ROOT / "results" / "reliability" / "summary.json"; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if recovered != len(results): raise SystemExit(1)


if __name__ == "__main__":
    main()
