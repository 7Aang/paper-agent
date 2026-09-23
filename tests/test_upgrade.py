import json
import urllib.error

import fitz
import pytest
from fastapi.testclient import TestClient

from paper_agent.agents import FallbackProvider, Pipeline, evidence_sufficient_for_extractive
from paper_agent.api import create_app
from paper_agent.config import canonical_hash, load_yaml
from paper_agent.evaluation import citation_metrics
from paper_agent.metrics import aggregate, retrieval_case
from paper_agent.network import request_bytes
from paper_agent.retrieval import LexicalRetriever, reciprocal_rank_fusion
from paper_agent.schemas import BenchmarkQuestion, RetrievalConfig
from paper_agent.store import Store


def make_store(tmp_path):
    path = tmp_path / "paper.pdf"
    doc = fitz.open(); page = doc.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 550, 800), "Dense retrieval maps questions and passages into vectors. Evidence quotes remain linked to pages.")
    doc.save(path); doc.close()
    store = Store(tmp_path / "data")
    store.ingest(path, "Dense Retrieval", "p1", 2024, "https://example.org/p1")
    return store


def test_metric_definitions_and_aggregation():
    case = retrieval_case(["a", "b", "c"], {"b"}, (1, 3))
    assert case["hit_at_1"] == 0 and case["recall_at_3"] == 1 and case["mrr"] == .5
    assert aggregate([case, case])["mrr"] == .5


def test_benchmark_schema_rejects_answerable_without_target():
    with pytest.raises(ValueError):
        BenchmarkQuestion(id="x", question="valid question", generation_source="test")


def test_config_is_strict(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("name: test\nmethod: bm25\nunknown: true\n")
    with pytest.raises(ValueError):
        load_yaml(path, RetrievalConfig)


def test_rrf_rewards_items_seen_by_both_retrievers():
    a = [{"id": "x"}, {"id": "y"}]
    b = [{"id": "z"}, {"id": "x"}]
    assert reciprocal_rank_fusion([a, b], 3)[0]["id"] == "x"


def test_query_expansion_is_explicit_and_optional():
    chunks = [{"id": "1", "paper_id": "p", "page": 1, "title": "Vector Search", "text": "dense vector search"}]
    plain = LexicalRetriever(chunks, query_expansion=False)
    expanded = LexicalRetriever(chunks, query_expansion=True)
    assert plain.search("embedding") == []
    assert expanded.search("embedding")[0]["id"] == "1"


def test_fastapi_adapter_exposes_bounded_routes(tmp_path):
    app = create_app(tmp_path / "data", tmp_path / "runs")
    paths = {route.path for route in app.routes}
    assert {"/health", "/papers", "/search", "/research"} <= paths
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/search", json={"query": "retrieval"}).json() == []
    report = client.post("/research", json={"query": "retrieval", "mode": "extractive"})
    assert report.status_code == 200 and report.json()["status"] == "insufficient_evidence"


def test_citation_metrics_are_structural_not_semantic(tmp_path):
    store = make_store(tmp_path)
    report = Pipeline(store, tmp_path / "runs").run("dense retrieval")
    metrics = citation_metrics(report, store)
    assert metrics["citation_precision_structural"] == 1
    assert metrics["semantic_entailment"] == "NOT RUN"


def test_trace_uses_versioned_schema(tmp_path):
    store = make_store(tmp_path)
    report = Pipeline(store, tmp_path / "runs").run("dense retrieval")
    trace = (tmp_path / "runs" / report["run_id"] / "trace.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in trace]
    assert all(row["schema_version"] == "paper-agent-trace-v2" for row in rows)
    assert all("stage" in row and "metadata" in row for row in rows)


@pytest.mark.parametrize("code", [429, 500])
def test_retryable_http_status_recovers(monkeypatch, code):
    calls = {"count": 0}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, _): return b"ok"

    def fake_open(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.HTTPError("https://example.org", code, "retry", {}, None)
        return Response()

    monkeypatch.setattr("paper_agent.network.urllib.request.urlopen", fake_open)
    monkeypatch.setattr("paper_agent.network.time.sleep", lambda _: None)
    assert request_bytes("https://example.org") == b"ok"
    assert calls["count"] == 3


def test_canonical_hash_is_order_independent_for_objects():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_exact_detail_gate_rejects_missing_named_constraints():
    evidence = [{"text": "ReAct alternates reasoning and acting."}]
    ok, missing = evidence_sufficient_for_extractive(
        "What was the exact p99 latency at Hangzhou Airport in August 2026?", evidence
    )
    assert not ok and {"p99", "hangzhou", "airport", "august", "2026"} <= set(missing)


def test_exact_detail_gate_accepts_constraints_present_in_evidence():
    evidence = [{"text": "The measured p99 latency was 20 ms at Hangzhou Airport in August 2026."}]
    assert evidence_sufficient_for_extractive(
        "What was the exact p99 latency at Hangzhou Airport in August 2026?", evidence
    )[0]


def test_model_fallback_is_bounded_and_metered():
    class Fake:
        base = "https://fixture.invalid"
        def __init__(self, model, fail):
            self.model, self.fail = model, fail
            self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usage_missing_calls": 0}
        def call(self, *_):
            self.usage["calls"] += 1
            if self.fail: raise TimeoutError("injected")
            return {"ok": True}
    provider = FallbackProvider([Fake("primary", True), Fake("fallback", False)])
    assert provider.call("Planner", "x", {}) == {"ok": True}
    assert provider.usage["calls"] == 2 and provider.usage["fallbacks"] == 1
