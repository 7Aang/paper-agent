"""Versioned schemas shared by benchmarks, traces and public APIs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkQuestion(StrictModel):
    id: str
    question: str = Field(min_length=3)
    type: str = "single_paper_factual"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    answerable: bool = True
    ground_truth_papers: list[str] = Field(default_factory=list)
    ground_truth_pages: list[int] = Field(default_factory=list)
    ground_truth_chunks: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    generation_source: str
    verified: bool = False
    split: Literal["train", "dev", "test"] = "test"

    @model_validator(mode="after")
    def answerable_has_target(self):
        if self.answerable and not (self.ground_truth_papers or self.ground_truth_chunks):
            raise ValueError("answerable questions require paper or chunk labels")
        return self


class TraceEvent(StrictModel):
    schema_version: str = "paper-agent-trace-v2"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    elapsed_s: float = Field(ge=0)
    stage: str
    event: str
    latency_ms: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0)
    timeout_count: int = Field(default=0, ge=0)
    failure_reason: str | None = None
    metadata: dict = Field(default_factory=dict)


class RetrievalConfig(StrictModel):
    name: str
    method: Literal["baseline", "bm25", "dense", "hybrid", "hybrid_reranker"]
    top_k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    level: Literal["paper", "chunk"] = "paper"
    field_weighting: bool = False
    query_expansion: bool = False
    dense_provider: str = "sentence_transformers"
    embedding_model: str | None = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = Field(default=60, ge=1)
    candidate_k: int = Field(default=30, ge=1)


class ExperimentMetadata(StrictModel):
    schema_version: str = "paper-agent-experiment-v1"
    run_id: str
    timestamp: str
    git_commit: str | None
    config: dict
    corpus_hash: str
    benchmark_hash: str
    status: Literal["COMPLETED", "NOT RUN", "REQUIRES API KEY", "REQUIRES DATA"]
    limitation: str | None = None
