"""FastAPI adapter for the same Store and Pipeline used by the CLI."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agents import Pipeline, provider_from_env
from .retrieval import Index
from .store import Store


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    mode: str = "optimized"


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: str = "extractive"
    strategy: str = "multi"
    max_papers: int = Field(default=4, ge=1, le=8)


def create_app(data_dir: str | Path | None = None, run_dir: str | Path | None = None) -> FastAPI:
    store = Store(data_dir or os.getenv("PAPER_AGENT_HOME", "data"))
    runs = Path(run_dir or os.getenv("PAPER_AGENT_RUNS", "runs"))
    app = FastAPI(title="Paper Agent API", version="0.2.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "papers": len(store.papers()), "corpus_signature": store.signature()}

    @app.get("/papers")
    def papers():
        return store.papers()

    @app.post("/search")
    def search(request: SearchRequest):
        if request.mode not in {"baseline", "bm25", "optimized"}:
            raise HTTPException(400, "mode must be baseline, bm25 or optimized")
        return Index(store.chunks()).search_papers(request.query, request.k, request.mode)

    @app.post("/research")
    def research(request: ResearchRequest):
        if request.mode not in {"extractive", "llm"} or request.strategy not in {"single", "multi"}:
            raise HTTPException(400, "invalid mode or strategy")
        try:
            provider = provider_from_env() if request.mode == "llm" else None
            return Pipeline(store, runs, provider).run(
                request.query, request.mode, request.max_papers, strategy=request.strategy
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


app = create_app()
