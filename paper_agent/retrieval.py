"""Pluggable lexical, dense, hybrid and reranked retrieval.

No benchmark IDs, relevance labels, or hand-written query answers enter this module.
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol

STOP = set("a an the of in on at to for and or is are was were be by with from as that this it its how what which does do can could would should explain describe compare paper papers study studies approach approaches method methods using use used".split())

EXPANSIONS = {
    "retrieve": ("retrieval", "search"),
    "retrieval": ("retrieve", "search"),
    "agent": ("tool", "reasoning"),
    "citation": ("evidence", "source"),
    "embedding": ("vector", "dense"),
    "vector": ("embedding", "dense"),
}


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+",text.lower()) if len(w)>1 and w not in STOP]


class Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tf = [Counter(tokens(c["text"])) for c in chunks]
        self.lengths = [sum(t.values()) for t in self.tf]
        self.avgdl = sum(self.lengths)/max(1,len(chunks))
        self.postings = defaultdict(list)
        for i, counter in enumerate(self.tf):
            for word, count in counter.items():
                self.postings[word].append((i,count))
        n = len(chunks)
        self.idf = {word:math.log(1+(n-len(post)+0.5)/(len(post)+0.5)) for word,post in self.postings.items()}
        self.titles = [set(tokens(c["title"])) for c in chunks]

    def search(self, query: str, k: int = 5, mode: str = "optimized", paper_id: str | None = None) -> list[dict]:
        if mode not in {"baseline","bm25","optimized"}:
            raise ValueError("Unknown retrieval mode")
        terms = set(tokens(query))
        if not terms:
            return []
        scores = defaultdict(float)
        for word in terms:
            for i, freq in self.postings.get(word,[]):
                if paper_id and self.chunks[i]["paper_id"] != paper_id:
                    continue
                if mode == "baseline":
                    scores[i] += freq
                else:
                    dl = self.lengths[i]
                    scores[i] += self.idf[word] * freq * 2.5 / (freq+1.5*(0.25+0.75*dl/self.avgdl))
        if mode == "optimized":
            # Title prior is generic and deliberately modest; evidence remains page text.
            for i in scores:
                scores[i] += 0.6*sum(self.idf.get(t,0) for t in terms & self.titles[i])
        ranked = sorted(scores,key=lambda i:(-scores[i],self.chunks[i]["id"]))[:k]
        return [{**self.chunks[i],"score":round(scores[i],6)} for i in ranked]

    def search_papers(self, query: str, k: int = 5, mode: str = "optimized") -> list[dict]:
        results = self.search(query,len(self.chunks),mode)
        seen = set()
        papers = []
        for row in results:
            if row["paper_id"] not in seen:
                seen.add(row["paper_id"])
                papers.append(row)
                if len(papers) >= k:
                    break
        return papers


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[dict]: ...


def expand_query(query: str) -> str:
    """Small transparent expansion dictionary; every added token is inspectable."""
    words = tokens(query)
    expanded = list(words)
    for word in words:
        expanded.extend(EXPANSIONS.get(word, ()))
    return " ".join(dict.fromkeys(expanded))


class LexicalRetriever:
    def __init__(self, chunks: list[dict], *, baseline: bool = False,
                 field_weighting: bool = False, query_expansion: bool = False):
        self.index = Index(chunks)
        self.mode = "baseline" if baseline else ("optimized" if field_weighting else "bm25")
        self.query_expansion = query_expansion
        self.name = self.mode + ("_qe" if query_expansion else "")

    def search(self, query: str, k: int = 10) -> list[dict]:
        return self.index.search(expand_query(query) if self.query_expansion else query, k, self.mode)


class DenseRetriever:
    """Sentence-transformers adapter. Model downloads are explicit, never silent."""
    def __init__(self, chunks: list[dict], model_name: str, cache_dir: str | Path | None = None):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("dense retrieval requires the 'retrieval' optional dependencies") from exc
        self.np = np
        self.chunks = chunks
        self.model_name = model_name
        self.name = "dense"
        self.model = SentenceTransformer(model_name, cache_folder=str(cache_dir) if cache_dir else None)
        texts = [f"{row['title']}\n{row['text']}" for row in chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, k: int = 10) -> list[dict]:
        vector = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ vector
        order = self.np.argsort(-scores)[:k]
        return [{**self.chunks[int(i)], "score": float(scores[int(i)])} for i in order]


class LSADenseRetriever:
    """Offline dense baseline using TF-IDF followed by truncated latent semantic analysis."""
    def __init__(self, chunks: list[dict], dimensions: int = 128):
        try:
            import numpy as np
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import svds
        except ImportError as exc:
            raise RuntimeError("LSA dense retrieval requires numpy and scipy") from exc
        self.np, self.chunks, self.name = np, chunks, "dense_lsa"
        document_tokens = [tokens(f"{row['title']} {row['text']}") for row in chunks]
        vocabulary = sorted({word for words in document_tokens for word in words})
        self.vocabulary = {word: index for index, word in enumerate(vocabulary)}
        document_frequency = Counter(word for words in document_tokens for word in set(words))
        self.idf = {word: math.log((len(chunks) + 1) / (document_frequency[word] + 1)) + 1 for word in vocabulary}
        rows, columns, values = [], [], []
        for row_index, words in enumerate(document_tokens):
            counts = Counter(words)
            for word, count in counts.items():
                rows.append(row_index); columns.append(self.vocabulary[word]); values.append((1 + math.log(count)) * self.idf[word])
        matrix = csr_matrix((values, (rows, columns)), shape=(len(chunks), len(vocabulary)), dtype=float)
        rank = max(2, min(dimensions, min(matrix.shape) - 1))
        _, singular, vt = svds(matrix, k=rank, random_state=0)
        order = np.argsort(-singular)
        self.components = vt[order].T
        dense = matrix @ self.components
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        self.embeddings = dense / np.maximum(norms, 1e-12)

    def search(self, query: str, k: int = 10) -> list[dict]:
        counts = Counter(tokens(query))
        vector = self.np.zeros(len(self.vocabulary), dtype=float)
        for word, count in counts.items():
            if word in self.vocabulary:
                vector[self.vocabulary[word]] = (1 + math.log(count)) * self.idf[word]
        latent = vector @ self.components
        norm = self.np.linalg.norm(latent)
        if norm <= 1e-12:
            return []
        scores = self.embeddings @ (latent / norm)
        order = self.np.argsort(-scores)[:k]
        return [{**self.chunks[int(i)], "score": float(scores[int(i)])} for i in order]


def reciprocal_rank_fusion(result_sets: list[list[dict]], k: int, rrf_k: int = 60) -> list[dict]:
    scores = defaultdict(float)
    rows = {}
    for result in result_sets:
        for rank, row in enumerate(result, 1):
            rows[row["id"]] = row
            scores[row["id"]] += 1.0 / (rrf_k + rank)
    ranked = sorted(scores, key=lambda item: (-scores[item], item))[:k]
    return [{**rows[item], "score": scores[item]} for item in ranked]


class HybridRetriever:
    def __init__(self, lexical: Retriever, dense: Retriever, *, candidate_k: int = 30, rrf_k: int = 60):
        self.lexical, self.dense = lexical, dense
        self.candidate_k, self.rrf_k = candidate_k, rrf_k
        self.name = "hybrid_rrf"

    def search(self, query: str, k: int = 10) -> list[dict]:
        return reciprocal_rank_fusion(
            [self.lexical.search(query, self.candidate_k), self.dense.search(query, self.candidate_k)],
            k,
            self.rrf_k,
        )


class CrossEncoderReranker:
    def __init__(self, base: Retriever, model_name: str, candidate_k: int = 30, cache_dir: str | Path | None = None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("reranking requires the 'retrieval' optional dependencies") from exc
        self.base = base
        self.model = CrossEncoder(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        self.candidate_k = candidate_k
        self.name = "hybrid_cross_encoder"

    def search(self, query: str, k: int = 10) -> list[dict]:
        candidates = self.base.search(query, self.candidate_k)
        if not candidates:
            return []
        scores = self.model.predict([(query, row["text"]) for row in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: (-float(pair[1]), pair[0]["id"]))[:k]
        return [{**row, "score": float(score)} for row, score in ranked]


def build_retriever(chunks: list[dict], config, cache_dir: str | Path | None = None) -> Retriever:
    lexical = LexicalRetriever(
        chunks,
        baseline=config.method == "baseline",
        field_weighting=config.field_weighting,
        query_expansion=config.query_expansion,
    )
    if config.method in {"baseline", "bm25"}:
        return lexical
    if config.dense_provider == "lsa":
        dense = LSADenseRetriever(chunks)
    elif not config.embedding_model:
        raise ValueError("embedding_model is required for dense retrieval")
    else:
        dense = DenseRetriever(chunks, config.embedding_model, cache_dir)
    if config.method == "dense":
        return dense
    hybrid = HybridRetriever(lexical, dense, candidate_k=config.candidate_k, rrf_k=config.rrf_k)
    if config.method == "hybrid":
        return hybrid
    if not config.reranker_model:
        raise ValueError("reranker_model is required")
    return CrossEncoderReranker(hybrid, config.reranker_model, config.candidate_k, cache_dir)


def paper_results(retriever: Retriever, query: str, k: int = 10) -> list[dict]:
    candidates = retriever.search(query, max(k, k * 20))
    seen, result = set(), []
    for row in candidates:
        if row["paper_id"] in seen:
            continue
        seen.add(row["paper_id"])
        result.append(row)
        if len(result) == k:
            break
    return result
