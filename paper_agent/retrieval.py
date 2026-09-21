"""English lexical retrieval: TF baseline, BM25, title-aware BM25.

No benchmark IDs, relevance labels, or hand-written query answers enter this module.
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict

STOP = set("a an the of in on at to for and or is are was were be by with from as that this it its how what which does do can could would should explain describe compare paper papers study studies approach approaches method methods using use used".split())


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
