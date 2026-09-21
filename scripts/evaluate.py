"""Frozen, seed-labeled small-corpus evaluation; not a public benchmark."""
import csv
import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_agent.agents import Pipeline, audit_claim, select_evidence
from paper_agent.retrieval import Index
from paper_agent.store import Store


def main():
    out=ROOT/'evaluation'/'results';out.mkdir(exist_ok=True)
    queries_path=ROOT/'evaluation'/'queries.json'
    queries=json.loads(queries_path.read_text())
    store=Store(ROOT/'data')
    chunks=store.chunks()
    if len(store.papers())!=12:
        raise RuntimeError('Expected complete 12-paper corpus; run fetch and corpus ingestion first')
    start=time.perf_counter();idx=Index(chunks);index_seconds=time.perf_counter()-start
    results={};details=[]
    for mode in ['baseline','bm25','optimized']:
        metrics=[];timings=[]
        for q in queries:
            start=time.perf_counter()
            hits=idx.search_papers(q['query'],3,mode)
            timings.append((time.perf_counter()-start)*1000)
            ids=[h['paper_id'] for h in hits];gold=set(q['relevant'])
            hit1=int(bool(ids) and ids[0] in gold)
            recall=len(set(ids)&gold)/len(gold)
            rr=next((1/(i+1) for i,pid in enumerate(ids) if pid in gold),0)
            metrics.append((hit1,recall,rr))
            details.append({'query_id':q['id'],'mode':mode,'query':q['query'],'gold':q['relevant'],'ranked':ids,'hit_at_1':hit1,'recall_at_3':recall,'mrr_at_3':rr})
        results[mode]={'hit_at_1':statistics.mean(x[0] for x in metrics),'recall_at_3':statistics.mean(x[1] for x in metrics),'mrr_at_3':statistics.mean(x[2] for x in metrics),'median_latency_ms':statistics.median(timings)}
    # Actual repeated cold import vs unchanged import; no artificial network latency.
    cold=[];warm=[]
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    for repeat in range(3):
        with tempfile.TemporaryDirectory(prefix='paper-agent-eval-') as temp:
            sample=Store(temp)
            for destination in [cold,warm]:
                start=time.perf_counter()
                for p in manifest:
                    sample.ingest(ROOT/'data'/'corpus'/(p['id']+'.pdf'),p['title'],p['id'],p['year'],'https://arxiv.org/abs/'+p['id'])
                destination.append(time.perf_counter()-start)
    cache={'cold_seconds':cold,'warm_seconds':warm,'cold_median_s':statistics.median(cold),'warm_median_s':statistics.median(warm),
        'median_reduction_pct':100*(1-statistics.median(warm)/statistics.median(cold))}
    context=[]
    for q in queries:
        pids=[h['paper_id'] for h in idx.search_papers(q['query'],3)]
        evidence=[e for pid in pids for e in select_evidence(idx,q['query'],pid)]
        full_text=sum(len(store.page(pid,p) or '') for pid in pids for p in range(1,next(x['pages'] for x in store.papers() if x['id']==pid)+1))
        excerpts=sum(len(e['text']) for e in evidence)
        context.append({'query_id':q['id'],'full_selected_papers_chars':full_text,'evidence_chars':excerpts,'reduction_pct':100*(1-excerpts/max(1,full_text))})
    gates=[]
    # 12 positives and 36 corruptions. This measures provenance detection only.
    for p in store.papers():
        e=next(c for c in chunks if c['paper_id']==p['id'])
        valid={'claim':e['text'],'quote':e['text'],'evidence_id':e['id']}
        candidates=[('valid',valid,True),('unknown_id',{**valid,'evidence_id':'not-in-run'},False),
            ('fabricated_quote',{**valid,'quote':'This is an invented assertion with no matching source text in this paper.'},False),
            ('short_quote',{**valid,'quote':'a'},False)]
        for kind,candidate,expected in candidates:
            accepted,reason=audit_claim(candidate,{e['id']:e},store)
            gates.append({'paper_id':p['id'],'kind':kind,'expected':expected,'accepted':accepted,'correct':accepted==expected,'reason':reason})
    summary={'scope':'Local seed-labeled 12-paper / 30-query English retrieval evaluation. Not held-out, not independently annotated, not an LLM answer-quality benchmark.',
        'query_sha256':hashlib.sha256(queries_path.read_bytes()).hexdigest(),'corpus_signature':store.signature(),
        'environment':{'python':platform.python_version(),'platform':platform.platform()},'papers':len(store.papers()),'pages':sum(p['pages'] for p in store.papers()),'chunks':len(chunks),'queries':len(queries),'index_build_s':index_seconds,
        'retrieval':results,'cache':cache,
        'context':{'unit':'characters, not tokens; selection only, quality tradeoff not evaluated','median_reduction_pct':statistics.median(x['reduction_pct'] for x in context)},
        'provenance_gate':{'cases':len(gates),'correct':sum(x['correct'] for x in gates),'description':'12 exact-source positives + 36 structural corruptions; semantic entailment NOT measured'}}
    for name,data in [('summary.json',summary),('retrieval_cases.json',details),('context_cases.json',context),('provenance_cases.json',gates)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=['# Reproducible local evaluation','',summary['scope'],'',f"Corpus: {summary['papers']} papers / {summary['pages']} pages / {summary['chunks']} chunks; {len(queries)} queries.",'',
        '| Retrieval | Hit@1 | Recall@3 | MRR@3 |','|---|---:|---:|---:|']
    for mode,m in results.items():rows.append(f"| {mode} | {m['hit_at_1']:.3f} | {m['recall_at_3']:.3f} | {m['mrr_at_3']:.3f} |")
    rows.extend(['',f"Cold import median: {cache['cold_median_s']:.4f}s; warm unchanged import: {cache['warm_median_s']:.4f}s; reduction: {cache['median_reduction_pct']:.2f}% (3 repeats).",'',
        f"Provenance gate: {sum(x['correct'] for x in gates)}/{len(gates)} constructed cases; not semantic accuracy.",'',
        f"Evidence selection median character reduction vs full selected papers: {summary['context']['median_reduction_pct']:.2f}%; not measured token/cost savings.",
        '', 'See raw per-query results, corpus provenance hashes, source code and test report for reproduction.'])
    (out/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
