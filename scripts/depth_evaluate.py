"""Frozen new-query regression and controlled single/multi-agent ablation.

Labels are assistant-authored diagnostics, not independent human judgments.
Model review is never counted as semantic gold accuracy.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
import threading
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_agent.agents import Pipeline,Provider,select_evidence,safe_error
from paper_agent.retrieval import Index
from paper_agent.store import Store


class BoundedProvider:
    def __init__(self, inner, limit=8):
        self.inner=inner; self.limit=limit; self.attempts=0; self.lock=threading.Lock()
        self.base=inner.base; self.model=inner.model
    @property
    def usage(self): return self.inner.usage
    def call(self,*args):
        with self.lock:
            if self.attempts>=self.limit: raise RuntimeError('Experiment API call budget exhausted')
            self.attempts+=1
        return self.inner.call(*args)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    out=ROOT/'evaluation/depth_results';out.mkdir(exist_ok=True)
    store=Store(ROOT/'data');index=Index(store.chunks())
    source=ROOT/'evaluation/heldout_v2.json'
    freeze=json.loads((ROOT/'evaluation/heldout_v2.freeze.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==freeze['sha256']
    queries=json.loads(source.read_text(encoding='utf-8'))
    retrieval={}
    for mode in ['baseline','bm25','optimized']:
        rows=[]
        for q in queries:
            ranked=[h['paper_id'] for h in index.search_papers(q['query'],3,mode)]
            relevant=set(q['relevant'])
            rows.append({'id':q['id'],'ranked':ranked,'hit1':bool(ranked and ranked[0] in relevant),
              'recall3':len(set(ranked)&relevant)/len(relevant),
              'rr3':next((1/(i+1) for i,p in enumerate(ranked) if p in relevant),0)})
        retrieval[mode]={'hit_at_1':statistics.mean(r['hit1'] for r in rows),
          'recall_at_3':statistics.mean(r['recall3'] for r in rows),'mrr_at_3':statistics.mean(r['rr3'] for r in rows),'cases':rows}
    (out/'retrieval.json').write_text(json.dumps({'scope':freeze['scope'],'query_sha256':freeze['sha256'],'corpus_signature':store.signature(),'metrics':retrieval},indent=2),encoding='utf-8')
    cases=[
      {'id':'a1','query':'How does ColBERT preserve token-level query document interaction while allowing document representations to be precomputed?','answerable':True,'target_papers':['2004.12832']},
      {'id':'a2','query':'How does Reflexion improve an agent across trials without updating its model weights?','answerable':True,'target_papers':['2303.11366']},
      {'id':'a3','query':'Compare how ReAct uses environment observations and how Reflexion stores feedback for subsequent trials.','answerable':True,'target_papers':['2210.03629','2303.11366']},
      {'id':'a4','query':'Compare how RAPTOR and GraphRAG organize summaries to answer questions that need a broad view of documents.','answerable':True,'target_papers':['2401.18059','2404.16130']},
      {'id':'n1','query':'What was the exact measured p99 latency in milliseconds when ReAct was deployed at Hangzhou Airport in August 2026?','answerable':False,'target_papers':[]},
      {'id':'n2','query':'What is the exact number of NVIDIA H200 GPUs used by the ColBERT paper in its 2026 production deployment?','answerable':False,'target_papers':[]},
      {'id':'n3','query':'What exact annual licensing price in Chinese yuan does the Reflexion paper report for a commercial enterprise installation?','answerable':False,'target_papers':[]},
      {'id':'n4','query':'What percentage did RAPTOR reduce battery consumption in a real PX4 drone flight test conducted in September 2026?','answerable':False,'target_papers':[]},
    ]
    packets={}
    for c in cases:
        selected=index.search_papers(c['query'],3)
        packets[c['id']]={'evidence':[e for p in selected for e in select_evidence(index,c['query'],p['paper_id'])]}
    spec={'cases':cases,'packets':packets,'scope':'Same frozen retrieval evidence and model; planner disabled in all arms to isolate orchestration. Single one generation; multi per-paper workers + ordering + review. All arms share ceiling 8 requests x 1800 maximum output tokens; actual usage measured. No cache. No independent human semantic gold.',
      'corpus_signature':store.signature()}
    spec_path=out/'ablation_frozen.json'
    spec_path.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.live:
        print(json.dumps({k:{m:v for m,v in val.items() if m!='cases'} for k,val in retrieval.items()}));return
    results=[];annotation=[]
    arms=[('single','single','question-aware-v4'),('multi_v3','multi','quote-only-v3'),('multi_v4','multi','question-aware-v4')]
    for ci,c in enumerate(cases):
        # Rotate order to avoid always putting the same arm first.
        for arm,strategy,policy in arms[ci%3:]+arms[:ci%3]:
            provider=BoundedProvider(Provider())
            row={'id':c['id'],'arm':arm,'answerable':c['answerable']}
            try:
                r=Pipeline(store,ROOT/'evaluation/depth_runs',provider,checkpoints=False).run(c['query'],'llm',3,
                    strategy=strategy,evidence_packet=packets[c['id']],review_policy=policy)
                present={x['paper_id'] for x in r['claims']}
                row.update({'status':r['status'],'elapsed_s':r['elapsed_s'],'usage':r['usage'],'attempts':provider.attempts,
                  'claims':len(r['claims']),'rejected':len(r['rejected']),'run_id':r['run_id'],
                  'abstained':not r['claims'],'target_paper_coverage':len(present&set(c['target_papers']))/len(c['target_papers']) if c['target_papers'] else None,
                  'all_quotes_on_page':all(x['quote'] in (store.page(x['paper_id'],x['page']) or '') for x in r['claims'])})
                for i,x in enumerate(r['claims']):
                    annotation.append({'case':c['id'],'arm':arm,'query':c['query'],'claim_index':i,'claim':x['claim'],'quote':x['quote'],
                      'paper_id':x['paper_id'],'page':x['page'],'human_supported':'','human_relevant':'','human_notes':''})
            except Exception as exc:
                row.update({'status':'failed','error':safe_error(exc),'usage':provider.usage,'attempts':provider.attempts})
            results.append(row)
            output={'model':provider.model,'spec_sha256':hashlib.sha256(spec_path.read_bytes()).hexdigest(),'scope':spec['scope'],'cases':results}
            (out/'ablation_live.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(row,ensure_ascii=False),flush=True)
            if row['status']=='failed':
                raise RuntimeError('Live experiment stopped on failed arm; partial results preserved')
    with (out/'human_review_pending.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(annotation[0]));writer.writeheader();writer.writerows(annotation)


if __name__=='__main__':main()
