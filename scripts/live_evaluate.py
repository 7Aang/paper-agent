"""Real provider smoke suite, opt-in credentials via environment only."""
import json
import os
import sys
import time
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_agent.agents import Pipeline,Provider,safe_error
from paper_agent.network import request_bytes,arxiv_search
from paper_agent.store import Store

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='live_summary.json')
    args=parser.parse_args()
    if not args.output.endswith('.json') or Path(args.output).name!=args.output:
        raise ValueError('Output must be a JSON filename')
    out=ROOT/'evaluation'/'results';out.mkdir(exist_ok=True)
    store=Store(ROOT/'data')
    results=[]
    try:
        raw=request_bytes(os.environ['PAPER_AGENT_BASE_URL'].rstrip('/')+'/models',headers={'Authorization':'Bearer '+os.environ['PAPER_AGENT_API_KEY']},timeout=30)
        models=[x['id'] for x in json.loads(raw).get('data',[])]
        selected=next((m for m in ['deepseek-flash','deepseek-v4-flash','deepseek-chat'] if m in models),os.environ.get('PAPER_AGENT_MODEL','deepseek-flash'))
        os.environ['PAPER_AGENT_MODEL']=selected
        print('MODEL',selected,flush=True)
    except Exception as exc:
        failure={'status':'provider_unavailable','error':safe_error(exc),'http_status':getattr(exc,'code',None)}
        (out/args.output).write_text(json.dumps(failure,indent=2))
        print(json.dumps(failure),flush=True)
        return
    cases=[
        'How do ReAct and Reflexion use feedback to improve language agent performance?',
        'Compare dense passage retrieval and ColBERT late interaction. What evidence supports their design differences?',
        '比较 Self-RAG 和 Corrective Retrieval Augmented Generation 如何处理不可靠的检索证据。'
    ]
    for i,query in enumerate(cases,1):
        provider=Provider()
        start=time.perf_counter()
        try:
            r=Pipeline(store,ROOT/'runs',provider).run(query,'llm',3)
            row={'case':i,'query':query,'run_id':r['run_id'],'status':r['status'],'claims':len(r['claims']),'rejected':len(r['rejected']),
                'errors':r['errors'],'elapsed_s':r['elapsed_s'],'usage':r['usage'],'review_policy':r['review_policy'],
                'all_quotes_on_source_page':all(c['quote'] in (store.page(c['paper_id'],c['page']) or '') for c in r['claims'])}
        except Exception as exc:
            row={'case':i,'query':query,'status':'failed','error':safe_error(exc),'http_status':getattr(exc,'code',None),'elapsed_s':time.perf_counter()-start,'usage':provider.usage}
        results.append(row)
        summary={'model':selected,'provider':'https://api.deepseek.com','cases':results,
            'scope':'3 real end-to-end smoke cases, not a statistical quality benchmark. Quote provenance checked against PDF text. Semantic reviewer is the same model family, not independent gold truth.'}
        (out/args.output).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    try:
        found=arxiv_search('retrieval augmented generation',2)
        (out/'connector_smoke.json').write_text(json.dumps({'status':'passed' if found else 'empty','results':found},ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception as exc:
        (out/'connector_smoke.json').write_text(json.dumps({'status':'failed','error':safe_error(exc),'http_status':getattr(exc,'code',None)}),encoding='utf-8')

if __name__=='__main__':main()
