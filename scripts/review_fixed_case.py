"""Isolate reviewer policy using the exact same off-question candidate claims."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from paper_agent.agents import Provider,REVIEW_INSTRUCTION,QUESTION_REVIEW_INSTRUCTION
out=ROOT/'evaluation/depth_results'
data=json.loads((out/'ablation_live.json').read_text(encoding='utf-8'))
row=next(r for r in data['cases'] if r['id']=='n1' and r['arm']=='multi_v3')
report=json.loads((ROOT/'evaluation/depth_runs'/row['run_id']/'report.json').read_text(encoding='utf-8'))
payload={'question':report['query'],'claims':report['claims']}
results=[]
for policy,instruction in [('quote-only-v3',REVIEW_INSTRUCTION),('question-aware-v4',QUESTION_REVIEW_INSTRUCTION)]:
    p=Provider();result=p.call('Reviewer',instruction,payload)
    accepted=[v['index'] for v in result.get('verdicts',[]) if v.get('supported') is True and not v.get('unsupported_details') and (policy=='quote-only-v3' or v.get('relevant') is True)]
    results.append({'policy':policy,'response':result,'accepted_indices':accepted,'usage':p.usage})
    print(json.dumps({'policy':policy,'accepted_indices':accepted,'usage':p.usage}),flush=True)
(out/'review_fixed_case.json').write_text(json.dumps({'scope':'Single diagnostic case with frozen candidate claims; not a general quality estimate','payload':payload,'results':results},ensure_ascii=False,indent=2),encoding='utf-8')
