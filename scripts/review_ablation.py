"""Re-audit the SAME saved live candidates with quote-only review.

This is a model-to-model audit, not independent accuracy measurement.
"""
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_agent.agents import Provider,REVIEW_INSTRUCTION

def main():
    out=ROOT/'evaluation'/'results'
    live=json.loads((out/'live_summary.json').read_text(encoding='utf-8'))
    provider=Provider(model=live['model'])
    results=[]
    for case in live['cases']:
        if not case.get('run_id'):continue
        report=json.loads((ROOT/'runs'/case['run_id']/'report.json').read_text(encoding='utf-8'))
        judged=provider.call('Reviewer',REVIEW_INSTRUCTION,{'question':report['query'],'claims':report['claims']})
        row={'run_id':case['run_id'],'claims':len(report['claims']),'verdicts':judged.get('verdicts',[])}
        results.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    summary={'scope':'Same 24 previously accepted live claims re-audited by quote-only prompt. Changes in model verdicts are not measured human accuracy.',
        'model':provider.model,'usage':provider.usage,'cases':results}
    (out/'review_ablation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
