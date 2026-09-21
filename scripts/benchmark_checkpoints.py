"""Real API cold/warm comparison. No credentials are persisted."""
import json
import sys
import time
import os
import uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_agent.agents import Pipeline,Provider
from paper_agent.store import Store


def main():
    os.environ['PAPER_AGENT_MODEL_REVISION']='benchmark-'+uuid.uuid4().hex
    store=Store(ROOT/'data')
    results=[]
    questions=[
      'How do ReAct and Reflexion use feedback to improve reasoning?',
      'How do DPR and ColBERT differ in passage retrieval?']
    for number,question in enumerate(questions):
        cold=None
        for variant in ['baseline','cold','warm']:
            provider=Provider()
            start=time.perf_counter()
            report=Pipeline(store,ROOT/'evaluation/upgrade_runs',provider,checkpoints=variant!='baseline').run(question,'llm',3)
            row={'case':number,'variant':variant,'query':question,'run_id':report['run_id'],
              'wall_s':time.perf_counter()-start,'usage':report['usage'],'checkpoints':report.get('checkpoints'),
              'claims':len(report['claims']),'rejected':len(report['rejected']),'status':report['status'],
              'quotes_verified':all(c['quote'] in (store.page(c['paper_id'],c['page']) or '') for c in report['claims'])}
            if variant=='cold':cold=report
            if variant=='warm':
                row['claims_identical_to_cold']=report['claims']==cold['claims']
                assert row['claims_identical_to_cold']
                assert report['usage']['calls']==0
            results.append(row)
            output={'model':provider.model,'cases':results,'scope':'Two questions, 3 candidate papers each; baseline disables checkpoints; cold then warm uses persisted cache across new Pipeline/Provider instances. Repeated identical inputs, not unseen-task quality or universal speedup.'}
            path=ROOT/'evaluation/results/checkpoints_live.json'
            path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
