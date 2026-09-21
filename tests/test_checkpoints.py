import pytest
from paper_agent.checkpoints import CheckpointProvider, StepCache


class CountingProvider:
    base='https://fixture.invalid'
    model='fixture-model'
    def __init__(self):
        self.usage={'calls':0}
        self.fail=False
    def call(self, role, instruction, payload):
        self.usage['calls']+=1
        if self.fail: raise RuntimeError('injected outage')
        return {'queries':[payload['question']]}


def test_restart_hits_and_input_model_prompt_revision_invalidate(tmp_path):
    path=tmp_path/'cache.db'
    p=CountingProvider()
    cp=CheckpointProvider(p,StepCache(path))
    payload={'question':'Dense retrieval'}
    expected=cp.call('Planner','plan',payload)
    fresh=CountingProvider()
    restarted=CheckpointProvider(fresh,StepCache(path))
    assert restarted.call('Planner','plan',payload)==expected
    assert fresh.usage['calls']==0
    restarted.call('Planner','changed prompt',payload)
    restarted.call('Planner','plan',{'question':'New evidence'})
    fresh.model='new-model'
    restarted.call('Planner','plan',payload)
    restarted.revision='new-policy'
    restarted.call('Planner','plan',payload)
    assert fresh.usage['calls']==4


def test_errors_are_not_cached(tmp_path):
    p=CountingProvider(); cp=CheckpointProvider(p,StepCache(tmp_path/'cache.db'))
    p.fail=True
    with pytest.raises(RuntimeError): cp.call('Planner','plan',{'question':'q'})
    p.fail=False
    cp.call('Planner','plan',{'question':'q'})
    assert p.usage['calls']==2


@pytest.mark.parametrize('damage',['expired','corrupt'])
def test_expired_or_corrupt_checkpoint_recomputed(tmp_path,damage):
    cache=StepCache(tmp_path/'cache.db'); p=CountingProvider()
    cp=CheckpointProvider(p,cache)
    cp.call('Planner','plan',{'question':'q'})
    with cache.connect() as conn:
        conn.execute('UPDATE steps SET '+('created=0' if damage=='expired' else "response='{}'"))
    cp.call('Planner','plan',{'question':'q'})
    assert p.usage['calls']==2
