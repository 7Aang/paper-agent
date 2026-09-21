import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path

import fitz
import pytest

from paper_agent.agents import Pipeline, Provider, audit_claim
from paper_agent.network import arxiv_pdf
from paper_agent.retrieval import Index
from paper_agent.server import App, handler
from paper_agent.store import Store, normalize


def pdf(path, texts):
    doc=fitz.open()
    for text in texts:
        page=doc.new_page()
        page.insert_textbox(fitz.Rect(40,40,550,800),text,fontsize=12)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def populated(tmp_path):
    store=Store(tmp_path/'db')
    path=pdf(tmp_path/'one.pdf',[
        'Dense retrieval uses a dual encoder to match questions with relevant passages. Training uses hard negatives and positive examples.',
        'The method has limitations on out of domain datasets. Evaluation conditions must be compared carefully before claiming improvements.'
    ])
    store.ingest(path,'Dense retrieval study','one',2020,'https://example.org/one')
    return store


def test_page_numbers_and_exact_quotes(populated):
    assert 'dual encoder' in populated.page('one',1)
    assert 'limitations' in populated.page('one',2)
    for c in populated.chunks():
        assert c['text'] in populated.page(c['paper_id'],c['page'])


def test_cache_hit_and_metadata_update(populated):
    result=populated.ingest(populated.pdf('one'),'New title','one',2021)
    assert result['cached']
    assert populated.papers()[0]['title']=='New title'
    assert populated.papers()[0]['year']==2021


def test_database_handles_are_closed(populated):
    populated.papers();populated.chunks();populated.signature()
    renamed=populated.db.with_suffix('.renamed')
    populated.db.rename(renamed)
    renamed.rename(populated.db)


def test_upload_duplicate_content_preserves_identity(populated):
    result=populated.ingest(populated.pdf('one'),'uploaded-file-name')
    assert result['id']=='one' and result['cached']
    assert len(populated.papers())==1
    assert populated.papers()[0]['title']=='Dense retrieval study'


def test_content_change_invalidates_cache(populated,tmp_path):
    p=pdf(tmp_path/'changed.pdf',['A completely different investigation of stellar spectra and stellar evolution. This new content replaces previous retrieval evidence.'])
    assert not populated.ingest(p,'Stellar spectra','one')['cached']
    assert 'dual encoder' not in populated.page('one',1)
    assert populated.page('one',2) is None


def test_corrupt_pdf_preserves_previous_record(populated,tmp_path):
    p=tmp_path/'bad.pdf';p.write_bytes(b'%PDF fake corruption')
    with pytest.raises(Exception):
        populated.ingest(p,paper_id='one')
    assert 'dual encoder' in populated.page('one',1)


@pytest.mark.parametrize('pid',['../x','..','/etc/passwd','a/b','x\\y'])
def test_ingest_path_ids_rejected(populated,pid):
    with pytest.raises(ValueError):
        populated.ingest(populated.pdf('one'),paper_id=pid)


def test_scanned_pdf_reports_no_text(tmp_path):
    p=pdf(tmp_path/'scan.pdf',[''])
    with pytest.raises(ValueError,match='OCR'):
        Store(tmp_path/'db').ingest(p)


def test_concurrent_duplicate_ingestion(populated):
    p=populated.pdf('one')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:populated.ingest(p,paper_id='one'),range(8)))
    assert all(r['cached'] for r in results)
    assert len(populated.papers())==1


def test_retrieval_empty_and_unknown(populated):
    idx=Index(populated.chunks())
    assert idx.search('the and of')==[]
    assert idx.search('qzxvabcunknown')==[]


@pytest.mark.parametrize('mode',['baseline','bm25','optimized'])
def test_retrieval_returns_relevant_page(populated,mode):
    assert Index(populated.chunks()).search('dual encoder hard negatives',mode=mode)[0]['page']==1


def test_paper_filter(populated):
    assert Index(populated.chunks()).search('retrieval',paper_id='missing')==[]


def test_gate_accepts_exact_provenance(populated):
    c=populated.chunks()[0]
    assert audit_claim({'claim':c['text'],'quote':c['text'],'evidence_id':c['id']},{c['id']:c},populated)[0]


@pytest.mark.parametrize('mutation',['unknown','short','fabricated','wrong_page','empty_claim'])
def test_gate_rejects_bad_evidence(populated,mutation):
    c=populated.chunks()[0].copy()
    claim={'claim':c['text'],'quote':c['text'],'evidence_id':c['id']}
    if mutation=='unknown':claim['evidence_id']='not-found'
    elif mutation=='short':claim['quote']='short'
    elif mutation=='fabricated':claim['quote']='Fabricated statement that is never present anywhere in the source document.'
    elif mutation=='wrong_page':c['page']=2
    elif mutation=='empty_claim':claim['claim']=''
    assert not audit_claim(claim,{c['id']:c},populated)[0]


def test_gate_does_not_claim_semantic_proof(populated):
    c=populated.chunks()[0]
    # The deliberately wrong claim still has a genuine quote; document the limit.
    ok,reason=audit_claim({'claim':'This system cures cancer.','quote':c['text'],'evidence_id':c['id']},{c['id']:c},populated)
    assert ok and reason=='provenance_verified'


def test_malformed_evidence_id_rejected(populated):
    assert audit_claim({'claim':'test','quote':'test','evidence_id':[]},{},populated)==(False,'invalid_evidence_id')


def test_existing_run_cannot_be_overwritten(populated,tmp_path):
    pipeline=Pipeline(populated,tmp_path/'runs')
    result=pipeline.run('retrieval')
    with pytest.raises(FileExistsError):
        pipeline.run('retrieval',run_id=result['run_id'])


def test_database_failure_rolls_back(populated):
    with pytest.raises(RuntimeError):
        with populated.connect() as conn:
            conn.execute('DELETE FROM papers')
            raise RuntimeError('simulated write failure')
    assert len(populated.papers())==1


def test_offline_pipeline_exports_and_provenance(populated,tmp_path):
    report=Pipeline(populated,tmp_path/'runs').run('dense retrieval dual encoder')
    assert report['status']=='completed'
    assert report['usage']['calls']==0
    folder=tmp_path/'runs'/report['run_id']
    for name in ['report.json','review.md','evidence.jsonl','trace.jsonl','comparison.csv','papers.bib']:
        assert (folder/name).stat().st_size>0
    for c in report['claims']:
        assert c['claim']==c['quote']
        assert c['quote'] in populated.page(c['paper_id'],c['page'])


def test_empty_corpus_abstains(tmp_path):
    r=Pipeline(Store(tmp_path/'db'),tmp_path/'runs').run('retrieval')
    assert r['status']=='insufficient_evidence' and not r['claims']


def test_no_silent_llm_fallback(populated,tmp_path):
    with pytest.raises(ValueError,match='requires'):
        Pipeline(populated,tmp_path/'runs').run('retrieval',mode='llm')


class FakeProvider:
    """Contract fixture only; never used to measure model quality."""
    def __init__(self, reject=False):
        self.usage={'calls':0}
        self.reject=reject
    def call(self,role,instruction,payload):
        self.usage['calls']+=1
        if role=='Planner':return {'queries':['dense retrieval'],'dimensions':['method']}
        if role=='Researcher':
            e=payload['evidence'][0]
            return {'claims':[{'claim':e['text'],'quote':e['text'],'evidence_id':e['id'],'kind':'method'}]}
        if role=='Synthesizer':return {'order':list(range(len(payload['claims'])))}
        return {'verdicts':[{'index':i,'supported':not self.reject,'relevant':True,'reason':'fixture'} for i in range(len(payload['claims']))]}


def test_four_role_llm_contract(populated,tmp_path):
    p=FakeProvider()
    r=Pipeline(populated,tmp_path/'runs',p).run('dense retrieval','llm')
    assert r['status']=='completed' and p.usage['calls']==4
    assert r['claims'][0]['semantic_status']=='model_reviewed_not_ground_truth'


def test_pipeline_resumes_after_reviewer_outage(populated,tmp_path):
    class FailsOnce(FakeProvider):
        fail=True
        def call(self,role,instruction,payload):
            if role=='Reviewer' and self.fail:
                self.usage['calls']+=1
                raise TimeoutError('injected outage')
            return super().call(role,instruction,payload)
    first=FailsOnce()
    with pytest.raises(TimeoutError):
        Pipeline(populated,tmp_path/'runs',first).run('dense retrieval','llm')
    second=FailsOnce();second.fail=False
    report=Pipeline(populated,tmp_path/'runs',second).run('dense retrieval','llm')
    assert report['status']=='completed'
    assert second.usage['calls']==1
    assert report['checkpoints']=={'hits':3,'misses':1}


def test_metadata_changes_signature(populated):
    before=populated.signature()
    p=populated.papers()[0]
    populated.ingest(populated.pdf('one'),p['title'],'one',2026,'https://example.org/revised')
    assert populated.signature()!=before


def test_single_agent_uses_one_call_and_marks_semantics_unreviewed(populated,tmp_path):
    p=FakeProvider()
    report=Pipeline(populated,tmp_path/'runs',p).run('dense retrieval','llm',strategy='single')
    assert report['status']=='completed' and p.usage['calls']==1
    assert report['review_policy']=='not_evaluated'
    assert all(c['semantic_status']=='not_evaluated' for c in report['claims'])


def test_true_but_irrelevant_claims_are_rejected(populated,tmp_path):
    class Irrelevant(FakeProvider):
        def call(self,role,instruction,payload):
            result=super().call(role,instruction,payload)
            if role=='Reviewer':
                for v in result['verdicts']:v['relevant']=False
            return result
    report=Pipeline(populated,tmp_path/'runs',Irrelevant()).run('How much does a license cost?','llm')
    assert not report['claims']
    assert report['rejected'][0]['reason']=='irrelevant_or_missing_relevance_verdict'


def test_tampered_frozen_evidence_rejected_before_api(populated,tmp_path):
    p=FakeProvider()
    chunk=dict(populated.chunks()[0]);chunk['text']='tampered evidence'
    with pytest.raises(ValueError,match='Frozen evidence'):
        Pipeline(populated,tmp_path/'runs',p).run('retrieval','llm',evidence_packet={'evidence':[chunk]})
    assert p.usage['calls']==0


def test_reviewer_rejection_is_enforced(populated,tmp_path):
    r=Pipeline(populated,tmp_path/'runs',FakeProvider(True)).run('dense retrieval','llm')
    assert not r['claims'] and r['rejected']


def test_researcher_failure_is_reported(populated,tmp_path):
    class Broken(FakeProvider):
        def call(self,role,instruction,payload):
            if role=='Researcher':raise TimeoutError('secret-test-value')
            return super().call(role,instruction,payload)
    r=Pipeline(populated,tmp_path/'runs',Broken()).run('retrieval','llm')
    assert r['errors'] and r['status']=='insufficient_evidence'
    assert 'secret-test-value' not in json.dumps(r)


@pytest.mark.parametrize('pid',['https://127.0.0.1/x','../../secret','file:///etc/passwd','2005.11401/../../x'])
def test_arxiv_rejects_arbitrary_urls(pid):
    with pytest.raises(ValueError):arxiv_pdf(pid)


@pytest.mark.parametrize('base',['http://api.example.org','https://user:password@example.org','https://example.org?key=secret'])
def test_provider_rejects_unsafe_configuration(base):
    with pytest.raises(ValueError):Provider('test',base,'test-model')


@pytest.fixture
def live_http(populated,tmp_path):
    app=App(populated.root,tmp_path/'runs')
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(app))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown();server.server_close();app.pool.shutdown()


def test_http_state_and_pdf(live_http):
    state=json.load(urllib.request.urlopen(live_http+'/api/state'))
    assert len(state['papers'])==1
    assert urllib.request.urlopen(live_http+'/api/papers/one/pdf').read().startswith(b'%PDF')


def test_http_mode_menu_contains_valid_options(live_http):
    from html.parser import HTMLParser
    class MenuParser(HTMLParser):
        active=None
        choices={}
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if tag=='select':
                self.active=attrs.get('id');self.choices[self.active]=[]
            if tag=='option' and self.active:
                self.choices[self.active].append(attrs.get('value'))
        def handle_endtag(self,tag):
            if tag=='select':self.active=None
    parser=MenuParser()
    parser.feed(urllib.request.urlopen(live_http+'/').read().decode())
    assert parser.choices['mode']==['extractive','llm']
    assert parser.choices['strategy']==['multi','single']


def test_http_requires_csrf_header(live_http):
    request=urllib.request.Request(live_http+'/api/research',data=b'{}')
    with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(request)
    assert e.value.code==403


def test_http_rejects_foreign_origin(live_http):
    request=urllib.request.Request(live_http+'/api/state',headers={'Origin':'https://evil.example'})
    with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(request)
    assert e.value.code==403


def test_http_no_path_traversal(live_http):
    with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(live_http+'/api/runs/../../papers.sqlite3')
    assert e.value.code==404


def test_http_upload_rejects_non_pdf(live_http):
    request=urllib.request.Request(live_http+'/api/upload',data=b'not pdf',headers={'X-Paper-Agent':'1'})
    with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(request)
    assert e.value.code==400


def test_http_async_job_and_artifacts(live_http):
    import time
    request=urllib.request.Request(live_http+'/api/research',data=json.dumps({'query':'dense retrieval','mode':'extractive','max_papers':1}).encode(),headers={'X-Paper-Agent':'1','Content-Type':'application/json'})
    job=json.load(urllib.request.urlopen(request))
    for _ in range(50):
        result=json.load(urllib.request.urlopen(live_http+'/api/runs/'+job['run_id']))
        if 'claims' in result:break
        time.sleep(.05)
    assert result['status']=='completed'
    for artifact in ['review.md','comparison.csv','papers.bib','report.json','evidence.jsonl','trace']:
        assert urllib.request.urlopen(live_http+'/api/runs/'+job['run_id']+'/'+artifact).status==200


def test_http_invalid_mode_rejected(live_http):
    request=urllib.request.Request(live_http+'/api/research',data=json.dumps({'query':'retrieval','mode':'invalid'}).encode(),headers={'X-Paper-Agent':'1'})
    with pytest.raises(urllib.error.HTTPError) as exc:urllib.request.urlopen(request)
    assert exc.value.code==400
