"""Local-only HTTP UI. Files exposed by opaque IDs, never arbitrary paths."""
from __future__ import annotations
import json
import mimetypes
import os
import re
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .agents import Pipeline, Provider, safe_error
from .network import arxiv_pdf, arxiv_search
from .retrieval import Index
from .store import Store


class App:
    def __init__(self, root, runs):
        self.store=Store(root)
        self.runs=Path(runs)
        self.runs.mkdir(parents=True,exist_ok=True)
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.jobs={}
        self.lock=threading.Lock()

    def submit(self, query, mode, count, strategy="multi"):
        if not isinstance(query,str) or not query.strip() or len(query)>2000:
            raise ValueError("Question must contain 1-2000 characters")
        if mode not in {"llm","extractive"} or type(count)!=int or not 1<=count<=8:
            raise ValueError("Invalid mode or paper limit")
        if strategy not in {"single","multi"}:
            raise ValueError("Invalid strategy")
        provider=Provider() if mode=="llm" else None
        with self.lock:
            if sum(j["status"] in {"queued","running"} for j in self.jobs.values())>=4:
                raise ValueError("Queue full; wait for a current task")
            rid=uuid.uuid4().hex
            self.jobs[rid]={"run_id":rid,"status":"queued","query":query}
        def work():
            with self.lock:
                self.jobs[rid]["status"]="running"
            try:
                result=Pipeline(self.store,self.runs,provider).run(query,mode,count,rid,strategy=strategy)
                with self.lock:
                    self.jobs[rid]={"run_id":rid,"status":result["status"],"query":query}
            except Exception as exc:
                failure={"run_id":rid,"status":"failed","error":safe_error(exc),"query":query}
                folder=self.runs/rid
                folder.mkdir(exist_ok=True)
                (folder/"failure.json").write_text(json.dumps(failure),encoding="utf-8")
                with self.lock:
                    self.jobs[rid]=failure
        self.pool.submit(work)
        return {"run_id":rid,"status":"queued"}

    def history(self):
        found={}
        for path in sorted(self.runs.glob("*/report.json"),key=lambda p:p.stat().st_mtime,reverse=True)[:30]:
            try:
                r=json.loads(path.read_text(encoding="utf-8"))
                found[r["run_id"]]={k:r[k] for k in ["run_id","query","status","mode"]}
            except (ValueError,KeyError):
                continue
        with self.lock:
            found.update(self.jobs)
        return list(found.values())


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # Do not log user questions or request headers.

        def send(self, value, status=200, content_type="application/json; charset=utf-8"):
            payload=json.dumps(value,ensure_ascii=False).encode() if not isinstance(value,bytes) else value
            self.send_response(status)
            self.send_header("Content-Type",content_type)
            self.send_header("Content-Length",str(len(payload)))
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Cache-Control","no-store")
            self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def trusted(self):
            host=self.headers.get("Host","")
            allowed={f"127.0.0.1:{self.server.server_port}",f"localhost:{self.server.server_port}"}
            origin=self.headers.get("Origin")
            return host in allowed and (not origin or origin in {"http://"+h for h in allowed})

        def do_GET(self):
            if not self.trusted():
                return self.send({"error":"Local origin required"},403)
            url=urlparse(self.path)
            path=url.path
            try:
                if path in {"/","/app.js","/style.css"}:
                    name="index.html" if path=="/" else path[1:]
                    target=Path(__file__).parent/"static"/name
                    return self.send(target.read_bytes(),content_type={"index.html":"text/html; charset=utf-8","app.js":"application/javascript; charset=utf-8","style.css":"text/css; charset=utf-8"}[name])
                if path=="/api/state":
                    return self.send({"papers":app.store.papers(),"runs":app.history(),
                        "llm_configured":all(os.getenv(k) for k in ["PAPER_AGENT_API_KEY","PAPER_AGENT_BASE_URL","PAPER_AGENT_MODEL"]),
                        "model":os.getenv("PAPER_AGENT_MODEL", ""),"version":"0.1.0"})
                if path=="/api/search":
                    query=parse_qs(url.query).get("q",[""])[0][:2000]
                    return self.send({"hits":Index(app.store.chunks()).search(query,8)})
                match=re.fullmatch(r"/api/runs/([a-f0-9]{32})(?:/(trace|review\.md|comparison\.csv|papers\.bib|report\.json|evidence\.jsonl))?",path)
                if match:
                    rid,artifact=match.groups()
                    folder=app.runs/rid
                    if artifact=="trace":
                        trace=folder/"trace.jsonl"
                        rows=[]
                        if trace.exists():
                            for line in trace.read_text(encoding="utf-8").splitlines():
                                try:
                                    rows.append(json.loads(line))
                                except ValueError:
                                    pass
                        return self.send({"events":rows})
                    if artifact:
                        target=folder/artifact
                        if not target.is_file():
                            return self.send({"error":"Artifact unavailable"},404)
                        return self.send(target.read_bytes(),content_type="text/plain; charset=utf-8")
                    if (folder/"report.json").exists():
                        report=json.loads((folder/"report.json").read_text(encoding="utf-8"))
                        report["stale"]=report.get("corpus_signature")!=app.store.signature()
                        return self.send(report)
                    if (folder/"failure.json").exists():
                        return self.send(json.loads((folder/"failure.json").read_text(encoding="utf-8")))
                    with app.lock:
                        job=app.jobs.get(rid)
                    return self.send(job or {"error":"Unknown run"},200 if job else 404)
                match=re.fullmatch(r"/api/papers/([A-Za-z0-9_.-]+)/pdf",path)
                if match:
                    target=app.store.pdf(match[1])
                    return self.send(target.read_bytes(),content_type="application/pdf") if target else self.send({"error":"Unknown paper"},404)
                return self.send({"error":"Not found"},404)
            except Exception as exc:
                return self.send({"error":safe_error(exc)},500)

        def do_POST(self):
            if not self.trusted() or self.headers.get("X-Paper-Agent")!="1":
                return self.send({"error":"Local origin and X-Paper-Agent header required"},403)
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<=30*1024*1024:
                    return self.send({"error":"Invalid request size"},413)
                raw=self.rfile.read(length)
                if self.path=="/api/upload":
                    title=unquote(self.headers.get("X-Paper-Title","Imported paper"))[:300]
                    with tempfile.TemporaryDirectory() as temp:
                        path=Path(temp)/"upload.pdf"
                        path.write_bytes(raw)
                        result=app.store.ingest(path,title=title)
                    return self.send(result)
                body=json.loads(raw)
                if self.path=="/api/research":
                    return self.send(app.submit(body.get("query"),body.get("mode","extractive"),body.get("max_papers",4),body.get("strategy","multi")),202)
                if self.path=="/api/arxiv/search":
                    return self.send({"papers":arxiv_search(body.get("query",""))})
                if self.path=="/api/arxiv/import":
                    pid=body.get("id","")
                    pdf=arxiv_pdf(pid)
                    with tempfile.TemporaryDirectory() as temp:
                        path=Path(temp)/"paper.pdf"
                        path.write_bytes(pdf)
                        result=app.store.ingest(path,title=str(body.get("title",pid))[:300],paper_id=pid,
                            year=body.get("year"),url="https://arxiv.org/abs/"+pid)
                    return self.send(result)
                return self.send({"error":"Not found"},404)
            except (ValueError,TypeError) as exc:
                return self.send({"error":str(exc)[:250]},400)
            except Exception as exc:
                return self.send({"error":safe_error(exc)},502)
    return Handler


def serve(root="data",runs="runs",port=8765):
    app=App(root,runs)
    server=ThreadingHTTPServer(("127.0.0.1",port),handler(app))
    print(f"Paper Agent: http://127.0.0.1:{server.server_port}",flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        app.pool.shutdown(wait=False,cancel_futures=True)
