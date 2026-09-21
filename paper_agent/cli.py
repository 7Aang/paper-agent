import argparse
import json
import os
from pathlib import Path
from .agents import Pipeline,Provider
from .retrieval import Index
from .store import Store


def main():
    parser=argparse.ArgumentParser(description="Evidence-first Paper Agent")
    parser.add_argument("--data",default=os.getenv("PAPER_AGENT_HOME","data"))
    sub=parser.add_subparsers(dest="command",required=True)
    ingest=sub.add_parser("ingest")
    ingest.add_argument("pdf")
    ingest.add_argument("--title")
    corpus=sub.add_parser("corpus")
    corpus.add_argument("--manifest",default="data_manifest.json")
    corpus.add_argument("--pdf-dir",default="data/corpus")
    search=sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--mode",choices=["baseline","bm25","optimized"],default="optimized")
    research=sub.add_parser("research")
    research.add_argument("query")
    research.add_argument("--mode",choices=["extractive","llm"],default="extractive")
    research.add_argument("--papers",type=int,default=4)
    research.add_argument("--workers",type=int,default=3)
    research.add_argument("--runs",default="runs")
    research.add_argument("--strategy",choices=["single","multi"],default="multi")
    research.add_argument("--fresh",action="store_true",help="Bypass durable LLM step checkpoints")
    server=sub.add_parser("serve")
    server.add_argument("--port",type=int,default=8765)
    server.add_argument("--runs",default="runs")
    args=parser.parse_args()
    store=Store(args.data)
    if args.command=="ingest":
        output=store.ingest(args.pdf,args.title)
    elif args.command=="corpus":
        output=[]
        for p in json.loads(Path(args.manifest).read_text(encoding="utf-8")):
            output.append(store.ingest(Path(args.pdf_dir)/(p["id"]+".pdf"),p["title"],p["id"],p["year"],"https://arxiv.org/abs/"+p["id"]))
    elif args.command=="search":
        output=Index(store.chunks()).search_papers(args.query,5,args.mode)
    elif args.command=="research":
        output=Pipeline(store,args.runs,Provider() if args.mode=="llm" else None,args.workers,checkpoints=not args.fresh).run(args.query,args.mode,args.papers,strategy=args.strategy)
    else:
        from .server import serve
        serve(args.data,args.runs,args.port)
        return
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
