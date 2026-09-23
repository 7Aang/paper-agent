"""Transactional PDF ingestion and page-addressable evidence storage."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import fitz

PARSER_VERSION = "blocks-page-180-30-v1"


def normalize(text: str) -> str:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return " ".join(text.replace("\x00", "").split())


def parse_pdf(path: Path) -> list[dict]:
    pages = []
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("Encrypted PDF requires an unlocked copy")
        if len(doc) > 500:
            raise ValueError("PDF exceeds 500-page limit")
        for number, page in enumerate(doc, 1):
            # Preserve text block order; do not interleave lines from two columns.
            blocks = page.get_text("blocks")
            text = normalize("\n".join(b[4] for b in blocks if b[6] == 0))
            pages.append({"page": number, "text": text})
    if sum(len(p["text"]) for p in pages) < 80:
        raise ValueError("No usable text layer. Scanned PDFs require OCR (not included).")
    return pages


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "pdfs").mkdir(exist_ok=True)
        self.db = self.root / "papers.sqlite3"
        with self.connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS papers(
                  id TEXT PRIMARY KEY, title TEXT NOT NULL, year INTEGER,
                  url TEXT NOT NULL, sha256 TEXT NOT NULL, parser TEXT NOT NULL,
                  filename TEXT NOT NULL, pages INTEGER NOT NULL, updated REAL NOT NULL,
                  authors TEXT NOT NULL DEFAULT '[]', doi TEXT, source_id TEXT);
                CREATE TABLE IF NOT EXISTS pages(
                  paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                  page INTEGER NOT NULL, text TEXT NOT NULL, PRIMARY KEY(paper_id,page));
                CREATE TABLE IF NOT EXISTS chunks(
                  id TEXT PRIMARY KEY,
                  paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                  page INTEGER NOT NULL, text TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
            """)
            columns={row[1] for row in conn.execute("PRAGMA table_info(papers)")}
            for name,definition in [("authors","TEXT NOT NULL DEFAULT '[]'"),("doi","TEXT"),("source_id","TEXT")]:
                if name not in columns:
                    conn.execute(f"ALTER TABLE papers ADD COLUMN {name} {definition}")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def ingest(self, path: str | Path, title: str | None = None,
               paper_id: str | None = None, year: int | None = None, url: str = "",
               authors: list[str] | None = None, doi: str | None = None,
               source_id: str | None = None) -> dict:
        start = time.perf_counter()
        path = Path(path)
        if path.stat().st_size > 30 * 1024 * 1024:
            raise ValueError("PDF exceeds 30 MB")
        payload = path.read_bytes()
        if not payload.startswith(b"%PDF"):
            raise ValueError("Expected a PDF file")
        digest = hashlib.sha256(payload).hexdigest()
        if paper_id is None:
            with self.connect() as conn:
                duplicate = conn.execute("SELECT id FROM papers WHERE sha256=?",(digest,)).fetchone()
            if duplicate:
                paper_id = duplicate["id"]
                # Preserve bibliographic title when the UI uploads the same PDF by filename.
                title = None
        paper_id = paper_id or digest[:16]
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", paper_id) or paper_id in {".", ".."}:
            raise ValueError("Invalid paper id")
        with self.connect() as conn:
            previous = conn.execute("SELECT * FROM papers WHERE id=?",(paper_id,)).fetchone()
        if previous and previous["sha256"] == digest and previous["parser"] == PARSER_VERSION and (self.root/"pdfs"/previous["filename"]).is_file():
            # Metadata changes must not be lost on a cache hit.
            with self.connect() as conn:
                conn.execute("UPDATE papers SET title=?,year=?,url=?,authors=?,doi=?,source_id=? WHERE id=?",
                    (title or previous["title"], year if year is not None else previous["year"], url or previous["url"],
                     json.dumps(authors,ensure_ascii=False) if authors is not None else previous["authors"],
                     doi if doi is not None else previous["doi"],source_id if source_id is not None else previous["source_id"],paper_id))
            return {"id":paper_id,"cached":True,"pages":previous["pages"],"seconds":time.perf_counter()-start}
        pages = parse_pdf(path)
        filename = digest + ".pdf"
        target = self.root / "pdfs" / filename
        if not target.exists():
            # Content-addressed file names permit safe concurrent imports.
            target.write_bytes(payload)
        chunks = []
        for page in pages:
            words = page["text"].split()
            for offset in range(0, len(words), 150):
                text = " ".join(words[offset:offset+180])
                if len(text) < 25:
                    continue
                chunks.append((f"{paper_id}:p{page['page']}:w{offset}",paper_id,page["page"],text))
        with self.connect() as conn:
            conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
            conn.execute("INSERT INTO papers(id,title,year,url,sha256,parser,filename,pages,updated,authors,doi,source_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (paper_id,title or path.stem,year,url,digest,PARSER_VERSION,filename,len(pages),time.time(),
                 json.dumps(authors or [],ensure_ascii=False),doi,source_id))
            conn.executemany("INSERT INTO pages VALUES(?,?,?)",[(paper_id,p["page"],p["text"]) for p in pages])
            conn.executemany("INSERT INTO chunks VALUES(?,?,?,?)",chunks)
        return {"id":paper_id,"cached":False,"pages":len(pages),"chunks":len(chunks),"seconds":time.perf_counter()-start}

    def papers(self) -> list[dict]:
        with self.connect() as conn:
            rows=[dict(x) for x in conn.execute("SELECT * FROM papers ORDER BY title")]
        for row in rows:
            try: row["authors"]=json.loads(row.get("authors") or "[]")
            except ValueError: row["authors"]=[]
        return rows

    def chunks(self) -> list[dict]:
        with self.connect() as conn:
            return [dict(x) for x in conn.execute("SELECT c.*,p.title,p.url,p.year FROM chunks c JOIN papers p ON c.paper_id=p.id ORDER BY c.id")]

    def chunk(self, chunk_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT c.*,p.title,p.url,p.year FROM chunks c JOIN papers p ON c.paper_id=p.id WHERE c.id=?",(chunk_id,)).fetchone()
            return dict(row) if row else None

    def page(self, paper_id: str, page: int) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT text FROM pages WHERE paper_id=? AND page=?",(paper_id,page)).fetchone()
            return row[0] if row else None

    def pdf(self, paper_id: str) -> Path | None:
        with self.connect() as conn:
            row = conn.execute("SELECT filename FROM papers WHERE id=?",(paper_id,)).fetchone()
            return self.root / "pdfs" / row[0] if row else None

    def signature(self) -> str:
        payload = [(p["id"],p["sha256"],p["title"],p["parser"],p["year"],p["url"],p["authors"],p.get("doi"),p.get("source_id")) for p in self.papers()]
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
