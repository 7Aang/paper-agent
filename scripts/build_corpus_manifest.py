"""Build a checked-in 50-paper arXiv manifest from official metadata.

The generated manifest is the frozen corpus definition. Search is only used to
discover candidates; every selected record is then validated against its arXiv
abstract page and receives title, author, date, abstract and PDF metadata.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_QUERIES = (
    "retrieval augmented generation",
    "language agents",
    "tool use large language models",
    "multi agent large language models",
)
USER_AGENT = "PaperAgent-Educational/0.3 (public arXiv corpus builder)"


def fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read(4 * 1024 * 1024).decode("utf-8", errors="replace")


def search_ids(query: str, size: int = 50) -> list[str]:
    url = "https://arxiv.org/search/?" + urllib.parse.urlencode(
        {
            "query": query,
            "searchtype": "all",
            "abstracts": "show",
            "order": "-announced_date_first",
            "size": size,
        }
    )
    page = fetch_text(url)
    found = re.findall(r"https://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", page)
    return list(dict.fromkeys(found))


def meta_values(page: str, name: str) -> list[str]:
    # arXiv currently emits name before content, but accept either attribute order.
    patterns = (
        rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]+content=["\']([^"\']*)["\']',
        rf'<meta[^>]+content=["\']([^"\']*)["\'][^>]+name=["\']{re.escape(name)}["\']',
    )
    values: list[str] = []
    for pattern in patterns:
        values.extend(re.findall(pattern, page, flags=re.I))
    return [" ".join(html.unescape(value).split()) for value in values]


def metadata(arxiv_id: str) -> dict:
    page = fetch_text(f"https://arxiv.org/abs/{arxiv_id}")
    titles = meta_values(page, "citation_title")
    authors = meta_values(page, "citation_author")
    dates = meta_values(page, "citation_date")
    abstracts = meta_values(page, "citation_abstract")
    pdfs = meta_values(page, "citation_pdf_url")
    dois = meta_values(page, "citation_doi")
    if not titles or not authors or not dates or not pdfs:
        raise ValueError("missing required citation metadata")
    year_match = re.search(r"(19|20)\d{2}", dates[0])
    if not year_match:
        raise ValueError("missing publication year")
    return {
        "id": arxiv_id,
        "title": titles[0],
        "authors": authors,
        "year": int(year_match.group(0)),
        "doi": dois[0] if dois else None,
        "source_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": pdfs[0].replace("http://", "https://"),
        "arxiv_id": arxiv_id,
        "abstract": abstracts[0] if abstracts else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--output", default="data_manifest.json")
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--refresh", action="store_true", help="Re-select additions while keeping the 12 seed papers")
    parser.add_argument("--exclude", action="append", default=[], help="arXiv ID to exclude after a failed download")
    args = parser.parse_args()
    if args.target < 12:
        raise SystemExit("target must keep at least the existing 12-paper seed corpus")

    output = ROOT / args.output
    existing = json.loads(output.read_text(encoding="utf-8"))
    records = existing[:12] if args.refresh else existing[: args.target]
    excluded = set(args.exclude)
    records = [row for row in records if row["id"] not in excluded]
    known = {row["id"] for row in records} | excluded
    groups: list[list[str]] = []
    for query in SEARCH_QUERIES:
        group = [arxiv_id for arxiv_id in search_ids(query) if arxiv_id not in known]
        groups.append(list(dict.fromkeys(group)))
        print(f"search={query!r} candidates={len(group)}", flush=True)
        time.sleep(args.delay)

    # Round-robin discovery prevents a single broad query from filling the
    # entire corpus before agent/tool-use candidates are considered.
    candidates: list[str] = []
    for position in range(max(map(len, groups), default=0)):
        for group in groups:
            if position < len(group) and group[position] not in candidates:
                candidates.append(group[position])

    for arxiv_id in candidates:
        if len(records) >= args.target:
            break
        try:
            record = metadata(arxiv_id)
        except Exception as exc:
            print(f"skip {arxiv_id}: {type(exc).__name__}", flush=True)
            continue
        records.append(record)
        known.add(arxiv_id)
        print(f"selected {len(records)}/{args.target}: {arxiv_id} {record['title']}", flush=True)
        time.sleep(args.delay)

    if len(records) != args.target:
        raise SystemExit(f"only validated {len(records)} papers; expected {args.target}")
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "papers": len(records), "path": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
