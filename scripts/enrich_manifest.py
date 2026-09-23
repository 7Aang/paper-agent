"""Fetch public arXiv metadata for manifest entries without downloading PDFs."""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_agent.network import request_bytes
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def main():
    path = ROOT / "data_manifest.json"
    existing = json.loads(path.read_text(encoding="utf-8"))
    ids = [row.get("arxiv_id", row["id"]) for row in existing]
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    root = ET.fromstring(request_bytes(url, limit=4 * 1024 * 1024))
    metadata = {}
    for entry in root.findall("a:entry", NS):
        source_id = entry.findtext("a:id", "", NS).split("/abs/")[-1].split("v")[0]
        metadata[source_id] = {
            "title": " ".join(entry.findtext("a:title", "", NS).split()),
            "authors": [node.findtext("a:name", "", NS) for node in entry.findall("a:author", NS)],
            "year": int(entry.findtext("a:published", "0000", NS)[:4]),
            "doi": entry.findtext("arxiv:doi", None, NS),
            "source_url": "https://arxiv.org/abs/" + source_id,
            "pdf_url": "https://arxiv.org/pdf/" + source_id,
            "arxiv_id": source_id,
        }
    enriched = []
    for row in existing:
        source_id = row.get("arxiv_id", row["id"]).split("v")[0]
        if source_id not in metadata:
            raise RuntimeError(f"metadata missing for {source_id}")
        enriched.append({"id": row["id"], **metadata[source_id]})
    path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "updated", "records": len(enriched)}))
    time.sleep(0.1)


if __name__ == "__main__":
    main()
