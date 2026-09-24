"""Download public research PDFs serially with caching; no credentials needed."""
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    dest = ROOT / "data" / "corpus"
    dest.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT / "data_manifest.json").read_text())
    results = []
    for item in records:
        target = dest / (item["id"] + ".pdf")
        url = item.get("pdf_url","https://arxiv.org/pdf/" + item["id"])
        try:
            if not target.exists():
                request = urllib.request.Request(url, headers={"User-Agent":"PaperAgent-Educational/0.1"})
                started = time.monotonic()
                parts = []
                size = 0
                with urllib.request.urlopen(request, timeout=20) as response:
                    while True:
                        if time.monotonic() - started > 90:
                            raise TimeoutError("download exceeded 90-second total limit")
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        parts.append(chunk)
                        size += len(chunk)
                        if size > 30 * 1024 * 1024:
                            raise ValueError("Invalid or oversized PDF")
                payload = b"".join(parts)
                if not payload.startswith(b"%PDF") or len(payload) > 30 * 1024 * 1024:
                    raise ValueError("Invalid or oversized PDF")
                target.write_bytes(payload)
                time.sleep(3.1)
            results.append({**item,"url":url,"sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"bytes":target.stat().st_size})
            print(item["id"], "OK", target.stat().st_size, flush=True)
        except Exception as exc:
            results.append({**item,"url":url,"error":str(exc)})
            print(item["id"], type(exc).__name__, flush=True)
    payload = json.dumps(results, indent=2, ensure_ascii=False)
    (dest / "provenance.json").write_text(payload, encoding="utf-8")
    # Keep a tracked, compact provenance ledger beside the public manifest so a
    # reviewer can audit URLs, byte sizes and hashes without committing PDFs.
    (ROOT / "corpus_provenance.json").write_text(payload, encoding="utf-8")
    if any("error" in r for r in results):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
