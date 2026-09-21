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
        url = "https://arxiv.org/pdf/" + item["id"]
        try:
            if not target.exists():
                request = urllib.request.Request(url, headers={"User-Agent":"PaperAgent-Educational/0.1"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read(30 * 1024 * 1024 + 1)
                if not payload.startswith(b"%PDF") or len(payload) > 30 * 1024 * 1024:
                    raise ValueError("Invalid or oversized PDF")
                target.write_bytes(payload)
                time.sleep(3.1)
            results.append({**item,"url":url,"sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"bytes":target.stat().st_size})
            print(item["id"], "OK", target.stat().st_size, flush=True)
        except Exception as exc:
            results.append({**item,"url":url,"error":str(exc)})
            print(item["id"], type(exc).__name__, flush=True)
    (dest / "provenance.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    if any("error" in r for r in results):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
