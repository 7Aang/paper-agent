"""Constrained public-paper connector. No arbitrary URL downloads."""
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

try:
    import truststore
except ImportError:
    truststore = None


def request_bytes(url, *, data=None, headers=None, timeout=45, limit=30*1024*1024):
    request = urllib.request.Request(url,data=data,headers={"User-Agent":"PaperAgent/0.1",**(headers or {})})
    for attempt in range(3):
        try:
            context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT) if truststore else ssl.create_default_context()
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                raw = response.read(limit+1)
                if len(raw)>limit:
                    raise ValueError("Response too large")
                return raw
        except urllib.error.HTTPError as exc:
            if exc.code not in {429,500,502,503,504} or attempt==2:
                raise
            time.sleep(0.5*2**attempt)
        except (TimeoutError, urllib.error.URLError):
            if attempt==2:
                raise
            time.sleep(0.5*2**attempt)


def arxiv_search(query: str, limit: int = 5) -> list[dict]:
    if not query.strip() or len(query)>500:
        raise ValueError("Search query must contain 1-500 characters")
    url = "https://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":"all:"+query,"start":0,"max_results":max(1,min(limit,10))})
    root = ET.fromstring(request_bytes(url,limit=2*1024*1024))
    ns = {"a":"http://www.w3.org/2005/Atom"}
    result=[]
    for e in root.findall("a:entry",ns):
        pid = e.findtext("a:id","",ns).split("/abs/")[-1]
        if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?",pid):
            continue
        result.append({"id":pid,"title":" ".join(e.findtext("a:title","",ns).split()),
            "abstract":" ".join(e.findtext("a:summary","",ns).split()),
            "year":int(e.findtext("a:published","0000",ns)[:4]),"url":"https://arxiv.org/abs/"+pid})
    return result


def arxiv_pdf(paper_id: str) -> bytes:
    if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?",paper_id):
        raise ValueError("Expected a modern arXiv ID, such as 2005.11401")
    raw = request_bytes("https://arxiv.org/pdf/"+paper_id)
    if not raw.startswith(b"%PDF"):
        raise ValueError("arXiv did not return a PDF")
    return raw
