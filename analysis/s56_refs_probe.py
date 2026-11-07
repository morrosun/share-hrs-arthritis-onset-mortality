# -*- coding: utf-8 -*-
"""Probe CrossRef / Europe PMC reachability through the local proxy.

Uses urllib only (requests lacks certifi in this venv). Writes a small log so the
result is inspectable without relying on stdout.
"""
import io, json, os, sys, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out", "_refs_probe.txt")

PROXY = "http://127.0.0.1:7890"
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}),
    urllib.request.HTTPSHandler(),
)
opener.addheaders = [("User-Agent", "manuscript-ref-check/1.0 (mailto:morrosun@njmu.edu.cn)")]


def get(url, timeout=40):
    with opener.open(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


lines = []
q = "Regression models and life-tables Cox 1972"
url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
    {"query.bibliographic": q, "rows": 3, "select": "DOI,title,container-title,issued,volume,issue,page,author"})
try:
    d = get(url)
    for it in d["message"]["items"]:
        lines.append("CROSSREF OK | %s | %s | %s" % (
            it.get("DOI"), (it.get("title") or [""])[0][:90], (it.get("container-title") or [""])[0]))
except Exception as e:
    lines.append("CROSSREF FAIL: %r" % (e,))

url2 = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
        + urllib.parse.urlencode({"query": "TITLE:regression AND TITLE:life-tables AND AUTH:Cox",
                                  "format": "json", "pageSize": 3}))
try:
    d2 = get(url2)
    for it in d2.get("resultList", {}).get("result", []):
        lines.append("EPMC OK | %s | %s | %s" % (it.get("pmid"), it.get("title", "")[:90], it.get("journalTitle", "")))
except Exception as e:
    lines.append("EPMC FAIL: %r" % (e,))

io.open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("wrote", OUT, len(lines))
