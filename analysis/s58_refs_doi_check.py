# -*- coding: utf-8 -*-
"""Resolve specific DOIs for references whose title search matched a derivative record."""
import io, json, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out", "refs_doi_check.txt")

UA = "manuscript-ref-check/1.0 (mailto:morrosun@njmu.edu.cn)"
DOIS = [
    ("hawker2014_plos", "10.1371/journal.pone.0091286"),
    ("strobe2007_lancet", "10.1016/S0140-6736(07)61602-X"),
    ("fried2001_jgerona", "10.1093/gerona/56.3.M146"),
    ("hunter2019", "10.1016/S0140-6736(19)30417-9"),
    ("bijlsma2011", "10.1016/S0140-6736(11)60243-2"),
    ("vos2012", "10.1016/S0140-6736(12)61729-2"),
]


def initials(given):
    if not given:
        return ""
    return "".join(p[0] for p in __import__("re").split(r"[\s\-\.]+", given.strip()) if p)


lines = []
for key, doi in DOIS:
    try:
        req = urllib.request.Request("https://api.crossref.org/works/" + doi,
                                     headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=40) as r:
            m = json.loads(r.read().decode("utf-8", "replace"))["message"]
        auths = []
        for a in m.get("author", [])[:6]:
            auths.append((a.get("family") or "") + " " + initials(a.get("given")))
        if len(m.get("author", [])) > 6:
            auths.append("et al")
        lines.append("%-18s %s | %s | %s %s;%s:%s | %s" % (
            key, m.get("DOI"), (m.get("container-title") or [""])[0],
            m["issued"]["date-parts"][0][0], m.get("volume"), m.get("issue"),
            m.get("page"), ", ".join(auths)))
        lines.append("    TITLE: " + (m.get("title") or [""])[0])
    except Exception as e:
        lines.append("%-18s FAIL %r" % (key, e))

io.open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("wrote", OUT)
