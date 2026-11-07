# -*- coding: utf-8 -*-
"""Resolve candidate references against CrossRef and keep the best match.

Input : hard-coded candidate list (key, bibliographic query, expected year, expected journal fragment)
Output: out/refs_candidates.json  (structured, for the manuscript builder)
        out/refs_candidates.txt   (human-readable audit, one line per candidate)

No proxy: direct HTTPS works in this environment (the local proxy refuses connections).
"""
import difflib, io, json, os, re, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTJ = os.path.join(ROOT, "out", "refs_candidates.json")
OUTT = os.path.join(ROOT, "out", "refs_candidates.txt")

UA = "manuscript-ref-check/1.0 (mailto:morrosun@njmu.edu.cn)"

# (key, query, expected year, journal fragment for sanity check)
CAND = [
    # --- burden and mortality of arthritis / osteoarthritis ---
    ("cross2014", "The global burden of hip and knee osteoarthritis estimates from the Global Burden of Disease 2010 study Cross", 2014, "Annals of the Rheumatic Diseases"),
    ("cui2020", "Global regional prevalence incidence and risk factors of knee osteoarthritis in population-based studies Cui", 2020, "EClinicalMedicine"),
    ("nuesch2011", "All cause and disease specific mortality in patients with knee or hip osteoarthritis population based cohort study Nuesch", 2011, "BMJ"),
    ("barbour2015", "Hip osteoarthritis and the risk of all-cause and disease-specific mortality in older women a population-based cohort study Barbour", 2015, "Arthritis & Rheumatology"),
    ("hawker2014", "All-cause mortality and serious cardiovascular events in people with hip and knee osteoarthritis a population based cohort study Hawker", 2014, "PLoS ONE"),
    ("turkiewicz2019", "Cause-specific mortality in osteoarthritis of peripheral joints Turkiewicz", 2019, "Osteoarthritis and Cartilage"),
    # --- rheumatoid arthritis, late onset ---
    ("kobak2018", "An autumn tale geriatric rheumatoid arthritis Kobak", 2018, "Therapeutic Advances in Musculoskeletal Disease"),
    ("schaardenburg1994", "Elderly-onset rheumatoid arthritis van Schaardenburg Breedveld", 1994, "Seminars in Arthritis and Rheumatism"),
    ("innala2014", "Age at onset determines severity and choice of treatment in early rheumatoid arthritis a prospective study Innala", 2014, "Arthritis Research & Therapy"),
    ("symmons2002", "Epidemiology of rheumatoid arthritis determinants of onset persistence and outcome Symmons", 2002, "Best Practice & Research Clinical Rheumatology"),
    # --- cohorts ---
    ("share2013", "Data resource profile the Survey of Health Ageing and Retirement in Europe SHARE Borsch-Supan", 2013, "International Journal of Epidemiology"),
    ("hrs2014", "Cohort profile the Health and Retirement Study Sonnega", 2014, "International Journal of Epidemiology"),
    ("elsa2013", "Cohort profile the English Longitudinal Study of Ageing Steptoe", 2013, "International Journal of Epidemiology"),
    ("charls2014", "Cohort profile the China Health and Retirement Longitudinal Study CHARLS Zhao", 2014, "International Journal of Epidemiology"),
    # --- methods: time scale, left truncation, cohort assembly ---
    ("cox1972", "Regression models and life-tables Cox", 1972, "Journal of the Royal Statistical Society"),
    ("korn1997", "Time-to-event analysis of longitudinal follow-up of a survey choice of the time-scale Korn Graubard Midthune", 1997, "American Journal of Epidemiology"),
    ("lamarca1998", "Left-truncated data with age as time scale an alternative for survival analysis in the elderly population Lamarca", 1998, "Journals of Gerontology"),
    ("thiebaut2004", "Choice of time-scale in Cox's model analysis of epidemiologic cohort data a simulation study Thiebaut Benichou", 2004, "Statistics in Medicine"),
    ("suissa2008", "Immortal time bias in pharmacoepidemiology Suissa", 2008, "American Journal of Epidemiology"),
    ("hernan2016", "Using big data to emulate a target trial when a randomized trial is not available Hernan Robins", 2016, "American Journal of Epidemiology"),
    ("ray2003", "Evaluating medication effects outside of clinical trials new-user designs Ray", 2003, "American Journal of Epidemiology"),
    # --- sensitivity analysis and multiplicity ---
    ("vanderweele2017", "Sensitivity analysis in observational research introducing the E-value VanderWeele Ding", 2017, "Annals of Internal Medicine"),
    ("holm1979", "A simple sequentially rejective multiple test procedure Holm", 1979, "Scandinavian Journal of Statistics"),
    ("benjamini1995", "Controlling the false discovery rate a practical and powerful approach to multiple testing Benjamini Hochberg", 1995, "Journal of the Royal Statistical Society"),
    ("strobe2007", "The Strengthening the Reporting of Observational Studies in Epidemiology STROBE statement guidelines for reporting observational studies von Elm", 2007, "The Lancet"),
    # --- measurement of function and comorbidity ---
    ("katz1963", "Studies of illness in the aged the index of ADL a standardized measure of biological and psychosocial function Katz", 1963, "JAMA"),
    ("lawton1969", "Assessment of older people self-maintaining and instrumental activities of daily living Lawton Brody", 1969, "The Gerontologist"),
    ("guccione1994", "The effects of specific medical conditions on the functional limitations of elders in the Framingham Study Guccione", 1994, "American Journal of Public Health"),
    ("verbrugge1994", "The disablement process Verbrugge Jette", 1994, "Social Science & Medicine"),
    # --- proportional hazards diagnostics ---
    ("schoenfeld1982", "Partial residuals for the proportional hazards regression model Schoenfeld", 1982, "Biometrika"),
    ("grambsch1994", "Proportional hazards tests and diagnostics based on weighted residuals Grambsch Therneau", 1994, "Biometrika"),
    # --- self-report validity and inflammation ---
    ("kriegsman1996", "Self-reports and general practitioner information on the presence of chronic diseases in community dwelling elderly Kriegsman", 1996, "Journal of Clinical Epidemiology"),
    ("gabay1999", "Acute-phase proteins and other systemic responses to inflammation Gabay Kushner", 1999, "New England Journal of Medicine"),
    ("erfc2010", "C-reactive protein concentration and risk of coronary heart disease stroke and mortality an individual participant meta-analysis", 2010, "The Lancet"),
    # --- ageing, frailty and comorbidity ---
    ("barnett2012", "Epidemiology of multimorbidity and implications for health care research and medical education a cross-sectional study Barnett", 2012, "The Lancet"),
    ("fried2001", "Frailty in older adults evidence for a phenotype Fried", 2001, "Journals of Gerontology"),
]

URL = "https://api.crossref.org/works"


def fetch(query, rows=5):
    params = {"query.bibliographic": query, "rows": rows,
              "select": "DOI,title,container-title,issued,volume,issue,page,author,type",
              "mailto": "morrosun@njmu.edu.cn"}
    url = URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode("utf-8", "replace"))["message"]["items"]


def year_of(it):
    try:
        return it["issued"]["date-parts"][0][0]
    except Exception:
        return None


def initials(given):
    if not given:
        return ""
    parts = re.split(r"[\s\-\.]+", given.strip())
    return "".join(p[0] for p in parts if p)


def fmt_authors(auths):
    names = []
    for a in auths or []:
        fam = (a.get("family") or "").strip()
        if not fam:
            continue
        names.append(fam + " " + initials(a.get("given")))
    if not names:
        return ""
    if len(names) > 6:
        return ", ".join(names[:6]) + ", et al"
    return ", ".join(names)


recs, lines = [], []
for key, q, yr, jr in CAND:
    try:
        items = fetch(q)
    except Exception as e:
        lines.append("%-18s FETCH-FAIL %r" % (key, e))
        continue
    scored = []
    for it in items:
        title = (it.get("title") or [""])[0]
        sim = difflib.SequenceMatcher(None, re.sub(r"\W+", " ", title.lower()),
                                      re.sub(r"\W+", " ", q.lower())).ratio()
        y = year_of(it)
        dy = abs((y or 0) - yr)
        score = sim - 0.06 * min(dy, 5)
        scored.append((score, sim, dy, it))
    scored.sort(key=lambda x: -x[0])
    score, sim, dy, it = scored[0]
    title = (it.get("title") or [""])[0]
    journal = (it.get("container-title") or [""])[0]
    rec = {
        "key": key, "doi": it.get("DOI"), "title": title, "journal": journal,
        "year": year_of(it), "volume": it.get("volume"), "issue": it.get("issue"),
        "page": it.get("page"), "authors": fmt_authors(it.get("author")),
        "sim": round(sim, 3), "dyear": dy,
        "journal_ok": jr.lower()[:12] in journal.lower(),
    }
    recs.append(rec)
    flag = "OK " if (sim >= 0.90 and dy <= 1 and rec["journal_ok"]) else "CHECK"
    lines.append("%-4s %-18s sim=%.2f dy=%d | %s | %s | %s %s;%s:%s | %s" % (
        flag, key, sim, dy, rec["doi"], journal, rec["year"],
        rec["volume"] or "-", rec["issue"] or "-", rec["page"] or "-", title[:80]))

json.dump(recs, io.open(OUTJ, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
io.open(OUTT, "w", encoding="utf-8").write("\n".join(lines))
print("candidates", len(CAND), "resolved", len(recs))
print("OK", sum(1 for l in lines if l.startswith("OK")), "CHECK", sum(1 for l in lines if l.startswith("CHECK")))
