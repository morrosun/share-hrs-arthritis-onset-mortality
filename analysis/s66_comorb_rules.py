# -*- coding: utf-8 -*-
"""Re-estimate the 70-79 year comorbidity models under two missing-data rules.

Rule A (main): comorbidity = sum of the six conditions with missing treated as absent.
Rule B (sensitivity): complete case, only person-waves with all six conditions reported.
"""
import io, os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUTDIR = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "s66_comorb_rules.txt")
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
COM = ["hibpe", "diabe", "hearte", "stroke", "cancre", "lunge"]
BINS = [45, 60, 70, 80, 90, 106]
BLAB = ["45-59", "60-69", "70-79", "80-89", "90+"]
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def gof(o):
    return "<60" if o < 60 else ("60-74" if o < 75 else ">=75")


# ---- HRS ----
h = pyreadr.read_r(BASE + r"\hrs.RData")["hrs"].copy()
h = h.rename(columns={h.columns[0]: "pid"})
h["yr"] = h["wave"].map(WY_H)
h["age"] = h["yr"] - h["rabyear"]
h["n_ok"] = h[COM].notna().sum(axis=1)
h["comorb"] = h[COM].fillna(0).sum(axis=1)
h = h.dropna(subset=["age", "arthre"]).sort_values(["pid", "wave"])

pers = {}
for pid, x in h.groupby("pid"):
    fp = x[x["arthre"] == 1]
    if fp.empty:
        pers[pid] = ("never", np.nan, np.nan)
        continue
    ip = x.index.get_loc(fp.index[0])
    neg = x.iloc[:ip]
    neg = neg[neg["arthre"] == 0]
    if neg.empty:
        pers[pid] = ("prevalent", np.nan, np.nan)
        continue
    pers[pid] = ("on", (neg["age"].iloc[-1] + fp["age"].iloc[0]) / 2, fp["age"].iloc[0])


def state_of(row):
    pat, mid, pos = pers[row["pid"]]
    if pat == "never":
        return "never"
    if pat == "prevalent":
        return "prevalent"
    return "never" if row["age"] < pos else "on_" + gof(mid)


h["state"] = h.apply(state_of, axis=1)
hh = h[(h["age"] >= 45) & (h["age"] <= 105)].copy()
hh["male"] = (hh["ragender"] == 1).astype(float)
hh["edu_c"] = pd.Categorical(hh["raeducl"], [1, 2, 3], ordered=True)
hh = hh.dropna(subset=["age", "male", "edu_c"])
hh["state"] = pd.Categorical(hh["state"], ["never", "prevalent", "on_<60", "on_60-74", "on_>=75"])

# ---- SHARE ----
s = pyreadr.read_r(BASE + r"\share.RData")["share"].copy()
s = s.rename(columns={s.columns[0]: "pid"})
s["yr"] = s["wave"].map(WY_S)
s["age"] = s["yr"] - s["rabyear"]
s["n_ok"] = s[COM].notna().sum(axis=1)
s["comorb"] = s[COM].fillna(0).sum(axis=1)
sd = {}
for pid, x in s.groupby("pid"):
    dg = x["radiagarthr"].dropna()
    dg = dg[dg >= 5]
    sd[pid] = ("on_" + gof(dg.iloc[0])) if not dg.empty else "never"
ss = s.copy()
ss["state"] = ss["pid"].map(sd).fillna("never")
ss = ss[(ss["age"] >= 45) & (ss["age"] <= 105)].copy()
ss["male"] = (ss["ragender"] == 1).astype(float)
ss["edu_c"] = pd.Categorical(ss["raeducl"], [1, 2, 3], ordered=True)
ss = ss.dropna(subset=["age", "male", "edu_c"])
ss["state"] = pd.Categorical(ss["state"], ["never", "on_<60", "on_60-74", "on_>=75"])

STATES = {"HRS": ["prevalent", "on_<60", "on_60-74", "on_>=75"], "SHARE": ["on_<60", "on_60-74", "on_>=75"]}
res = []


def run(data, lib, rule, strict):
    d = data[(data["age"] >= 70) & (data["age"] < 80)].copy()
    if strict:
        d = d[d["n_ok"] == 6]
    d = d.dropna(subset=["comorb"])
    m = smf.gee("comorb ~ C(state) + C(edu_c) + male + age", groups="pid", data=d,
                family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence()).fit()
    P("\n--- %s | %s | waves=%d persons=%d ---" % (lib, rule, len(d), d["pid"].nunique()))
    for stt in STATES[lib]:
        k = "C(state)[T.%s]" % stt
        est = np.exp(m.params[k])
        lo = np.exp(m.params[k] - 1.96 * m.bse[k])
        hi = np.exp(m.params[k] + 1.96 * m.bse[k])
        P("  %-11s %.3f (%.3f-%.3f)" % (stt, est, lo, hi))
        res.append(dict(lib=lib, rule=rule, state=stt, est=est, lo=lo, hi=hi, waves=len(d)))
    # direct contrasts among strata
    for a, b in [("on_60-74", "on_<60"), ("on_>=75", "on_<60"), ("on_>=75", "on_60-74")]:
        ka, kb = "C(state)[T.%s]" % a, "C(state)[T.%s]" % b
        bb = m.params[ka] - m.params[kb]
        V = m.cov_params()
        se = np.sqrt(V.loc[ka, ka] + V.loc[kb, kb] - 2 * V.loc[ka, kb])
        P("  diff %-18s %.3f (%.3f-%.3f)" % (a.replace("on_", "") + " vs " + b.replace("on_", ""),
                                             np.exp(bb), np.exp(bb - 1.96 * se), np.exp(bb + 1.96 * se)))


P("=" * 78)
P("# Comorbidity at 70-79 years: main rule (missing = absent) vs complete case")
P("=" * 78)
P("HRS person-waves with all six reported: %.1f%%; SHARE: %.1f%%" % (
    100 * (hh[hh["age"].between(70, 79)]["n_ok"] == 6).mean(),
    100 * (ss[ss["age"].between(70, 79)]["n_ok"] == 6).mean()))
run(hh, "HRS", "A: missing = absent (main)", False)
run(hh, "HRS", "B: complete case (all six)", True)
run(ss, "SHARE", "A: missing = absent (main)", False)
run(ss, "SHARE", "B: complete case (all six)", True)

pd.DataFrame(res).to_csv(os.path.join(os.path.dirname(OUT), "tableS8_comorb_rules.csv"),
                         index=False, encoding="utf-8-sig")
io.open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("\nwrote", OUT)
