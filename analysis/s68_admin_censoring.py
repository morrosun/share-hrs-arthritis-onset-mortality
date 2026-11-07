# -*- coding: utf-8 -*-
"""Sensitivity: administrative censoring at the death-ascertainment endpoint (2021).

Current main analysis censors respondents alive at their last interview at that
interview. If death ascertainment extends to 2021, respondents with no death
record can instead be followed to 2021. This script compares the two rules in
SHARE (prevalent) and in the HRS incident cohort.
"""
import io, os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyreadr
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "s68_admin_censoring.txt")
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
END = 2021
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def fit(X, label, cols):
    m = CoxPHFitter()
    m.fit(X[cols], duration_col="exit", event_col="event", entry_col="entry")
    P("\n[%s] n=%d deaths=%d person-years=%.0f" % (
        label, len(X), int(X["event"].sum()), (X["exit"] - X["entry"]).sum()))
    out = []
    for k, lab in [("g74", "60-74 vs <60"), ("g75", ">=75 vs <60")]:
        if k not in m.summary.index:
            continue
        q = m.summary.loc[k]
        P("  %-14s %.2f (%.2f-%.2f)" % (lab, q["exp(coef)"], q["exp(coef lower 95%)"] if "exp(coef lower 95%)" in q else q["exp(coef) lower 95%"],
                                        q["exp(coef) upper 95%"]))
        out.append((lab, q["exp(coef)"], q["exp(coef) lower 95%"], q["exp(coef) upper 95%"]))
    return out


# ================= SHARE =================
s = pyreadr.read_r(BASE + r"\share.RData")["share"].copy()
s = s.rename(columns={s.columns[0]: "pid"})
s["yr"] = s["wave"].map(WY_S)
s["age"] = s["yr"] - s["rabyear"]
s = s.sort_values(["pid", "wave"])

rows = []
for pid, x in s.groupby("pid"):
    dg = x["radiagarthr"].dropna()
    dg = dg[dg >= 5]
    if dg.empty:
        continue
    rows.append(dict(pid=pid, onset=float(dg.iloc[0]), entry=x["age"].iloc[0], last=x["age"].iloc[-1],
                     dead=x["radyear"].max(), byear=x["rabyear"].iloc[0],
                     sex=x["ragender"].iloc[0], edu=x["raeducl"].iloc[0], smoke=x["smoken"].iloc[0]))
D = pd.DataFrame(rows)
D = D[D["byear"].notna() & (D["entry"] >= 40)].copy()
D["event"] = D["dead"].notna().astype(int)
D["death_age"] = D["dead"] - D["byear"]
D["exit_last"] = np.where(D["event"] == 1, D["death_age"], D["last"])
D["exit_2021"] = np.where(D["event"] == 1, D["death_age"], END - D["byear"])
for c in ["g74", "g75"]:
    D[c] = 0.0
D["g74"] = ((D["onset"] >= 60) & (D["onset"] < 75)).astype(float)
D["g75"] = (D["onset"] >= 75).astype(float)
D["male"] = (D["sex"] == 1).astype(float)
D["smoke_ever"] = (D["smoke"] >= 1).astype(float)
D = D.dropna(subset=["edu", "smoke_ever"])
cols = ["exit", "entry", "event", "g74", "g75", "male", "edu", "smoke_ever"]

P("=" * 80)
P("# SHARE prevalent cohort")
P("=" * 80)
A = D[D["exit_last"] > D["entry"]].copy()
A["exit"] = A["exit_last"]
fit(A, "A. censor at last interview (current main analysis)", cols)

B = D[D["exit_2021"] > D["entry"]].copy()
B["exit"] = B["exit_2021"]
P("\n  respondents added by 2021 censoring: %d, of whom deaths %d" % (len(B) - len(A), 0))
fit(B, "B. censor at 2021 (death ascertainment endpoint)", cols)

# ================= HRS incident =================
h = pyreadr.read_r(BASE + r"\hrs.RData")["hrs"].copy()
h = h.rename(columns={h.columns[0]: "pid"})
h["yr"] = h["wave"].map(WY_H)
h["age"] = h["yr"] - h["rabyear"]
h = h.dropna(subset=["age", "arthre"]).sort_values(["pid", "wave"])

rows = []
for pid, x in h.groupby("pid"):
    fp = x[x["arthre"] == 1]
    if fp.empty:
        continue
    ip = x.index.get_loc(fp.index[0])
    neg = x.iloc[:ip]
    neg = neg[neg["arthre"] == 0]
    if neg.empty:
        continue
    rows.append(dict(pid=pid, pos=fp["age"].iloc[0], last=x["age"].iloc[-1],
                     dead=x["radyear"].max(), byear=x["rabyear"].iloc[0],
                     sex=x["ragender"].iloc[0], edu=x["raeducl"].iloc[0], smoke=x["smoken"].iloc[0],
                     lastneg=neg["age"].iloc[-1]))
I = pd.DataFrame(rows)
I["event"] = I["dead"].notna().astype(int)
I["death_age"] = I["dead"] - I["byear"]
I["mid"] = I["lastneg"] / 2 + I["pos"] / 2
I["g74"] = ((I["mid"] >= 60) & (I["mid"] < 75)).astype(float)
I["g75"] = (I["mid"] >= 75).astype(float)
I["male"] = (I["sex"] == 1).astype(float)
I["smoke_ever"] = (I["smoke"] >= 1).astype(float)
I = I.dropna(subset=["edu", "smoke_ever", "g74", "g75"])
I = I[(I["mid"] >= 40)]
I["entry"] = I["pos"]
I["exit_last"] = np.where(I["event"] == 1, I["death_age"], I["last"])
I["exit_2021"] = np.where(I["event"] == 1, I["death_age"], END - I["byear"])
# same-year deaths retained with 0.5 years under both rules
same = (I["event"] == 1) & (I["exit_last"] <= I["entry"])
I.loc[same, "exit_last"] = I.loc[same, "entry"] + 0.5
same2 = (I["event"] == 1) & (I["exit_2021"] <= I["entry"])
I.loc[same2, "exit_2021"] = I.loc[same2, "entry"] + 0.5

P("\n" + "=" * 80)
P("# HRS incident cohort")
P("=" * 80)
A2 = I[I["exit_last"] > I["entry"]].copy()
A2["exit"] = A2["exit_last"]
fit(A2, "A. censor at last interview (current main analysis)", cols)

B2 = I[I["exit_2021"] > I["entry"]].copy()
B2["exit"] = B2["exit_2021"]
P("\n  respondents added by 2021 censoring: %d (all alive at last interview)" % (len(B2) - len(A2)))
fit(B2, "B. censor at 2021 (death ascertainment endpoint)", cols)

io.open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("\nwrote", OUT)
