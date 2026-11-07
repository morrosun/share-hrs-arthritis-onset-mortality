# -*- coding: utf-8 -*-
"""Third follow-up rule: exclude every respondent with only one interview, dead or alive.

Rule A (current main): censor at last interview, but keep respondents whose only
interview was entry and who died later (their death supplies follow-up). This
conditions the risk set on death.
Rule B: censor all survivors at 2021 (assumes complete death ascertainment).
Rule C: keep only respondents with at least two interviews, so that no one enters
the risk set because they died.
"""
import io, os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyreadr
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "s69_followup_rules.txt")
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
END = 2021
COLS = ["exit", "entry", "event", "g74", "g75", "male", "edu", "smoke_ever"]
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def fit(X, label):
    X = X[X["exit"] <= 110].copy()
    m = CoxPHFitter()
    m.fit(X[COLS], duration_col="exit", event_col="event", entry_col="entry")
    P("\n[%s] n=%d deaths=%d person-years=%.0f" % (
        label, len(X), int(X["event"].sum()), (X["exit"] - X["entry"]).sum()))
    for k, lab in [("g74", "60-74 vs <60"), ("g75", ">=75 vs <60")]:
        q = m.summary.loc[k]
        P("  %-14s %.2f (%.2f-%.2f)" % (lab, q["exp(coef)"], q["exp(coef) lower 95%"], q["exp(coef) upper 95%"]))


# ---------------- SHARE ----------------
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
    rows.append(dict(pid=pid, onset=float(dg.iloc[0]), entry=x["age"].iloc[0], last=x["age"].iloc[-1], w0=x["wave"].min(),
                     nwave=len(x), dead=x["radyear"].max(), byear=x["rabyear"].iloc[0],
                     sex=x["ragender"].iloc[0], edu=x["raeducl"].iloc[0], smoke=x["smoken"].iloc[0]))
D = pd.DataFrame(rows)
D = D[D["byear"].notna() & (D["entry"] >= 40) & (D["w0"] != 8)].copy()
D["event"] = D["dead"].notna().astype(int)
D["death_age"] = D["dead"] - D["byear"]
D["grp"] = pd.cut(D["onset"], [40, 59, 74, 120], labels=["<60", "60-74", ">=75"])
D["g74"] = (D["grp"] == "60-74").astype(float)
D["g75"] = (D["grp"] == ">=75").astype(float)
D["male"] = (D["sex"] == 1).astype(float)
D["smoke_ever"] = (D["smoke"] >= 1).astype(float)
D = D.dropna(subset=["edu", "smoke_ever"])

P("=" * 82)
P("# SHARE prevalent cohort")
P("=" * 82)
P("all with usable onset and follow-up from age 40: %d" % len(D))
P("  with only one interview (last <= entry): %d, of whom died later %d, same-year deaths %d, alive %d" % (
    (D["last"] <= D["entry"]).sum(),
    ((D["last"] <= D["entry"]) & (D["event"] == 1) & (D["death_age"] > D["entry"])).sum(),
    ((D["last"] <= D["entry"]) & (D["event"] == 1) & (np.abs(D["death_age"] - D["entry"]) <= 1)).sum(),
    ((D["last"] <= D["entry"]) & (D["event"] == 0)).sum()))

A = D.copy()
A["exit"] = np.where(A["event"] == 1, A["death_age"], A["last"])
same = (A["event"] == 1) & (A["exit"] <= A["entry"])
A.loc[same, "exit"] = A.loc[same, "entry"] + 0.5
A = A[A["exit"] > A["entry"]]
fit(A, "A. keep later deaths with one interview (current main)")

B = D.copy()
B["exit"] = np.where(B["event"] == 1, B["death_age"], 2021 - B["byear"])
B.loc[(B["event"] == 1) & (B["exit"] <= B["entry"]), "exit"] = B.loc[(B["event"] == 1) & (B["exit"] <= B["entry"]), "entry"] + 0.5
B = B[B["exit"] > B["entry"]]
fit(B, "B. all survivors assumed alive to 2021")

C = D[D["last"] > D["entry"] + 1e-9].copy()
C["exit"] = np.where(C["event"] == 1, C["death_age"], C["last"])
fit(C, "C. require at least two interviews")

# ---------------- HRS incident ----------------
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
    rows.append(dict(pid=pid, pos=fp["age"].iloc[0], last=x["age"].iloc[-1], nwave=len(x),
                     dead=x["radyear"].max(), byear=x["rabyear"].iloc[0],
                     sex=x["ragender"].iloc[0], edu=x["raeducl"].iloc[0], smoke=x["smoken"].iloc[0],
                     lastneg=neg["age"].iloc[-1]))
I = pd.DataFrame(rows)
I["event"] = I["dead"].notna().astype(int)
I["death_age"] = I["dead"] - I["byear"]
I["mid"] = I["lastneg"] / 2 + I["pos"] / 2
I["grp"] = pd.cut(I["mid"], [40, 59, 74, 120], labels=["<60", "60-74", ">=75"])
I["g74"] = (I["grp"] == "60-74").astype(float)
I["g75"] = (I["grp"] == ">=75").astype(float)
I["male"] = (I["sex"] == 1).astype(float)
I["smoke_ever"] = (I["smoke"] >= 1).astype(float)
I = I.dropna(subset=["edu", "smoke_ever", "g74", "g75"])
I = I[I["mid"] >= 40]
I["entry"] = I["pos"]

P("\n" + "=" * 82)
P("# HRS incident cohort")
P("=" * 82)
P("eligible: %d; with only one interview after report: %d (died later %d, same-year %d, alive %d)" % (
    len(I), (I["last"] <= I["entry"]).sum(),
    ((I["last"] <= I["entry"]) & (I["event"] == 1) & (I["death_age"] > I["entry"])).sum(),
    ((I["last"] <= I["entry"]) & (I["event"] == 1) & (np.abs(I["death_age"] - I["entry"]) <= 1)).sum(),
    ((I["last"] <= I["entry"]) & (I["event"] == 0)).sum()))

A2 = I.copy()
A2["exit"] = np.where(A2["event"] == 1, A2["death_age"], A2["last"])
same2 = (A2["event"] == 1) & (A2["exit"] <= A2["entry"])
A2.loc[same2, "exit"] = A2.loc[same2, "entry"] + 0.5
A2 = A2[A2["exit"] > A2["entry"]]
fit(A2, "A. keep later and same-year deaths with one interview (current main)")

C2 = I[I["last"] > I["entry"] + 1e-9].copy()
C2["exit"] = np.where(C2["event"] == 1, C2["death_age"], C2["last"])
fit(C2, "C. require at least two interviews")

io.open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("\nwrote", OUT)
