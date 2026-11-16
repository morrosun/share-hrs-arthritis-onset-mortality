# -*- coding: utf-8 -*-
"""Targeted follow-up to s71: HRS sample reconciliation, death ascertainment,
and the 2021-censoring rule applied to the full cohort rather than a pre-filtered one."""
import os
import io, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyreadr
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUTDIR = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "s72_reconcile.txt")
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
END = 2021
L = []


def P(s=""):
    L.append(str(s))
    print(s, flush=True)


def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=["<60", "60-74", ">=75"])


def fit(X, label, cols=("exit", "entry", "event", "g74", "g75", "male", "edu", "smoke_ever")):
    X = X[(X["exit"] <= 110) & (X["exit"] > X["entry"])]
    m = CoxPHFitter()
    m.fit(X[list(cols)], duration_col="exit", event_col="event", entry_col="entry")
    P("  [%s] n=%d deaths=%d py=%.0f" % (label, len(X), int(X["event"].sum()),
                                        (X["exit"] - X["entry"]).sum()))
    for k, lab in [("g74", "60-74 vs <60"), ("g75", ">=75 vs <60")]:
        q = m.summary.loc[k]
        P("    %-13s %.3f (%.3f-%.3f)" % (lab, q["exp(coef)"], q["exp(coef) lower 95%"],
                                          q["exp(coef) upper 95%"]))
    return X


# ---------------- SHARE, unfiltered person file ----------------
s = pyreadr.read_r(BASE + r"\share.RData")["share"].copy()
s = s.rename(columns={s.columns[0]: "pid"})
s["yr"] = s["wave"].map(WY_S)
s["age"] = s["yr"] - s["rabyear"]
agg = s.groupby("pid").agg(entry=("age", "min"), last=("age", "max"),
                           sex=("ragender", "first"), byear=("rabyear", "first"),
                           edu=("raeducl", "first"), smoke=("smoken", "max"))
agg["onset"] = s.dropna(subset=["radiagarthr"]).groupby("pid")["radiagarthr"].first()
agg["dead"] = s.groupby("pid")["radyear"].max()
agg["last_yr"] = s.groupby("pid")["yr"].max()
D = agg[agg["onset"].notna()].copy()
D = D[(D["onset"] >= 5) & D["byear"].notna() & (D["entry"] >= 40)].copy()
D["event"] = D["dead"].notna().astype(int)
D["death_age"] = D["dead"] - D["byear"]
D["exit"] = np.where(D["event"] == 1, D["death_age"], D["last"])
D["grp"] = grp(D["onset"])
D["g74"] = (D["grp"] == "60-74").astype(float)
D["g75"] = (D["grp"] == ">=75").astype(float)
D["male"] = (D["sex"] == 1).astype(float)
D["smoke_ever"] = (D["smoke"] >= 1).astype(float)
D["edu"] = pd.to_numeric(D["edu"], errors="coerce")
D = D.dropna(subset=["edu"])

P("=" * 88)
P("# A. SHARE: everyone with a usable onset, before any follow-up filter")
P("=" * 88)
P("n=%d deaths=%d" % (len(D), D["event"].sum()))
one = D["last"] <= D["entry"]
P("single interview: %d (died later %d, same-year %d, no death record %d)" % (
    one.sum(), (one & (D["event"] == 1) & (D["death_age"] > D["entry"])).sum(),
    (one & (D["event"] == 1) & (D["death_age"] <= D["entry"] + 1e-9)).sum(),
    (one & (D["event"] == 0)).sum()))

P("\n# B. SHARE death ascertainment: by year of last interview, all persons (no follow-up filter)")
P("  year   n     deaths  %dead   median age")
for yr in sorted(D["last_yr"].dropna().unique()):
    q = D[D["last_yr"] == yr]
    if len(q) < 100:
        continue
    P("  %4d %6d %7d %6.1f%% %8.0f" % (yr, len(q), q["event"].sum(), 100 * q["event"].mean(), q["last"].median()))
P("  (a person is 'dead' only if a death year is recorded; the remaining are censored at the last interview)")
P("  deaths occurring >1 y after the last interview: %d of %d (%.1f%%)" % (
    (D["event"] == 1).mul((D["dead"] - D["last_yr"]) > 1).sum(), D["event"].sum(),
    100 * ((D.loc[D["event"] == 1, "dead"] - D.loc[D["event"] == 1, "last_yr"]) > 1).mean()))

# ---------------- HRS, unfiltered person file ----------------
h = pyreadr.read_r(BASE + r"\hrs.RData")["hrs"].copy()
h = h.rename(columns={h.columns[0]: "pid"})
h["yr"] = h["wave"].map(WY_H)
h["age"] = h["yr"] - h["rabyear"]
hh = h.dropna(subset=["age", "arthre"]).sort_values(["pid", "wave"])
recs = []
for pid, a in hh.groupby("pid"):
    fp = a[a["arthre"] == 1]
    rec = dict(pid=pid, entry=a["age"].iloc[0], last=a["age"].iloc[-1], last_yr=a["yr"].max(),
               nwave=len(a), dead=a["radyear"].max(), byear=a["rabyear"].iloc[0],
               sex=a["ragender"].iloc[0], edu=a["raeducl"].iloc[0], smoke=a["smoken"].iloc[0])
    if fp.empty:
        rec.update(pattern="never", onset=np.nan, pos=np.nan)
    else:
        ip = a.index.get_loc(fp.index[0])
        neg = a.iloc[:ip]
        neg = neg[neg["arthre"] == 0]
        rec.update(pattern="prevalent" if neg.empty else "incident",
                   onset=((neg["age"].iloc[-1] + fp["age"].iloc[0]) / 2) if not neg.empty else np.nan,
                   pos=fp["age"].iloc[0])
    recs.append(rec)
H = pd.DataFrame(recs).set_index("pid")
H["event"] = H["dead"].notna().astype(int)
H["exit"] = np.where(H["event"] == 1, H["dead"] - H["byear"], H["last"])
H["ok_exit"] = H["exit"] > H["entry"]
H["ok_age"] = H["entry"] >= 40
H["edu_n"] = pd.to_numeric(H["edu"], errors="coerce")

P("\n" + "=" * 88)
P("# C. HRS: where 36,529 -> 34,537 comes from, by pattern")
P("=" * 88)
P("%-10s %6s %6s %6s %6s %6s %6s" % ("pattern", "all", "exit<=entry", "age<40", "edu miss", "kept", "deaths"))
for pat in ["never", "prevalent", "incident"]:
    q = H[H["pattern"] == pat]
    e1 = (~q["ok_exit"]).sum()
    e2 = (q["ok_exit"] & ~q["ok_age"]).sum()
    e3 = (q["ok_exit"] & q["ok_age"] & q["edu_n"].isna()).sum()
    kept = q[q["ok_exit"] & q["ok_age"] & q["edu_n"].notna()]
    P("%-10s %6d %9d %8d %8d %6d %6d" % (pat, len(q), e1, e2, e3, len(kept), kept["event"].sum()))
P("total kept %d deaths %d (published 34,537 / 13,056)" % (
    (H["ok_exit"] & H["ok_age"] & H["edu_n"].notna()).sum(),
    H.loc[H["ok_exit"] & H["ok_age"] & H["edu_n"].notna(), "event"].sum()))
P("kept WITHOUT the age>=40 filter: %d persons, %d deaths" % (
    (H["ok_exit"] & H["edu_n"].notna()).sum(), H.loc[H["ok_exit"] & H["edu_n"].notna(), "event"].sum()))
P("=> the published HRS mortality sample includes %d persons aged <40 at entry" % (
    (H["ok_exit"] & ~H["ok_age"]).sum()))

# incident analytic, published definition
I = H[(H["pattern"] == "incident") & (H["onset"] >= 40) & H["ok_exit"]].copy()
I["grp"] = grp(I["onset"])
I["g74"] = (I["grp"] == "60-74").astype(float)
I["g75"] = (I["grp"] == ">=75").astype(float)
I["male"] = (I["sex"] == 1).astype(float)
I["smoke_ever"] = (I["smoke"] >= 1).astype(float)
I["edu"] = pd.to_numeric(I["edu"], errors="coerce")
same = (I["event"] == 1) & (I["exit"] <= I["pos"])
I.loc[same, "exit"] = I.loc[same, "pos"] + 0.5
I["entry2"] = I["pos"]
I = I[I["exit"] > I["entry2"]].dropna(subset=["edu"])
P("\nincident analytic from 7,252: %d persons, %d deaths (published 6,359 / 1,931)" % (
    len(I), I["event"].sum()))

P("\n# D. HRS death ascertainment (pid index fixed), all incident-eligible persons")
J = H[H["pattern"] == "incident"].copy()
J["gap"] = J["dead"] - J["last_yr"]
P("deaths %d; gap<0 %d; gap=0 %d; 1-4 %d; >=5 %d; median %.1f" % (
    J["event"].sum(), (J["gap"] < 0).sum(), (J["gap"] == 0).sum(),
    J["gap"].between(1, 4).sum(), (J["gap"] >= 5).sum(), J["gap"].median()))
P("  year   n     deaths  %dead   median age")
for yr in sorted(J["last_yr"].dropna().unique()):
    q = J[J["last_yr"] == yr]
    if len(q) < 100:
        continue
    P("  %4d %6d %7d %6.1f%% %8.0f" % (yr, len(q), q["event"].sum(), 100 * q["event"].mean(), q["last"].median()))

P("\n" + "=" * 88)
P("# E. Follow-up rules applied to the FULL cohorts (not pre-filtered)")
P("=" * 88)
P("SHARE: n=%d eligible; HRS: n=%d incident eligible" % (len(D), len(J)))
P("same-year deaths retained at entry+0.5 in every rule; age time scale; exit capped at 110.")

# SHARE rules
P("\n--- SHARE")
D["exit_last"] = np.where(D["event"] == 1, D["death_age"], D["last"])
D["exit_2021"] = np.where(D["event"] == 1, D["death_age"], END - D["byear"])
for c in ["exit_last", "exit_2021"]:
    m = (D["event"] == 1) & (D[c] <= D["entry"])
    D.loc[m, c] = D.loc[m, "entry"] + 0.5
X = D.copy()
X["exit"] = X["exit_last"]
fit(X, "R1 last interview, same-year retained")
X2 = D[(D["last"] > D["entry"])].copy()
X2["exit"] = X2["exit_last"]
fit(X2, "R2 >=2 interviews")
X3 = D.copy()
X3["exit"] = X3["exit_2021"]
fit(X3, "R3 censor at 2021")

# HRS rules
P("\n--- HRS incident (from 7,240 eligible, before the exit>entry filter)")
J["exit_last"] = np.where(J["event"] == 1, J["dead"] - J["byear"], J["last"])
J["exit_2021"] = np.where(J["event"] == 1, J["dead"] - J["byear"], END - J["byear"])
for c in ["exit_last", "exit_2021"]:
    m = (J["event"] == 1) & (J[c] <= J["pos"])
    J.loc[m, c] = J.loc[m, "pos"] + 0.5
J["entry"] = J["pos"]
J["grp"] = grp(J["onset"])
J["g74"] = (J["grp"] == "60-74").astype(float)
J["g75"] = (J["grp"] == ">=75").astype(float)
J["male"] = (J["sex"] == 1).astype(float)
J["smoke_ever"] = (J["smoke"] >= 1).astype(float)
J["edu"] = pd.to_numeric(J["edu"], errors="coerce")
J = J.dropna(subset=["edu"])
X = J.copy(); X["exit"] = X["exit_last"]
fit(X, "R1 last interview, same-year retained (= published)")
X2 = J[J["last"] > J["pos"]].copy(); X2["exit"] = X2["exit_last"]
fit(X2, "R2 >=2 interviews")
X3 = J.copy(); X3["exit"] = X3["exit_2021"]
fit(X3, "R3 censor at 2021")

io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("\nwrote", OUT)
