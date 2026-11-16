# -*- coding: utf-8 -*-
"""Audit the 8,729 SHARE respondents excluded for having no follow-up interval.

Question raised in review: SHARE defines follow-up by interviews, whereas HRS
defines it by death or interview. If respondents whose only interview was their
entry interview but who have a death record are excluded, the prevalent cohort is
conditioned on surviving to a second interview.

This script:
1. rebuilds the 27,264 respondents with a usable diagnosis age;
2. classifies the 8,729 excluded by whether a death was recorded and when;
3. refits the main model under a unified "death or interview" rule.
"""
import io, os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyreadr
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUTDIR = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "s65_share_followup_rule.txt")
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}

lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


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
    rows.append(dict(
        pid=pid, onset=float(dg.iloc[0]),
        entry=x["age"].iloc[0], entry_wave=x["wave"].iloc[0],
        last=x["age"].iloc[-1], last_wave=x["wave"].iloc[-1],
        n_wave=len(x),
        dead=x["radyear"].max(), byear=x["rabyear"].iloc[0],
        sex=x["ragender"].iloc[0], edu=x["raeducl"].iloc[0], smoke=x["smoken"].iloc[0],
    ))
D = pd.DataFrame(rows)
D["event"] = D["dead"].notna().astype(int)
D["death_age"] = D["dead"] - D["byear"]
D["exit_interview"] = np.where(D["event"] == 1, D["death_age"], D["last"])
D["entry"] = D["entry"]

P("=" * 80)
P("# SHARE respondents with a usable diagnosis age")
P("=" * 80)
P("total: %d" % len(D))
D = D[D["byear"].notna()]
P("with birth year: %d" % len(D))
D = D[D["entry"] >= 40]
P("entry age >= 40: %d" % len(D))

no_int = D["last"] <= D["entry"] + 1e-9
P("\nlast interview <= entry interview (no interview-based follow-up): %d" % no_int.sum())
E = D[no_int]
P("  with a death record: %d" % E["event"].sum())
P("  death year AFTER entry year (death provides follow-up): %d" % (E["event"].eq(1) & (E["death_age"] > E["entry"] + 1e-9)).sum())
P("  death in the same year as entry: %d" % (E["event"].eq(1) & (abs(E["death_age"] - E["entry"]) <= 1e-9)).sum())
P("  death year BEFORE entry year (anomalous): %d" % (E["event"].eq(1) & (E["death_age"] < E["entry"] - 1e-9)).sum())
P("  alive at the only interview: %d" % E["event"].eq(0).sum())
P("\n  by entry wave:")
tab = E.groupby("entry_wave").agg(n=("pid", "size"), deaths=("event", "sum"),
                                  death_after=("death_age", lambda z: 0))
tab["death_after"] = E[E["event"] == 1].groupby("entry_wave").apply(
    lambda z: (z["death_age"] > z["entry"] + 1e-9).sum())
P(tab.to_string())

# ---- main analysis cohort as currently defined ----
A = D[(D["exit_interview"] > D["entry"])].copy()
A["g74"] = ((A["onset"] >= 60) & (A["onset"] < 75)).astype(float)
A["g75"] = (A["onset"] >= 75).astype(float)
A["male"] = (A["sex"] == 1).astype(float)
A["smoke_ever"] = (A["smoke"] >= 1).astype(float)
A = A.dropna(subset=["edu", "smoke_ever"])
A["exit"] = A["exit_interview"]
P("\ncurrent prevalent cohort (interview-based follow-up): %d, deaths %d" % (len(A), int(A["event"].sum())))

# ---- unified rule: keep respondents whose only contact was entry but who died after it ----
B = D.copy()
keep_no_int_dead = (B["last"] <= B["entry"] + 1e-9) & (B["event"] == 1) & (B["death_age"] >= B["entry"] - 1e-9)
B = B[(B["exit_interview"] > B["entry"]) | keep_no_int_dead].copy()
B.loc[keep_no_int_dead & (B["death_age"] <= B["entry"] + 1e-9), "exit_interview"] = B.loc[
    keep_no_int_dead & (B["death_age"] <= B["entry"] + 1e-9), "entry"] + 0.5
B["g74"] = ((B["onset"] >= 60) & (B["onset"] < 75)).astype(float)
B["g75"] = (B["onset"] >= 75).astype(float)
B["male"] = (B["sex"] == 1).astype(float)
B["smoke_ever"] = (B["smoke"] >= 1).astype(float)
B = B.dropna(subset=["edu", "smoke_ever"])
B["exit"] = B["exit_interview"]
P("unified rule cohort: %d, deaths %d (+%d people, +%d deaths)" % (
    len(B), int(B["event"].sum()), len(B) - len(A), int(B["event"].sum() - A["event"].sum())))


def fit(X, label):
    m = CoxPHFitter()
    cols = ["exit", "entry", "event", "g74", "g75", "male", "edu", "smoke_ever"]
    m.fit(X[cols], duration_col="exit", event_col="event", entry_col="entry")
    P("\n[%s] n=%d deaths=%d" % (label, len(X), int(X["event"].sum())))
    for k, lab in [("g74", "60-74 vs <60"), ("g75", ">=75 vs <60")]:
        q = m.summary.loc[k]
        P("  %-14s %.2f (%.2f-%.2f)" % (lab, q["exp(coef)"], q["exp(coef) lower 95%"], q["exp(coef) upper 95%"]))
    return m


fit(A, "interview-based follow-up (current main analysis)")
fit(B, "unified rule: death or interview")

io.open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("\nwrote", OUT)
