# -*- coding: utf-8 -*-
"""Small checks: smoking counts and definition, the 881 one-interview HRS respondents,
and the upper age range of the HRS model."""
import os
import io, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import pyreadr

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUTDIR = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "s73_smoke_age.txt")
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
L = []
def P(s=""):
    L.append(str(s)); print(s, flush=True)

s = pyreadr.read_r(BASE + r"\share.RData")["share"].copy()
s = s.rename(columns={s.columns[0]: "pid"})
P("SHARE: rows with smoken missing = %d of %d (%.2f%%)" % (
    s["smoken"].isna().sum(), len(s), 100 * s["smoken"].isna().mean()))
P("SHARE: persons with all smoken missing = %d" % (
    s.groupby("pid")["smoken"].apply(lambda x: x.isna().all()).sum()))
ever_arth = s[s["radiagarthr"].notna()]["pid"].nunique()
P("SHARE: persons ever reporting arthritis = %d" % ever_arth)
q = s[s["radiagarthr"].notna()].groupby("pid")["smoken"].apply(lambda x: x.notna().any())
P("  of these, with no smoking value in any wave: %d" % (~q).sum())

h = pyreadr.read_r(BASE + r"\hrs.RData")["hrs"].copy()
h = h.rename(columns={h.columns[0]: "pid"})
h["yr"] = h["wave"].map(WY_H)
h["age"] = h["yr"] - h["rabyear"]
hh = h.dropna(subset=["age", "arthre"]).sort_values(["pid", "wave"])
P("\nHRS: persons = %d; persons with no smoken in any wave = %d" % (
    hh["pid"].nunique(), hh.groupby("pid")["smoken"].apply(lambda x: x.isna().all()).sum()))

# incident cohort, first interview after positive report
rows = []
for pid, a in hh.groupby("pid"):
    fp = a[a["arthre"] == 1]
    if fp.empty:
        continue
    ip = a.index.get_loc(fp.index[0])
    neg = a.iloc[:ip]
    neg = neg[neg["arthre"] == 0]
    if neg.empty:
        continue
    rows.append(dict(pid=pid, pos=fp["age"].iloc[0], last=a["age"].iloc[-1], last_yr=a["yr"].max(),
                     dead=a["radyear"].max(), byear=a["rabyear"].iloc[0],
                     smoke_first=a["smoken"].iloc[0], smoke_ever=a["smoken"].max(),
                     onset=(neg["age"].iloc[-1] + fp["age"].iloc[0]) / 2))
I = pd.DataFrame(rows)
I = I[(I["onset"] >= 40)].copy()
I["entry"] = I["pos"]
I["event"] = I["dead"].notna().astype(int)
I["exit"] = np.where(I["event"] == 1, I["dead"] - I["byear"], I["last"])
m = (I["event"] == 1) & (I["exit"] <= I["entry"])
I.loc[m, "exit"] = I.loc[m, "entry"] + 0.5
I = I[I["exit"] > I["entry"]]
I["grp"] = pd.cut(I["onset"], [4, 59.999, 74.999, 120], labels=["<60", "60-74", ">=75"])

one = I[I["last"] <= I["pos"]]
P("\nHRS incident analytic: n=%d; single interview after report: %d" % (len(I), len(one)))
P("  their age at report: median %.0f (IQR %.0f-%.0f); max %.0f" % (
    one["pos"].median(), one["pos"].quantile(.25), one["pos"].quantile(.75), one["pos"].max()))
P("  years from report to 2021: median %.0f (min %.0f, max %.0f)" % (
    (2021 - (one["pos"] + one["byear"])).median(),
    (2021 - (one["pos"] + one["byear"])).min(), (2021 - (one["pos"] + one["byear"])).max()))
P("  onset stratum: " + ", ".join("%s %d" % (k, v) for k, v in one["grp"].value_counts().items()))
P("  deaths among them: %d" % one["event"].sum())
P("  implied age in 2021 if alive: median %.0f, max %.0f" % (
    (2021 - one["byear"]).median(), (2021 - one["byear"]).max()))

P("\nHRS exit age: median %.0f, 99th %.0f, max %.0f; share above 100: %.2f%%" % (
    I["exit"].median(), I["exit"].quantile(.99), I["exit"].max(), 100 * (I["exit"] > 100).mean()))
P("HRS persons with exit > 110: %d" % (I["exit"] > 110).sum())

P("\nHRS smoking: first wave vs ever (max); disagreement %d of %d (%.1f%%)" % (
    ((I["smoke_first"] >= 1) != (I["smoke_ever"] >= 1)).sum(), len(I),
    100 * ((I["smoke_first"] >= 1) != (I["smoke_ever"] >= 1)).mean()))
P("  ever-smoker share: first wave %.1f%%, ever %.1f%%" % (
    100 * (I["smoke_first"] >= 1).mean(), 100 * (I["smoke_ever"] >= 1).mean()))

# refit with the two smoking definitions
from lifelines import CoxPHFitter
I["g74"] = (I["grp"] == "60-74").astype(float)
I["g75"] = (I["grp"] == ">=75").astype(float)
I["male"] = (h.groupby("pid")["ragender"].first().reindex(I["pid"]).values == 1).astype(float)
I["edu"] = pd.to_numeric(h.groupby("pid")["raeducl"].first().reindex(I["pid"]).values, errors="coerce")
I = I.dropna(subset=["edu"])
for lab, col in [("smoking at first wave", "smoke_first"), ("ever smoking (max wave)", "smoke_ever")]:
    X = I.copy()
    X["sm"] = (X[col] >= 1).astype(float)
    X = X[(X["exit"] <= 110)]
    mm = CoxPHFitter()
    mm.fit(X[["exit", "entry", "event", "g74", "g75", "male", "edu", "sm"]],
           duration_col="exit", event_col="event", entry_col="entry")
    a = mm.summary.loc["g74"]; b = mm.summary.loc["g75"]
    P("  %-28s 60-74 %.3f (%.3f-%.3f) | >=75 %.3f (%.3f-%.3f)" % (
        lab, a["exp(coef)"], a["exp(coef) lower 95%"], a["exp(coef) upper 95%"],
        b["exp(coef)"], b["exp(coef) lower 95%"], b["exp(coef) upper 95%"]))

io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("\nwrote", OUT)
