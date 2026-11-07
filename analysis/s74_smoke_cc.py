# -*- coding: utf-8 -*-
"""Complete-case smoking refit and exact counts for the corrected table notes."""
import os
import io, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, pyreadr
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "s74_smoke_cc.txt")
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
L = []
def P(s=""):
    L.append(str(s)); print(s, flush=True)

s = pyreadr.read_r(BASE + r"\share.RData")["share"].copy()
s = s.rename(columns={s.columns[0]: "pid"})
s["yr"] = s["wave"].map(WY_S)
s["age"] = s["yr"] - s["rabyear"]
agg = s.groupby("pid").agg(entry=("age", "min"), last=("age", "max"), sex=("ragender", "first"),
                           byear=("rabyear", "first"), edu=("raeducl", "first"),
                           smoke=("smoken", "max"))
agg["onset"] = s.dropna(subset=["radiagarthr"]).groupby("pid")["radiagarthr"].first()
agg["dead"] = s.groupby("pid")["radyear"].max()
D = agg[agg["onset"].notna()].copy()
D = D[(D["onset"] >= 5) & D["byear"].notna() & (D["entry"] >= 40)].copy()
D["event"] = D["dead"].notna().astype(int)
D["exit"] = np.where(D["event"] == 1, D["dead"] - D["byear"], D["last"])
D = D[(D["exit"] > D["entry"]) & (D["exit"] <= 110)]
D["grp"] = pd.cut(D["onset"], [4, 59.999, 74.999, 120], labels=["<60", "60-74", ">=75"])
D["g74"] = (D["grp"] == "60-74").astype(float)
D["g75"] = (D["grp"] == ">=75").astype(float)
D["male"] = (D["sex"] == 1).astype(float)
D["smoke_ever"] = (D["smoke"] >= 1).astype(float)
D["edu"] = pd.to_numeric(D["edu"], errors="coerce")
D = D.dropna(subset=["edu"])
P("SHARE analytic n=%d deaths=%d; missing smoking (all waves) %d" % (
    len(D), D["event"].sum(), D["smoke"].isna().sum()))

X = D.dropna(subset=["smoke"]).copy()
m = CoxPHFitter()
m.fit(X[["exit", "entry", "event", "g74", "g75", "male", "edu", "smoke_ever"]],
      duration_col="exit", event_col="event", entry_col="entry")
P("complete case: n=%d deaths=%d" % (len(X), int(X["event"].sum())))
for k in ["g74", "g75"]:
    q = m.summary.loc[k]
    P("  %s %.3f (%.3f-%.3f)" % (k, q["exp(coef)"], q["exp(coef) lower 95%"], q["exp(coef) upper 95%"]))

# IPW denominator: ever arthritis with follow-up from 40
E = agg[agg["onset"].notna()].copy()
E = E[E["byear"].notna() & (E["entry"] >= 40)]
E["event"] = E["dead"].notna().astype(int)
E["exit"] = np.where(E["event"] == 1, E["dead"] - E["byear"], E["last"])
E = E[E["exit"] > E["entry"]]
P("\never-arthritis with follow-up from 40 (IPW denominator): n=%d; missing smoking %d" % (
    len(E), E["smoke"].isna().sum()))

# missing smoking in the 27,264 cohort
P("all with usable onset (27,264) missing smoking: %d" %
  agg[(agg["onset"] >= 5) & agg["byear"].notna() & (agg["onset"].notna())]["smoke"].isna().sum())

io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT)
