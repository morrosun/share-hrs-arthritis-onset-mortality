# -*- coding: utf-8 -*-
"""Quantify the SHARE selection effect with inverse-probability weighting.

Among all respondents who ever reported arthritis (n=37,536 with follow-up and
entry age >= 40), availability of a usable diagnosis age is strongly related to
entry age, sex and entry wave. We model that availability, then re-fit the main
prevalent mortality model in the 18,530 with usable onset age, weighted by the
inverse probability of availability, and compare with the unweighted estimate.
"""
import io, os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "s62_selection_ipw.txt")
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
    if not (x["arthre"] == 1).any():
        continue
    dg = x["radiagarthr"].dropna()
    dg = dg[dg >= 5]
    rows.append(dict(
        pid=pid, has_dg=int(not dg.empty),
        onset=float(dg.iloc[0]) if not dg.empty else np.nan,
        entry=x["age"].iloc[0], last=x["age"].iloc[-1],
        dead=x["radyear"].max(), byear=x["rabyear"].iloc[0],
        sex=x["ragender"].iloc[0], edu=x["raeducl"].iloc[0],
        smoke=x["smoken"].iloc[0], first_wave=x["wave"].iloc[0],
    ))
D = pd.DataFrame(rows)
D["event"] = D["dead"].notna().astype(int)
D["exit"] = np.where(D["event"] == 1, D["dead"] - D["byear"], D["last"])
_zero = D["exit"] <= D["entry"]
D.loc[_zero & (D["event"] == 1), "exit"] = D.loc[_zero & (D["event"] == 1), "entry"] + 0.5
D = D[(D["exit"] > D["entry"]) & (D["entry"] >= 40) & (D["first_wave"] != 8)].copy()
D["female"] = (D["sex"] == 0).astype(float)

P("=" * 78)
P("# SHARE selection: availability model and IPW re-estimation")
P("=" * 78)
P("ever arthritis with follow-up and entry >= 40: %d (usable onset %d)" % (len(D), D["has_dg"].sum()))

# ---- availability model ----
D["eb"] = pd.cut(D["entry"], [40, 55, 65, 75, 85, 110], labels=["40-54", "55-64", "65-74", "75-84", "85+"])
D["wave_g"] = D["first_wave"].replace({7: 8})   # wave 7 enrolled 10 respondents; merge with wave 8
D["wave_c"] = D["wave_g"].astype("category")
X = pd.get_dummies(D[["eb", "wave_c"]], drop_first=True, dtype=float)
X["female"] = D["female"].values
X = sm.add_constant(X)
y = D["has_dg"].values
logit = sm.Logit(y, X).fit(disp=0)
P("\nAvailability model (logit), n=%d, pseudo R2=%.3f" % (len(D), logit.prsquared))
P("  entry-age band effects (ref 40-54):")
for c in X.columns:
    if c.startswith("eb_"):
        P("    %-8s OR %.2f (%.2f-%.2f)" % (c[3:], np.exp(logit.params[c]),
                                            np.exp(logit.conf_int().loc[c, 0]),
                                            np.exp(logit.conf_int().loc[c, 1])))
P("    female   OR %.2f (%.2f-%.2f)" % (np.exp(logit.params["female"]),
                                        np.exp(logit.conf_int().loc["female", 0]),
                                        np.exp(logit.conf_int().loc["female", 1])))
P("  wave effects:")
for c in X.columns:
    if c.startswith("wave_c_"):
        P("    wave %-4s OR %.2f" % (c[7:], np.exp(logit.params[c])))

D["p_avail"] = logit.predict(X)
D["ipw"] = np.where(D["has_dg"] == 1, 1.0 / D["p_avail"], np.nan)

# ---- main model in the analytic cohort ----
A = D[D["has_dg"] == 1].copy()

A["g74"] = ((A["onset"] >= 60) & (A["onset"] < 75)).astype(float)
A["g75"] = (A["onset"] >= 75).astype(float)
A["male"] = (A["sex"] == 1).astype(float)
A["smoke_ever"] = (A["smoke"] >= 1).astype(float)
A = A.dropna(subset=["edu", "smoke_ever", "g74", "g75"])

P("\nAnalytic cohort: %d, deaths %d" % (len(A), int(A["event"].sum())))
P("  IPW: mean %.3f, median %.3f, max %.3f" % (A["ipw"].mean(), A["ipw"].median(), A["ipw"].max()))
ess = A["ipw"].sum() ** 2 / (A["ipw"] ** 2).sum()
P("  effective sample size after weighting: %.0f" % ess)


def fit(weights=None, label=""):
    cols = ["exit", "entry", "event", "g74", "g75", "male", "edu", "smoke_ever"]
    m = CoxPHFitter()
    if weights is None:
        m.fit(A[cols], duration_col="exit", event_col="event", entry_col="entry")
    else:
        A2 = A.copy()
        A2["w"] = weights
        m.fit(A2[cols + ["w"]], duration_col="exit", event_col="event", entry_col="entry",
              weights_col="w", robust=True)
    P("\n[%s]" % label)
    for key, lab in [("g74", "60-74 vs <60"), ("g75", ">=75 vs <60")]:
        q = m.summary.loc[key]
        P("  %-14s HR %.2f (%.2f-%.2f)  p=%.3g" % (
            lab, q["exp(coef)"], q["exp(coef) lower 95%"], q["exp(coef) upper 95%"], q["p"]))
    return m


fit(None, "unweighted (main analysis)")
fit(A["ipw"].values, "IPW for availability of a diagnosis age")

# ---- sensitivity: trim extreme weights at the 99th percentile ----
cap = A["ipw"].quantile(0.99)
fit(A["ipw"].clip(upper=cap).values, "IPW, weights trimmed at 99th percentile (%.2f)" % cap)

io.open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("\nwrote", OUT)
