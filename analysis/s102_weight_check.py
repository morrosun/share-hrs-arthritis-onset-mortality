# -*- coding: utf-8 -*-
"""Step 102: why the maximum weight equals the 99th percentile in Table S8, panel A.

The table reports the maximum weight and the 99th percentile as the same value. That
can happen either because the weights take few distinct values (one covariate pattern
holding more than 1% of the cohort) or because of a transcription error. This script
rebuilds the availability model exactly as in s62 and reports the distribution.

Output: out/s102_weight_check.txt
"""
import os
import io, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, pyreadr, statsmodels.api as sm

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
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
    rows.append(dict(pid=pid, has_dg=int(not dg.empty),
                     onset=float(dg.iloc[0]) if not dg.empty else np.nan,
                     entry=x["age"].iloc[0], last=x["age"].iloc[-1],
                     dead=x["radyear"].max(), byear=x["rabyear"].iloc[0],
                     sex=x["ragender"].iloc[0], edu=x["raeducl"].iloc[0],
                     smoke=x["smoken"].iloc[0], first_wave=x["wave"].iloc[0]))
D = pd.DataFrame(rows)
D["event"] = D["dead"].notna().astype(int)
D["exit"] = np.where(D["event"] == 1, D["dead"] - D["byear"], D["last"])
_z = D["exit"] <= D["entry"]
D.loc[_z & (D["event"] == 1), "exit"] = D.loc[_z & (D["event"] == 1), "entry"] + 0.5
D = D[(D["exit"] > D["entry"]) & (D["entry"] >= 40) & (D["first_wave"] != 8)].copy()
D["female"] = (D["sex"] == 0).astype(float)
D["eb"] = pd.cut(D["entry"], [40, 55, 65, 75, 85, 110], labels=["40-54", "55-64", "65-74", "75-84", "85+"])
D["wave_g"] = D["first_wave"].replace({7: 8})
D["wave_c"] = D["wave_g"].astype("category")
X = pd.get_dummies(D[["eb", "wave_c"]], drop_first=True, dtype=float)
X["female"] = D["female"].values
X = sm.add_constant(X)
logit = sm.Logit(D["has_dg"].values, X).fit(disp=0)
D["p_avail"] = logit.predict(X)
D["ipw"] = np.where(D["has_dg"] == 1, 1.0 / D["p_avail"], np.nan)
A = D[D["has_dg"] == 1].copy()
A = A.dropna(subset=["edu", "smoke"])

P("analytic cohort: %d" % len(A))
w = A["ipw"].values
u = np.unique(np.round(w, 6))
P("distinct weight values: %d" % len(u))
P("max %.4f, 99th percentile %.4f, 99.5th %.4f"
  % (w.max(), np.percentile(w, 99), np.percentile(w, 99.5)))
P("respondents at the maximum weight: %d (%.2f%% of the cohort)"
  % ((np.round(w, 6) == u[-1]).sum(), 100 * (np.round(w, 6) == u[-1]).mean()))
P("trimming at the 99th percentile changes %d weights"
  % (w > np.percentile(w, 99)).sum())
P("top five weights and their counts:")
for v in u[-5:][::-1]:
    P("  %.4f  n=%d" % (v, (np.round(w, 6) == v).sum()))

io.open(f"{OUT}\\s102_weight_check.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("\nWrote out/s102_weight_check.txt")
