# -*- coding: utf-8 -*-
"""Impact of retaining same-year deaths among never/prevalent HRS respondents,
and documentation of the three-group vs pairwise HRS contrast."""
import os
import io, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, pyreadr
from lifelines import CoxTimeVaryingFitter, CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "s78_sameyear_impact.txt")
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
L = []
def P(s=""):
    L.append(str(s)); print(s, flush=True)

h = pyreadr.read_r(BASE + r"\hrs.RData")["hrs"].copy()
h = h.rename(columns={h.columns[0]: "pid"})
h["yr"] = h["wave"].map(WY_H); h["age"] = h["yr"] - h["rabyear"]
h = h.dropna(subset=["age", "arthre"]).sort_values(["pid", "wave"])

persons = []
for pid, a in h.groupby("pid"):
    fp = a[a["arthre"] == 1]
    base = dict(pid=pid, entry=a["age"].iloc[0], last=a["age"].iloc[-1], dead=a["radyear"].max(),
                byear=a["rabyear"].iloc[0], sex=a["ragender"].iloc[0], edu=a["raeducl"].iloc[0],
                smoke=a["smoken"].iloc[0])
    if fp.empty:
        persons.append(dict(base, pattern="never", onset=np.nan, pos=np.nan)); continue
    ip = a.index.get_loc(fp.index[0]); prior = a.iloc[:ip]
    neg = prior[prior["arthre"] == 0]
    pat = "prevalent" if neg.empty else "incident"
    mid = (neg["age"].iloc[-1] + fp["age"].iloc[0]) / 2 if pat == "incident" else np.nan
    persons.append(dict(base, pattern=pat, onset=mid, pos=fp["age"].iloc[0]))
P0 = pd.DataFrame(persons)
P0["event"] = P0["dead"].notna().astype(int)
P0["exit"] = np.where(P0["event"] == 1, P0["dead"] - P0["byear"], P0["last"])
P0 = P0[P0["exit"].notna()].copy()

def grp_of(x):
    return "<60" if x < 60 else ("60-74" if x < 75 else ">=75")

def build(P0, retain_same_year):
    rows = []
    for r in P0.itertuples():
        e0, ex = r.entry, r.exit
        if not np.isfinite(ex):
            continue
        zero = (r.event == 1) and (ex <= e0)
        if ex <= e0:
            if retain_same_year and zero:
                ex = e0 + 0.5
            else:
                continue
        base = dict(pid=r.pid, sex=r.sex, edu=r.edu, smoke=r.smoke)
        if r.pattern == "never":
            rows.append(dict(base, start=e0, stop=ex, event=r.event, state="never"))
        elif r.pattern == "prevalent":
            rows.append(dict(base, start=e0, stop=ex, event=r.event, state="prevalent"))
        else:
            ex_inc = max(ex, r.pos + 0.5) if r.event == 1 else ex
            if r.pos > e0:
                rows.append(dict(base, start=e0, stop=min(r.pos, ex), event=0, state="never"))
            rows.append(dict(base, start=max(r.pos, e0), stop=ex_inc, event=r.event, state="inc_" + grp_of(r.onset)))
    T = pd.DataFrame(rows)
    T = T[T["stop"] > T["start"]].reset_index(drop=True)
    T["male"] = (T["sex"] == 1).astype(float)
    T["smoke_ever"] = (T["smoke"] >= 1).astype(float)
    T["edu"] = pd.to_numeric(T["edu"], errors="coerce")
    T = T.dropna(subset=["edu", "male", "smoke_ever"]).reset_index(drop=True)
    T["d_prev"] = (T["state"] == "prevalent").astype(float)
    T["d_lt60"] = (T["state"] == "inc_<60").astype(float)
    T["d_6074"] = (T["state"] == "inc_60-74").astype(float)
    T["d_75"] = (T["state"] == "inc_>=75").astype(float)
    T["d_ever"] = (T["state"] != "never").astype(float)
    return T

def fit(T, form, keys, lab):
    D = T[["pid", "start", "stop", "event", "male", "edu", "smoke_ever"] + keys].copy()
    m = CoxTimeVaryingFitter()
    m.fit(D, id_col="pid", start_col="start", stop_col="stop", event_col="event", formula=form)
    P("\n--- %s: persons=%d events=%d ---" % (lab, D["pid"].nunique(), int(m.event_observed.sum())))
    for k in keys:
        q = m.summary.loc[k]
        P("  %-12s %.3f (%.3f-%.3f)" % (k, q["exp(coef)"], q["exp(coef) lower 95%"], q["exp(coef) upper 95%"]))
    return m

for retain in [False, True]:
    T = build(P0, retain)
    lab = "same-year deaths RETAINED" if retain else "same-year deaths EXCLUDED (current)"
    for st, x in T.groupby("state"):
        P("%-6s %-18s py=%9.0f events=%5d persons=%6d" % (
            "KEEP" if retain else "DROP", st, (x["stop"] - x["start"]).sum(), int(x["event"].sum()), x["pid"].nunique()))
    fit(T, "d_ever + male + edu + smoke_ever", ["d_ever"], lab + " | any vs never")
    fit(T, "d_prev + d_lt60 + d_6074 + d_75 + male + edu + smoke_ever",
        ["d_prev", "d_lt60", "d_6074", "d_75"], lab + " | five-state")
    P("")

# incident contrast: three-group vs pairwise, all ages
T = build(P0, False)
I = T[T["state"].str.startswith("inc_")].copy()
I = I.rename(columns={"start": "entry", "stop": "exit"})
I["g74"] = (I["state"] == "inc_60-74").astype(float)
I["g75"] = (I["state"] == "inc_>=75").astype(float)
m = CoxPHFitter()
m.fit(I[["entry", "exit", "event", "g74", "g75", "male", "edu", "smoke_ever"]],
      duration_col="exit", event_col="event", entry_col="entry")
P("HRS incident, THREE-GROUP model (reference <60):")
for k in ["g74", "g75"]:
    q = m.summary.loc[k]
    P("  %-4s %.3f (%.3f-%.3f)" % (k, q["exp(coef)"], q["exp(coef) lower 95%"], q["exp(coef) upper 95%"]))
q74, q75 = m.summary.loc["g74"], m.summary.loc["g75"]
P("  implied >=75 vs 60-74: %.3f" % (q75["exp(coef)"] / q74["exp(coef)"]))

Y = I[I["state"].isin(["inc_<60", "inc_>=75"])].copy()
Y["cmp"] = (Y["state"] == "inc_>=75").astype(float)
m2 = CoxPHFitter()
m2.fit(Y[["entry", "exit", "event", "cmp", "male", "edu", "smoke_ever"]],
       duration_col="exit", event_col="event", entry_col="entry")
q = m2.summary.loc["cmp"]
P("HRS incident, PAIRWISE >=75 vs <60: %.2f (%.2f-%.2f) n=%d deaths=%d" % (
    q["exp(coef)"], q["exp(coef) lower 95%"], q["exp(coef) upper 95%"], len(Y), int(Y["event"].sum())))

io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT)
