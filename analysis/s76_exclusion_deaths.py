# -*- coding: utf-8 -*-
"""Counts needed for the v13 text: deaths among HRS exclusions, and what the two-interview rule (rule B) removes."""
import os
import io, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, pyreadr

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUTDIR = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "s76_exclusion_deaths.txt")
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
L = []
def P(s=""):
    L.append(str(s)); print(s, flush=True)

# HRS
h = pyreadr.read_r(BASE + r"\hrs.RData")["hrs"].copy()
h = h.rename(columns={h.columns[0]: "pid"})
h["yr"] = h["wave"].map(WY_H); h["age"] = h["yr"] - h["rabyear"]
hh = h.dropna(subset=["age", "arthre"]).sort_values(["pid", "wave"])
recs = []
for pid, a in hh.groupby("pid"):
    fp = a[a["arthre"] == 1]
    rec = dict(pid=pid, entry=a["age"].iloc[0], last=a["age"].iloc[-1], dead=a["radyear"].max(),
               byear=a["rabyear"].iloc[0], edu=a["raeducl"].iloc[0])
    if fp.empty:
        rec.update(pattern="never", pos=np.nan)
    else:
        ip = a.index.get_loc(fp.index[0])
        neg = a.iloc[:ip]; neg = neg[neg["arthre"] == 0]
        rec.update(pattern="prevalent" if neg.empty else "incident", pos=fp["age"].iloc[0])
    recs.append(rec)
H = pd.DataFrame(recs)
H["event"] = H["dead"].notna().astype(int)
H["exit"] = np.where(H["event"] == 1, H["dead"] - H["byear"], H["last"])
H["edu_n"] = pd.to_numeric(H["edu"], errors="coerce")

P("HRS exclusions among never/prevalent (exit <= entry):")
for pat in ["never", "prevalent"]:
    q = H[(H["pattern"] == pat) & (H["exit"] <= H["entry"])]
    P("  %-10s n=%d deaths=%d (%.1f%%)" % (pat, len(q), q["event"].sum(), 100 * q["event"].mean()))
q = H[(H["pattern"].isin(["never", "prevalent"])) & (H["exit"] <= H["entry"])]
P("  total      n=%d deaths=%d" % (len(q), q["event"].sum()))
q2 = H[H["edu_n"].isna() & (H["exit"] > H["entry"])]
P("  education missing (included by follow-up): n=%d deaths=%d" % (len(q2), q2["event"].sum()))

# Rule B composition (>=2 interviews), HRS incident
I = H[(H["pattern"] == "incident") & (H["pos"].notna())].copy()
I["entry"] = I["pos"]
I = I[I["exit"] > I["entry"]]
one = I[I["last"] <= I["entry"]]
P("\nHRS incident, last interview = first positive report: n=%d deaths=%d" % (len(one), one["event"].sum()))

# SHARE
s = pyreadr.read_r(BASE + r"\share.RData")["share"].copy()
s = s.rename(columns={s.columns[0]: "pid"})
s["yr"] = s["wave"].map(WY_S); s["age"] = s["yr"] - s["rabyear"]
agg = s.groupby("pid").agg(entry=("age", "min"), last=("age", "max"), byear=("rabyear", "first"),
                           edu=("raeducl", "first"))
agg["onset"] = s.dropna(subset=["radiagarthr"]).groupby("pid")["radiagarthr"].first()
agg["dead"] = s.groupby("pid")["radyear"].max()
D = agg[agg["onset"].notna()].copy()
D = D[(D["onset"] >= 5) & D["byear"].notna() & (D["entry"] >= 40)]
D["event"] = D["dead"].notna().astype(int)
D["exit"] = np.where(D["event"] == 1, D["dead"] - D["byear"], D["last"])
D = D[(D["exit"] > D["entry"]) & (D["exit"] <= 110)]
D = D.dropna(subset=["edu"])
same = (D["event"] == 1) & (D["exit"] <= D["entry"])
D.loc[same, "exit"] = D.loc[same, "entry"] + 0.5
one = D[D["last"] <= D["entry"]]
P("SHARE analytic, last interview = entry interview: n=%d deaths=%d" % (len(one), one["event"].sum()))
P("SHARE analytic total: n=%d deaths=%d" % (len(D), D["event"].sum()))

io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT)
