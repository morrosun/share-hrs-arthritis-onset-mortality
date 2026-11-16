# -*- coding: utf-8 -*-
"""Independent audit of the v11 data and code (does not reuse s33/s68/s69 outputs).

Checks
  1  cohort construction, step by step, both cohorts
  2  HRS 36,529 -> 34,537 reconciliation
  3  smoking: missingness and the (x >= 1) coding of NaN
  4  death ascertainment beyond the last interview
  5  onset group boundaries for midpoint values
  6  four follow-up rules fitted under one consistent specification
"""
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
OUT = os.path.join(OUTDIR, "s71_data_code_audit.txt")
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}
END = 2021
L = []


def P(s=""):
    L.append(str(s))
    print(s, flush=True)


def grp_ok(x):
    """Boundaries used in the published main analysis (s33)."""
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=["<60", "60-74", ">=75"])


def grp_bug(x):
    """Boundaries used in s68/s69: right-closed, so 59.5 -> 60-74 and 74.5 -> >=75."""
    return pd.cut(x, [40, 59, 74, 120], labels=["<60", "60-74", ">=75"])


COLS = ["exit", "entry", "event", "g74", "g75", "male", "edu", "smoke_ever"]


def add_indicators(D):
    D = D.copy()
    D["g74"] = (D["grp"] == "60-74").astype(float)
    D["g75"] = (D["grp"] == ">=75").astype(float)
    D["male"] = (D["sex"] == 1).astype(float)
    D["smoke_ever"] = (D["smoke"] >= 1).astype(float)   # NaN >= 1 -> False (see part 3)
    D["edu"] = pd.to_numeric(D["edu"], errors="coerce")
    return D


def fit3(X, label):
    X = X[(X["exit"] <= 110)].copy()
    X = X[X["exit"] > X["entry"]]
    m = CoxPHFitter()
    m.fit(X[COLS], duration_col="exit", event_col="event", entry_col="entry")
    P("\n  [%s] n=%d deaths=%d py=%.0f" % (label, len(X), int(X["event"].sum()),
                                          (X["exit"] - X["entry"]).sum()))
    for k, lab in [("g74", "60-74 vs <60"), ("g75", ">=75 vs <60")]:
        q = m.summary.loc[k]
        P("    %-13s %.3f (%.3f-%.3f)" % (lab, q["exp(coef)"], q["exp(coef) lower 95%"],
                                          q["exp(coef) upper 95%"]))
    return X


# =====================================================================
P("=" * 88)
P("# 1. SHARE cohort construction")
P("=" * 88)
s = pyreadr.read_r(BASE + r"\share.RData")["share"].copy()
s = s.rename(columns={s.columns[0]: "pid"})
P("columns: %s" % list(s.columns))
P("rows=%d persons=%d" % (len(s), s["pid"].nunique()))
P("missing per key column: " + ", ".join(
    "%s=%d" % (c, s[c].isna().sum()) for c in ["rabyear", "radiagarthr", "radyear", "raeducl", "smoken"]))
P("radiagarthr dtype=%s; integer-valued: %s" % (
    s["radiagarthr"].dtype, bool((s["radiagarthr"].dropna() % 1 == 0).all())))
P("radiagarthr distribution of fractional parts: %s" %
  s["radiagarthr"].dropna().mod(1).value_counts().head(5).to_dict())

s["yr"] = s["wave"].map(WY_S)
s["age"] = s["yr"] - s["rabyear"]
P("unmapped waves: %s" % s.loc[s["yr"].isna(), "wave"].value_counts().to_dict())

agg = s.groupby("pid").agg(entry=("age", "min"), last=("age", "max"), nwave=("wave", "size"),
                           sex=("ragender", "first"), byear=("rabyear", "first"),
                           edu=("raeducl", "first"), smoke=("smoken", "max"),
                           smoke_first=("smoken", "first"))
agg["onset"] = s.dropna(subset=["radiagarthr"]).groupby("pid")["radiagarthr"].first()
agg["dead"] = s.groupby("pid")["radyear"].max()
P("\npersons with any radiagarthr: %d" % agg["onset"].notna().sum())
c = agg[agg["onset"].notna()].copy()
P("  onset >= 5 y:                    %d" % (c["onset"] >= 5).sum())
c = c[c["onset"] >= 5]
P("  + year of birth present:         %d" % c["byear"].notna().sum())
c = c[c["byear"].notna()]
c["event"] = c["dead"].notna().astype(int)
c["death_age"] = c["dead"] - c["byear"]
c["exit"] = np.where(c["event"] == 1, c["death_age"], c["last"])
P("  + entry age >= 40:               %d" % (c["entry"] >= 40).sum())
c = c[c["entry"] >= 40]
P("  + exit > entry (published rule): %d" % (c["exit"] > c["entry"]).sum())
P("  + exit <= 110:                   %d" % ((c["exit"] > c["entry"]) & (c["exit"] <= 110)).sum())
P("  deaths among published cohort:   %d" % c.loc[(c["exit"] > c["entry"]) & (c["exit"] <= 110), "event"].sum())
P("  with only one interview (last==entry): %d" % (c["last"] <= c["entry"]).sum())
P("  missing edu: %d; missing smoke(max): %d; missing smoke(first): %d" % (
    c["edu"].isna().sum(), c["smoke"].isna().sum(), c["smoke_first"].isna().sum()))
P("  smoke max vs first disagree: %d" % ((c["smoke"] >= 1) != (c["smoke_first"] >= 1)).sum())

# published cohort
S = c[(c["exit"] > c["entry"]) & (c["exit"] <= 110)].copy()
S["grp"] = grp_ok(S["onset"])
S = add_indicators(S).dropna(subset=["edu"])
P("\n  analytic (published): n=%d deaths=%d py=%.0f" % (len(S), S["event"].sum(),
                                                       (S["exit"] - S["entry"]).sum()))

P("\n" + "=" * 88)
P("# 2. HRS cohort construction and the 36,529 -> 34,537 gap")
P("=" * 88)
h = pyreadr.read_r(BASE + r"\hrs.RData")["hrs"].copy()
h = h.rename(columns={h.columns[0]: "pid"})
h["yr"] = h["wave"].map(WY_H)
h["age"] = h["yr"] - h["rabyear"]
P("rows=%d persons=%d; unmapped waves: %s" % (len(h), h["pid"].nunique(),
                                             h.loc[h["yr"].isna(), "wave"].value_counts().to_dict()))
hh = h.dropna(subset=["age", "arthre"]).sort_values(["pid", "wave"])

recs = []
for pid, a in hh.groupby("pid"):
    fp = a[a["arthre"] == 1]
    rec = dict(pid=pid, entry=a["age"].iloc[0], last=a["age"].iloc[-1], nwave=len(a),
               dead=a["radyear"].max(), byear=a["rabyear"].iloc[0], sex=a["ragender"].iloc[0],
               edu=a["raeducl"].iloc[0], smoke=a["smoken"].iloc[0])
    if fp.empty:
        rec.update(pattern="never", onset=np.nan, pos=np.nan)
    else:
        ip = a.index.get_loc(fp.index[0])
        neg = a.iloc[:ip]
        neg = neg[neg["arthre"] == 0]
        rec.update(pattern="prevalent" if neg.empty else "incident",
                   onset=(neg["age"].iloc[-1] + fp["age"].iloc[0]) / 2 if not neg.empty else np.nan,
                   pos=fp["age"].iloc[0])
    recs.append(rec)
H = pd.DataFrame(recs)
H["event"] = H["dead"].notna().astype(int)
H["exit"] = np.where(H["event"] == 1, H["dead"] - H["byear"], H["last"])
P("\npersons: %d  (never %d / prevalent %d / incident %d)" % (
    len(H), (H["pattern"] == "never").sum(), (H["pattern"] == "prevalent").sum(),
    (H["pattern"] == "incident").sum()))

H["ok_exit"] = H["exit"] > H["entry"]
H["ok_age"] = H["entry"] >= 40
H["ok_edu"] = pd.to_numeric(H["edu"], errors="coerce").notna()
P("dropped because exit<=entry: %d (deaths %d)" % ((~H["ok_exit"]).sum(), H.loc[~H["ok_exit"], "event"].sum()))
P("dropped because entry<40:    %d" % (H["ok_exit"] & ~H["ok_age"]).sum())
P("dropped because edu missing: %d" % (H["ok_exit"] & H["ok_age"] & ~H["ok_edu"]).sum())
base = H[H["ok_exit"] & H["ok_age"] & H["ok_edu"]]
P("=> base persons for the mortality model: %d deaths %d" % (len(base), base["event"].sum()))
for pat in ["never", "prevalent", "incident"]:
    d = base[base["pattern"] == pat]
    P("   %-10s %6d persons %6d deaths" % (pat, len(d), d["event"].sum()))

# incident analytic sample as published
I = base[(base["pattern"] == "incident") & (base["onset"] >= 40)].copy()
I["grp"] = grp_ok(I["onset"])
I["entry2"] = I["pos"]
fix = (I["exit"] <= I["entry2"]) & (I["event"] == 1)
I.loc[fix, "exit"] = I.loc[fix, "entry2"] + 0.5
I = I[I["exit"] > I["entry2"]]
I = add_indicators(I)
P("incident analytic (published): n=%d deaths=%d" % (len(I), I["event"].sum()))
P("  of which one interview after report: %d" % (I["last"] <= I["entry2"]).sum())

P("\n" + "=" * 88)
P("# 3. Smoking: missingness and the NaN coding")
P("=" * 88)
for nm, d in [("SHARE", S), ("HRS base", base)]:
    sm = pd.to_numeric(d["smoke"], errors="coerce")
    P("%-9s missing smoken=%d (%.2f%%); coded as never-smoker by (x>=1): %d; "
      "ever-smoker share with NaN as never: %.1f%%" % (
          nm, sm.isna().sum(), 100 * sm.isna().mean(), sm.isna().sum(),
          100 * (sm.fillna(0) >= 1).mean()))
P("NOTE: every script uses smoke_ever = (smoke >= 1), so NaN becomes 0 (never),")
P("      and dropna(subset=['smoke_ever']) never removes anyone. The 'complete-case")
P("      analysis for smoking' therefore cannot have removed missing-smoking records.")

P("\n" + "=" * 88)
P("# 4. Death ascertainment beyond the last interview")
P("=" * 88)
for nm, d in [("SHARE", S), ("HRS incident", I)]:
    d = d.copy()
    d["last_year"] = d["yr_last"] if "yr_last" in d.columns else np.nan
P("(a) survivors: years between last interview and the 2021 endpoint")
for nm, d, lastcol in [("SHARE", S, "last"), ("HRS", I, "last")]:
    pass
# rebuild last interview year explicitly
s_lastyr = s.groupby("pid")["yr"].max()
h_lastyr = hh.groupby("pid")["yr"].max()
S2 = S.join(s_lastyr.rename("last_yr"))
I2 = I.join(h_lastyr.rename("last_yr"))
for nm, d in [("SHARE", S2), ("HRS incident", I2)]:
    alive = d[d["event"] == 0]
    gap = END - alive["last_yr"]
    P("%-13s alive at last interview n=%d; median years to 2021=%.0f; <=2y %.1f%%; >=8y %.1f%%" % (
        nm, len(alive), gap.median(), 100 * (gap <= 2).mean(), 100 * (gap >= 8).mean()))
P("\n(b) deaths: gap between last interview year and death year")
for nm, d in [("SHARE", S2), ("HRS incident", I2)]:
    dead = d[d["event"] == 1].copy()
    dead["gap"] = dead["dead"] - dead["last_yr"]
    P("%-13s deaths n=%d; gap<0 %d (%.1f%%); gap=0 %d; gap 1-4 %d; gap>=5 %d; median %.0f" % (
        nm, len(dead), (dead["gap"] < 0).sum(), 100 * (dead["gap"] < 0).mean(),
        (dead["gap"] == 0).sum(), dead["gap"].between(1, 4).sum(), (dead["gap"] >= 5).sum(),
        dead["gap"].median()))
P("\n(c) by last-interview year: share of people last seen in that year who ever have a death record")
for nm, d in [("SHARE", S2), ("HRS incident", I2)]:
    P("  " + nm)
    for yr in sorted(d["last_yr"].dropna().unique()):
        if yr > 2016:
            continue
        q = d[d["last_yr"] == yr]
        if len(q) < 50:
            continue
        P("    last seen %d: n=%5d  died by 2021: %4d (%4.1f%%)  median age at last interview %.0f" % (
            yr, len(q), q["event"].sum(), 100 * q["event"].mean(), q["last"].median()))

P("\n" + "=" * 88)
P("# 5. Onset group boundaries for midpoint values")
P("=" * 88)
for nm, d in [("SHARE", S), ("HRS incident", I)]:
    frac = d["onset"].mod(1)
    in_59 = ((d["onset"] > 59) & (d["onset"] < 60)).sum()
    in_74 = ((d["onset"] > 74) & (d["onset"] < 75)).sum()
    mis = (grp_ok(d["onset"]).astype(str) != grp_bug(d["onset"]).astype(str)).sum()
    P("%-13s non-integer onsets %d (%.1f%%); in (59,60): %d; in (74,75): %d; "
      "misclassified by [40,59,74,120] bins: %d" % (
          nm, (frac > 0).sum(), 100 * (frac > 0).mean(), in_59, in_74, mis))

P("\n" + "=" * 88)
P("# 6. Four follow-up rules, one specification")
P("=" * 88)
P("groups [4,59.999,74.999,120]; exit capped at 110; same-year deaths at entry+0.5 in every rule;")
P("SHARE smoking = ever (max over waves), HRS = first wave, as in the published models.")

# SHARE
D = c.copy()
D["grp"] = grp_ok(D["onset"])
D = add_indicators(D).dropna(subset=["edu"])
D["exit_last"] = np.where(D["event"] == 1, D["death_age"], D["last"])
D["exit_2021"] = np.where(D["event"] == 1, D["death_age"], END - D["byear"])
for cname in ["exit_last", "exit_2021"]:
    same = (D["event"] == 1) & (D[cname] <= D["entry"])
    D.loc[same, cname] = D.loc[same, "entry"] + 0.5

P("\n--- SHARE prevalent")
for lab, mask, col in [
    ("R0 published main (same-year deaths dropped)", D["exit_last"] > D["entry"], "exit_last"),
    ("R1 last interview, same-year retained", D["exit_last"] > D["entry"], "exit_last"),
    ("R2 >=2 interviews, same-year retained", (D["exit_last"] > D["entry"]) & (D["last"] > D["entry"]), "exit_last"),
    ("R3 censor at 2021, same-year retained", D["exit_2021"] > D["entry"], "exit_2021"),
]:
    X = D[mask].copy()
    if lab.startswith("R0"):
        X = X[~((X["event"] == 1) & (X["death_age"] <= X["entry"]))]
    X["exit"] = X[col]
    fit3(X, lab)

# HRS
D2 = I.copy()
D2["exit_last"] = np.where(D2["event"] == 1, D2["dead"] - D2["byear"], D2["last"])
D2["exit_2021"] = np.where(D2["event"] == 1, D2["dead"] - D2["byear"], END - D2["byear"])
for cname in ["exit_last", "exit_2021"]:
    same = (D2["event"] == 1) & (D2[cname] <= D2["entry2"])
    D2.loc[same, cname] = D2.loc[same, "entry2"] + 0.5
D2["entry"] = D2["entry2"]

P("\n--- HRS incident")
for lab, mask, col in [
    ("R1 last interview, same-year retained (= published)", D2["exit_last"] > D2["entry"], "exit_last"),
    ("R2 >=2 interviews, same-year retained", (D2["exit_last"] > D2["entry"]) & (D2["last"] > D2["entry"]), "exit_last"),
    ("R3 censor at 2021, same-year retained", D2["exit_2021"] > D2["entry"], "exit_2021"),
]:
    X = D2[mask].copy()
    X["exit"] = X[col]
    fit3(X, lab)
    # pairwise >=75 vs <60 for comparison with Table 5
    Y = X[X["grp"].isin(["<60", ">=75"])].copy()
    Y["cmp"] = (Y["grp"] == ">=75").astype(float)
    m = CoxPHFitter()
    m.fit(Y[["exit", "entry", "event", "cmp", "male", "edu", "smoke_ever"]],
          duration_col="exit", event_col="event", entry_col="entry")
    q = m.summary.loc["cmp"]
    P("      pairwise >=75 vs <60: %.2f (%.2f-%.2f) n=%d deaths=%d" % (
        q["exp(coef)"], q["exp(coef) lower 95%"], q["exp(coef) upper 95%"],
        len(Y), int(Y["event"].sum())))

io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("\nwrote", OUT)
