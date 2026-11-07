# -*- coding: utf-8 -*-
"""Reconcile the HRS incident-cohort counts 946 vs 1,328 and the SHARE 8,190.

Rebuilds the incident cohorts with the same rules as s31/s33 and cross-tabulates,
for every respondent, whether an interview occurred after the first positive report,
whether a death was recorded, and which filter removed them.
"""
import io, os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pyreadr

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_S = {1: 2004, 2: 2007, 4: 2011, 5: 2013, 6: 2015, 7: 2017, 8: 2020}
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}

lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def build(df, wave_map, sex_col="ragender"):
    df = df.copy()
    df["yr"] = df["wave"].map(wave_map)
    df["age"] = df["yr"] - df["rabyear"]
    df = df.dropna(subset=["age", "arthre"]).sort_values(["pid", "wave"])
    rows = []
    for pid, x in df.groupby("pid"):
        fp = x[x["arthre"] == 1]
        if fp.empty:
            continue
        ip = x.index.get_loc(fp.index[0])
        prior = x.iloc[:ip]
        neg = prior[prior["arthre"] == 0]
        if neg.empty:
            continue
        rows.append(dict(
            pid=pid, pos=fp["age"].iloc[0], pos_wave=fp["wave"].iloc[0],
            last=x["age"].iloc[-1], last_wave=x["wave"].iloc[-1], n_waves=len(x),
            dead=x["radyear"].max(), byear=x["rabyear"].iloc[0],
            sex=x[sex_col].iloc[0], edu=x["raeducl"].iloc[0], smoke=x["smoken"].iloc[0],
            lastneg=neg["age"].iloc[-1],
        ))
    I = pd.DataFrame(rows)
    I["event"] = I["dead"].notna().astype(int)
    I["exit"] = np.where(I["event"] == 1, I["dead"] - I["byear"], I["last"])
    I["male"] = (I["sex"] == 1).astype(float)
    I["smoke_ever"] = (I["smoke"] >= 1).astype(float)
    I["mid"] = I["lastneg"] / 2 + I["pos"] / 2
    I["grp"] = pd.cut(I["mid"], [40, 59, 74, 120], labels=["<60", "60-74", ">=75"])
    return I


def reconcile(I, name):
    P("=" * 78)
    P("# " + name)
    P("=" * 78)
    P("all incident (negative then positive): %d" % len(I))
    P("  grp non-missing:                    %d" % I["grp"].notna().sum())
    P("  grp + edu non-missing (7,240-type): %d" % I.dropna(subset=["grp", "edu"]).shape[0])

    for label, D in [("all", I), ("grp+edu", I.dropna(subset=["grp", "edu"]))]:
        no_int = D["last"] <= D["pos"] + 1e-9
        P("")
        P("[%s] n=%d" % (label, len(D)))
        P("  last <= pos (no interview after first positive): %d" % no_int.sum())
        sub = D[no_int]
        same_yr_death = sub["event"].eq(1) & (sub["dead"] - sub["byear"] <= sub["pos"] + 0.999)
        later_death = sub["event"].eq(1) & ~same_yr_death
        alive = sub["event"].eq(0)
        P("    of these: died in the same year as first positive: %d" % same_yr_death.sum())
        P("              died later (death record after report)  : %d" % later_death.sum())
        P("              alive at last interview                 : %d" % alive.sum())
        P("    exit <= entry among them: %d" % (sub["exit"] <= sub["pos"] + 1e-9).sum())
        P("  last == pos exactly: %d" % (D["last"] == D["pos"]).sum())
        P("  last <  pos (impossible): %d" % (D["last"] < D["pos"]).sum())
        P("  with >=1 interview after first positive: %d" % (~no_int).sum())
        P("  died after first positive with a later interview: %d" % (
            (~no_int) & D["event"].eq(1)).sum())

    # where do the extra people come from?
    A = I.dropna(subset=["grp", "edu"])
    extra = I.loc[~I.index.isin(A.index)]
    P("")
    P("dropped by grp/edu filter: %d" % len(extra))
    if len(extra):
        P("  of these, last <= pos: %d" % (extra["last"] <= extra["pos"] + 1e-9).sum())
        P("  reasons: grp missing %d, edu missing %d" % (
            extra["grp"].isna().sum(), extra["edu"].isna().sum()))
        P("  grp distribution: " + str(extra["grp"].value_counts(dropna=False).to_dict()))

    # final v5 analytic set: grp+edu, exit>entry after 0.5-year retention
    B = A.copy()
    fix = (B["exit"] <= B["pos"]) & (B["event"] == 1)
    B.loc[fix, "exit"] = B.loc[fix, "pos"] + 0.5
    B = B[B["exit"] > B["pos"]]
    P("")
    P("v5 analytic (grp+edu, same-year deaths retained): %d, deaths %d" % (len(B), int(B["event"].sum())))
    no_int_B = B["last"] <= B["pos"] + 1e-9
    P("  of analytic set, last <= pos: %d" % no_int_B.sum())
    P("    died same year: %d; died later: %d; alive: %d" % (
        (no_int_B & (B["dead"] - B["byear"] <= B["pos"] + 0.999)).sum(),
        (no_int_B & B["event"].eq(1) & ~(B["dead"] - B["byear"] <= B["pos"] + 0.999)).sum(),
        (no_int_B & B["event"].eq(0)).sum()))


s = pyreadr.read_r(BASE + r"\share.RData")["share"].copy()
s = s.rename(columns={s.columns[0]: "pid"})
reconcile(build(s, WY_S), "SHARE incident")

h = pyreadr.read_r(BASE + r"\hrs.RData")["hrs"].copy()
h = h.rename(columns={h.columns[0]: "pid"})
reconcile(build(h, WY_H), "HRS incident")

io.open(os.path.join(OUT, "s61_reconcile_946.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\nwrote s61_reconcile_946.txt")
