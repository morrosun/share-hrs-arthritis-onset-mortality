# -*- coding: utf-8 -*-
"""Step 107: why retaining a boundary death as an event moves the adjusted estimate.

At the boundary age nobody in the older group is at risk (earliest entry 61 and 76
years), so the death carries no information about the contrast. It does enter the
covariate coefficients, however, and that is what shifts the adjusted group coefficient.
This script separates the two routes:

  drop    - boundary deaths removed (as reported)
  keep    - retained as events at boundary + 0.5 y, inside the window
  censor  - same people, same risk time, but censored instead of dying

If drop and censor agree, the movement comes from the event status; if keep and drop
agree once covariates are removed, the movement is transmitted through the adjustment.

Output: out/s107_boundary_mechanism.txt, out/_coxf_*.csv (for the R refit)
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd
from lifelines import CoxPHFitter

BASE = os.environ.get("AGING_DB_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
OUT = os.environ.get("AGING_DB_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
WY_H = {5: 2000, 6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012,
        12: 2014, 13: 2016, 14: 2018, 15: 2020}
lines = []


def P(s=""):
    lines.append(str(s))
    print(s, flush=True)


def grp(x):
    return pd.cut(x, [4, 59.999, 74.999, 120], labels=['<60', '60-74', '>=75'])


h = pyreadr.read_r(f"{BASE}\\hrs.RData")['hrs'].copy()
h = h.rename(columns={h.columns[0]: 'pid'})
h['yr'] = h['wave'].map(WY_H)
h['age'] = h['yr'] - h['rabyear']
h = h.dropna(subset=['age', 'arthre']).sort_values(['pid', 'wave'])

rows = []
for pid, x in h.groupby('pid'):
    fp = x[x['arthre'] == 1]
    if fp.empty:
        continue
    ip = x.index.get_loc(fp.index[0])
    neg = x.iloc[:ip]
    neg = neg[neg['arthre'] == 0]
    if neg.empty:
        continue
    rows.append(dict(pid=pid, lastneg=neg['age'].iloc[-1], pos=fp['age'].iloc[0],
                     last=x['age'].iloc[-1], dead=x['radyear'].max(), byear=x['rabyear'].iloc[0],
                     sex=x['ragender'].iloc[0], edu=x['raeducl'].iloc[0], smoke=x['smoken'].iloc[0],
                     poswave=fp['wave'].iloc[0], lastwave=x['wave'].max()))
I = pd.DataFrame(rows)
I['onset'] = (I['lastneg'] + I['pos']) / 2
I['event'] = (I['dead'].notna()).astype(int)
I['exit'] = np.where(I['event'] == 1, I['dead'] - I['byear'], I['last'])
I['male'] = (I['sex'] == 1).astype(float)
I['smoke_ever'] = (I['smoke'] >= 1).astype(float)
I['grp'] = grp(I['onset'])
I = I[(I['onset'] >= 40)].dropna(subset=['grp', 'edu'])
I['entry'] = I['pos']
fix = (I['exit'] <= I['entry']) & (I['event'] == 1)
I.loc[fix, 'exit'] = I.loc[fix, 'entry'] + 0.5
I = I[I['exit'] > I['entry']].copy()
B = I[I['lastwave'] > I['poswave']].copy()

PAIRS = {'c1': ('60-74', '<60', 60.0, 80.0), 'c2': ('>=75', '60-74', 75.0, 95.0)}


def fit(X, a, covars):
    Z = X.copy()
    Z['cmp'] = (Z['grp'].astype(str) == a).astype(float)
    cols = ['e', 'x', 'ev', 'cmp'] + (['male', 'edu', 'smoke_ever'] if covars else [])
    m = CoxPHFitter()
    m.fit(Z[cols], duration_col='x', event_col='ev', entry_col='e')
    q = m.summary.loc['cmp']
    return float(q['exp(coef)']), float(m.summary.loc['cmp', 'se(coef)'])


for tag, (a, b, lo, hi) in PAIRS.items():
    X = B[B['grp'].astype(str).isin([a, b])].copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    bd = (X['ev'] == 1) & (X['exit'] == lo)
    P("=" * 74)
    P(f"{a} vs {b}, window {lo:.0f}-{hi:.0f} (rule B): {int(bd.sum())} deaths at {lo:.0f} years, "
      f"{int((X.loc[bd, 'grp'].astype(str) == a).sum())} of them in {a}")
    P("=" * 74)
    risk_o = X[(X['grp'].astype(str) == a) & (X['e'] <= lo) & (X['x'] > lo)]
    P(f"  older group in the risk set at {lo:.0f} years: {len(risk_o)} respondents")
    # is there any reference-group event time in (lo, lo+0.5]? if not, keeping those
    # people at risk for that half year cannot change the partial likelihood at all
    mid = X[(X['grp'].astype(str) == b) & (X['ev'] == 1) & (X['x'] > lo) & (X['x'] <= lo + 0.5)]
    P(f"  reference-group deaths at {lo:.0f} < age <= {lo+0.5:.1f}: {len(mid)}")
    nxt = X.loc[(X['grp'].astype(str) == b) & (X['ev'] == 1) & (X['x'] > lo), 'x'].min()
    P(f"  next reference-group death time after the boundary: {nxt:.2f}")
    # in the censoring variant those people stop at lo+0.5, i.e. before that event time,
    # so they cannot enter the denominator of any later event
    P(f"  of the {int(bd.sum())} boundary deaths, how many are still at risk at t={nxt:.2f} "
      f"if censored at {lo+0.5:.1f}: "
      f"{int(((X['exit'] == lo) & (lo + 0.5 >= nxt)).sum())}")
    variants = {}
    variants['drop'] = X[X['x'] > X['e']].copy()
    Z = X.copy(); Z.loc[bd, 'x'] = lo + 0.5
    variants['keep'] = Z[Z['x'] > Z['e']].copy()
    Z = X.copy(); Z.loc[bd, 'x'] = lo + 0.5; Z.loc[bd, 'ev'] = 0
    variants['censor'] = Z[Z['x'] > Z['e']].copy()
    for nm, D in variants.items():
        D.assign(cmp=(D['grp'].astype(str) == a).astype(int)).to_csv(
            f"{OUT}\\_coxf_{tag}_{nm}.csv", index=False)
    P("")
    P(f"  {'variant':8s} {'n':>6s} {'deaths':>7s} {'no covariates':>15s} {'adjusted':>13s}")
    for nm, D in variants.items():
        h0, _ = fit(D, a, False)
        h1, _ = fit(D, a, True)
        P(f"  {nm:8s} {len(D):6,} {int(D['ev'].sum()):7,} {h0:15.4f} {h1:13.4f}")
    P("")

io.open(f"{OUT}\\s107_boundary_mechanism.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("Wrote out/s107_boundary_mechanism.txt and out/_coxf_*.csv")
