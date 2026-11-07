# -*- coding: utf-8 -*-
"""Step 97: independent re-check of the two 0.61 shared-window estimates under rule B.

The two point estimates are close (0.6086 and 0.6108 before rounding). This script
verifies that they are not a copy error by fitting each contrast three ways:

  (a) as in s94 -- group a as the indicator, adjusted for sex, education, smoking;
  (b) reversed -- group b as the indicator; the hazard ratio must be the reciprocal;
  (c) unadjusted -- the same cohort and window without covariates.

If (a)+(b) are reciprocal and (c) lies in the same region, the two values are genuine.

Outputs: out/s97_verify_ruleb.txt
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
I['event'] = I['dead'].notna().astype(int)
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

P("=" * 84)
P("# HRS incident cohort under rule B (a later interview exists)")
P("=" * 84)
P(f"n = {len(B):,}   deaths = {int(B['event'].sum()):,}")


def frame(d, a, b, lo, hi):
    X = d.copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    X = X[(X['grp'].astype(str).isin([a, b])) & (X['x'] > X['e'])].copy()
    return X


def fit(X, indicator, covars):
    Z = X.copy()
    Z['cmp'] = indicator
    cols = ['e', 'x', 'ev', 'cmp'] + covars
    m = CoxPHFitter()
    m.fit(Z[cols], duration_col='x', event_col='ev', entry_col='e')
    q = m.summary.loc['cmp']
    return float(q['exp(coef)']), float(q['exp(coef) lower 95%']), float(q['exp(coef) upper 95%'])


for a, b, lo, hi in [('60-74', '<60', 60.0, 80.0), ('>=75', '60-74', 75.0, 95.0)]:
    X = frame(B, a, b, lo, hi)
    d_a = int(X.loc[X['grp'].astype(str) == a, 'ev'].sum())
    d_b = int(X.loc[X['grp'].astype(str) == b, 'ev'].sum())
    P("")
    P(f"  {a} vs {b}   window {lo:.0f}-{hi:.0f}")
    P(f"    n={len(X):,}  deaths={d_a} ({a}) + {d_b} ({b}) = {d_a + d_b}")
    h1, l1, u1 = fit(X, (X['grp'].astype(str) == a).astype(float),
                     ['male', 'edu', 'smoke_ever'])
    h2, l2, u2 = fit(X, (X['grp'].astype(str) == b).astype(float),
                     ['male', 'edu', 'smoke_ever'])
    h3, l3, u3 = fit(X, (X['grp'].astype(str) == a).astype(float), [])
    P(f"    (a) adjusted, {a} as indicator : {h1:.6f} ({l1:.4f}-{u1:.4f})")
    P(f"    (b) adjusted, {b} as indicator : {h2:.6f} ({l2:.4f}-{u2:.4f})")
    P(f"        product of the two         : {h1*h2:.6f}   (must be 1)")
    P(f"    (c) unadjusted                 : {h3:.6f} ({l3:.4f}-{u3:.4f})")
    P(f"    1/(a) = {1/h1:.6f}   vs (b) = {h2:.6f}   |difference| = {abs(1/h1 - h2):.2e}")

io.open(f"{OUT}\\s97_verify_ruleb.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("\nWrote out/s97_verify_ruleb.txt")
