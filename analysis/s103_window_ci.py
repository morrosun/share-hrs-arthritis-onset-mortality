# -*- coding: utf-8 -*-
"""Step 103: decompose the 1.0407 vs 1.0416 gap in Table 5 (HRS 60-74 vs <60).

Table 5 reports the same contrast twice: once inside the shared window (4,394
respondents, 725 deaths, 1.0407) and once without restriction (4,927, 980, 1.0416).
The two confidence intervals coincide. The note attributes the near-identity to the
partial likelihood (events at ages where only one group is at risk carry no
information about the contrast), but the residual 0.001 in the point estimate has not
been attributed. This script separates the two boundary operations:

  (a) entry clipped at 60, exit clipped at 80   -> the windowed fit
  (b) entry not clipped, exit clipped at 80
  (c) entry clipped at 60, exit not clipped
  (d) neither clipped                           -> the unrestricted fit

Output: out/s103_window_ci.txt
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

A = I[(I['grp'].astype(str) == '60-74') | (I['grp'].astype(str) == '<60')].copy()
P(f"pair: n={len(A):,}  deaths={int(A['event'].sum()):,}")

SPECS = [('a. entry clipped 60, exit clipped 80', 60.0, 80.0),
         ('b. entry free, exit clipped 80', None, 80.0),
         ('c. entry clipped 60, exit free', 60.0, None),
         ('d. entry free, exit free', None, None)]
for lab, lo, hi in SPECS:
    X = A.copy()
    X['e'] = X['entry'] if lo is None else X['entry'].clip(lower=lo)
    X['x'] = X['exit'] if hi is None else X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= (hi if hi is not None else 1e9))).astype(int)
    X = X[X['x'] > X['e']].copy()
    X['cmp'] = (X['grp'].astype(str) == '60-74').astype(float)
    m = CoxPHFitter()
    m.fit(X[['e', 'x', 'ev', 'cmp', 'male', 'edu', 'smoke_ever']],
          duration_col='x', event_col='ev', entry_col='e')
    q = m.summary.loc['cmp']
    # events inside 60-80, and events at which both groups are at risk
    P(f"  {lab:42s} n={len(X):5,}  deaths={int(X['ev'].sum()):4,}  "
      f"HR {q['exp(coef)']:.4f} ({q['exp(coef) lower 95%']:.3f}-{q['exp(coef) upper 95%']:.3f})  "
      f"se {q['se(coef)']:.5f}")

io.open(f"{OUT}\\s103_window_ci.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("\nWrote out/s103_window_ci.txt")
