# -*- coding: utf-8 -*-
"""Step 100: deaths by group in the fixed Table S7 panel D windows, by recruitment period.

The minimum-events rule marks a contrast not estimable when the reference group has
fewer than five deaths inside the window. Panel D reports only the total deaths in the
60-75 year window of the 60-74 versus <60 contrast (423, 51, none) and the 75-90 year
window of the >=75 versus 60-74 contrast (9 deaths, four in the reference group in the
middle period). Whether the middle-period 60-74 versus <60 cell also falls under the
rule depends on how many of its 51 deaths are in the <60 group, which the table does
not show. This script supplies that count for every period and both contrasts.

Output: out/s100_period_events.txt
"""
import os
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pyreadr, numpy as np, pandas as pd

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
                     poswave=fp['wave'].iloc[0], entry_wave=x['wave'].min(), lastwave=x['wave'].max()))
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

PERIODS = [('All entry waves', None),
           ('Early (waves 5-8)', [5, 6, 7, 8]),
           ('Middle (waves 9-11)', [9, 10, 11]),
           ('Late (waves 12-15)', [12, 13, 14, 15])]
CONTRASTS = [('60-74', '<60', 60, 75), ('>=75', '60-74', 75, 90)]

P("Table S7 panel D: respondents and deaths by group inside the fixed windows")
P("=" * 88)
for lab, ws in PERIODS:
    d = I if ws is None else I[I['entry_wave'].isin(ws)]
    P("")
    P(f"{lab}: n={len(d):,}")
    for a, b, lo, hi in CONTRASTS:
        X = d[d['grp'].astype(str).isin([a, b])].copy()
        X['ev'] = ((X['event'] == 1) & (X['exit'] >= lo) & (X['exit'] <= hi)).astype(int)
        X = X[(X['exit'].clip(upper=hi) > X['entry'].clip(lower=lo))]
        nd = X.groupby('grp', observed=True)['ev'].sum().to_dict()
        n_a, n_b = int((X['grp'].astype(str) == a).sum()), int((X['grp'].astype(str) == b).sum())
        P(f"  {a} vs {b}, window {lo}-{hi}: n={len(X):,} ({a} {n_a:,}, {b} {n_b:,}); "
          f"deaths={int(X['ev'].sum()):,} ({a} {nd.get(a,0)}, {b} {nd.get(b,0)})"
          f"  -> reference group {'BELOW' if nd.get(b,0) < 5 else 'at or above'} 5")

io.open(f"{OUT}\\s100_period_events.txt", 'w', encoding='utf-8').write("\n".join(lines))
P("\nWrote out/s100_period_events.txt")
