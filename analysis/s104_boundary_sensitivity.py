# -*- coding: utf-8 -*-
"""Step 104: boundary deaths at the window's lower edge - sensitivity to their handling.

The window fits drop records whose exit equals their entry. At the lower edge of a
window these are deaths that occur exactly at 60 or 75 years: they contribute no
person-time inside the window, so the Cox fit excludes them. The main analysis takes
the opposite decision for a death in the same calendar year as entry, retaining it
with 0.5 years. This script asks whether the two rule-B estimates (and the two rule-A
estimates) change when the boundary deaths are retained with 0.5 years, the way
same-year deaths are.

For each rule and contrast it fits:
  (0) as reported            - boundary deaths excluded
  (1) boundary deaths kept   - exit set to lo + 0.5, entry at lo
  (2) entry moved to lo-0.5  - same people, entry half a year earlier
and counts the boundary deaths by group.

Output: out/s104_boundary_sensitivity.txt, out/tableS12_boundary.csv
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


h = pyreadr.read_r(os.path.join(BASE, "hrs.RData"))['hrs'].copy()
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

RULES = [('rule A', I), ('rule B', B)]
PAIRS = [('60-74', '<60', 60.0, 80.0), ('>=75', '60-74', 75.0, 95.0)]
out = []


def fit(D, a, b, lo, hi, mode):
    X = D[D['grp'].astype(str).isin([a, b])].copy()
    X['e'] = X['entry'].clip(lower=lo)
    X['x'] = X['exit'].clip(upper=hi)
    X['ev'] = ((X['event'] == 1) & (X['exit'] <= hi + 1e-9)).astype(int)
    if mode == 'keep':
        m = (X['ev'] == 1) & (X['exit'] == lo)
        X.loc[m, 'x'] = lo + 0.5
    elif mode == 'eps':
        # the death stays an event at the boundary, with an infinitesimal slice of
        # risk time inside the window: both groups are in the risk set at that age
        m = (X['ev'] == 1) & (X['exit'] == lo)
        X.loc[m, 'x'] = lo + 1e-6
    elif mode == 'earlier':
        m = (X['ev'] == 1) & (X['exit'] == lo)
        X.loc[m, 'e'] = lo - 0.5
    X = X[X['x'] > X['e']].copy()
    if X.empty:
        return None
    X['cmp'] = (X['grp'].astype(str) == a).astype(float)
    m = CoxPHFitter()
    m.fit(X[['e', 'x', 'ev', 'cmp', 'male', 'edu', 'smoke_ever']],
          duration_col='x', event_col='ev', entry_col='e')
    q = m.summary.loc['cmp']
    return dict(n=len(X), deaths=int(X['ev'].sum()), hr=float(q['exp(coef)']),
                lo=float(q['exp(coef) lower 95%']), hi=float(q['exp(coef) upper 95%']),
                na=int(X.loc[X['grp'].astype(str) == a, 'ev'].sum()),
                nb=int(X.loc[X['grp'].astype(str) == b, 'ev'].sum()))


for rname, D in RULES:
    P("=" * 86)
    P(f"# {rname}")
    P("=" * 86)
    for a, b, lo, hi in PAIRS:
        Dp = D[D['grp'].astype(str).isin([a, b])]
        bd_a = int(((Dp['event'] == 1) & (Dp['exit'] == lo) & (Dp['grp'].astype(str) == a)).sum())
        bd_b = int(((Dp['event'] == 1) & (Dp['exit'] == lo) & (Dp['grp'].astype(str) == b)).sum())
        P("")
        P(f"{a} vs {b}, window {lo:.0f}-{hi:.0f}: boundary deaths {bd_a} ({a}) + {bd_b} ({b})")
        P(f"  moving their entry to {lo-0.5:.1f} adds {0.5*bd_a:.1f} person-years below the window in "
          f"{a} and {0.5*bd_b:.1f} in {b}")
        for mode, lab in [('drop', 'as reported (excluded)'), ('eps', 'kept, epsilon inside window'),
                          ('keep', 'kept, +0.5 y inside window'),
                          ('earlier', 'kept, entry -0.5 y below window')]:
            r = fit(D, a, b, lo, hi, mode)
            if r is None:
                P(f"  {lab:34s} not estimable")
                continue
            P(f"  {lab:34s} n={r['n']:5,} deaths={r['deaths']:5,} "
              f"({r['na']}+{r['nb']})  HR {r['hr']:.4f} ({r['lo']:.3f}-{r['hi']:.3f})")
            out.append(dict(rule=rname, contrast=f"{a} vs {b}", handling=mode, n=r['n'],
                            deaths=r['deaths'], hr=r['hr'], lo=r['lo'], hi=r['hi'],
                            bd_a=bd_a, bd_b=bd_b))

pd.DataFrame(out).to_csv(os.path.join(OUT, "tableS12_boundary.csv"), index=False, encoding='utf-8-sig')
io.open(os.path.join(OUT, "s104_boundary_sensitivity.txt"), 'w', encoding='utf-8').write("\n".join(lines))
P("\nWrote out/s104_boundary_sensitivity.txt and out/tableS12_boundary.csv")
